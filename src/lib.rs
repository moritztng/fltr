use core::slice::from_raw_parts;
use memmap2::{MmapOptions, MmapRaw};
use rayon::prelude::*;
use std::{
    error::Error,
    fs::File,
    io::Write, path::Path, time::Instant,
};
use tokenizers::tokenizer::Tokenizer;

const vocab_size: usize = 32000;
const state_size: usize = 2048;
const n_layers: usize = 32;
const n_kv_heads: usize = 8;
const n_heads: usize = 32;
const head_size: usize = 128;
const hidden_dim: usize = 14336;
const dim: usize = head_size * n_heads;
const kv_dim: usize = head_size * n_kv_heads;
const q_group_size: usize = 64;
struct Quantized<A, B> {
    values: A,
    scales: B,
}
type QuantizedTensor<'a> = Quantized<&'a [i8], &'a [f32]>;
type QuantizedBuffer = Quantized<Vec<i8>, Vec<f32>>;
impl QuantizedBuffer {
    fn to_tensor(&self) -> QuantizedTensor<'_> {
        QuantizedTensor {
            values: &self.values,
            scales: &self.scales,
        }
    }
}

impl QuantizedTensor<'_> {
    fn from_ptr<const A: usize>(ptr: &mut *const u8) -> QuantizedTensor<'static> {
        unsafe {
            let tensor = QuantizedTensor {
                values: from_raw_parts(*ptr as *const i8, A),
                scales: from_raw_parts(ptr.add(A) as *const f32, A / q_group_size),
            };
            *ptr = ptr.add(A + 4 * (A / q_group_size));
            tensor
        }
    }
}

struct Layer {
    query: QuantizedTensor<'static>,
    key: QuantizedTensor<'static>,
    value: QuantizedTensor<'static>,
    heads: QuantizedTensor<'static>,
    rms_attention: &'static [f32],
    rms_feedforward: &'static [f32],
    ff1: QuantizedTensor<'static>,
    ff2: QuantizedTensor<'static>,
    swiglu: QuantizedTensor<'static>,
}

struct Weights {
    embeddings: Vec<f32>,
    layers: Vec<Layer>,
    rms_final: &'static [f32],
    output: QuantizedTensor<'static>,
}

pub struct Cache {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct Buffer {
    state: Vec<f32>,
    qstate: QuantizedBuffer,
    query: Vec<f32>,
    attention: Vec<f32>,
    swiglu: Vec<f32>,
    ff_hidden: Vec<f32>,
    qhidden: QuantizedBuffer,
    logits: Vec<f32>,
}

pub struct Model {
    state: Vec<f32>,
    buffer: Buffer,
    cache: Cache,
    weights: Weights,
    position: usize,
    tokenizer: Tokenizer,
    mmap: MmapRaw,
}

fn quantize(out: &mut QuantizedBuffer, x: &[f32]) {
    const q_max: f32 = 127f32;
    for ((out_group, x_group), scale) in out
        .values
        .chunks_exact_mut(q_group_size)
        .zip(x.chunks_exact(q_group_size))
        .zip(out.scales.iter_mut())
    {
        let group_max = x_group.iter().map(|x| x.abs()).reduce(f32::max).unwrap();
        *scale = group_max / q_max;
        for (out_x, x_x) in out_group.iter_mut().zip(x_group.iter()) {
            *out_x = (*x_x / *scale).round() as i8;
        }
    }
}

fn dequantize(out: &mut [f32], x: &QuantizedTensor) {
    for ((out_group, x_group), scale) in out
        .chunks_exact_mut(q_group_size)
        .zip(x.values.chunks_exact(q_group_size))
        .zip(x.scales.iter())
    {
        for (out_x, x_x) in out_group.iter_mut().zip(x_group.iter()) {
            *out_x = *x_x as f32 * scale;
        }
    }
}

fn matmul(out: &mut [f32], a: &QuantizedTensor, b: &QuantizedTensor) {
    out.par_iter_mut()
        .zip_eq(a.values.par_chunks_exact(b.values.len()))
        .zip_eq(a.scales.par_chunks_exact(b.values.len() / q_group_size))
        .for_each(|((out_x, a_row), a_row_scales)| {
            let mut x = 0f32;
            for (((a_row_group, b_group), a_row_scale), b_scale) in a_row
                .chunks_exact(q_group_size)
                .zip(b.values.chunks_exact(q_group_size))
                .zip(a_row_scales.iter())
                .zip(b.scales.iter())
            {
                let mut gx = 0i32;
                for (a_row_x, b_x) in a_row_group.iter().zip(b_group.iter()) {
                    gx += *a_row_x as i32 * *b_x as i32;
                }
                x += gx as f32 * a_row_scale * b_scale;
            }
            *out_x = x;
        })
}

fn smul(matrix: &mut [f32], scalar: f32) {
    for matrix_x in matrix.iter_mut() {
        *matrix_x *= scalar;
    }
}

fn softmax(x: &mut [f32]) {
    let max = *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let mut sum = 0f32;
    for x_x in x.iter_mut() {
        *x_x = (*x_x - max).exp();
        sum += *x_x;
    }

    for x_x in x.iter_mut() {
        *x_x /= sum;
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (a_x, b_x) in a.iter_mut().zip(b.iter()) {
        *a_x += b_x;
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], weights: &[f32]) {
    let mut rms = x.iter().fold(0f32, |acc, x| acc + x.powi(2));

    rms = 1f32 / (rms / dim as f32 + 1e-5).sqrt();
    for ((out_x, weights_x), x_x) in out.iter_mut().zip(weights.iter()).zip(x.iter()) {
        *out_x = weights_x * (rms * x_x);
    }
}

fn ptr_to_slice<const A: usize>(ptr: &mut *const u8) -> &'static [f32] {
    unsafe {
        let slice = from_raw_parts(*ptr as *const f32, A);
        *ptr = ptr.add(4 * A);
        slice
    }
}

impl Model {
    pub fn from_dir(path: &Path) -> Model {
        let mmap: MmapRaw = MmapOptions::new()
            .map_raw_read_only(&File::open(path.join("weights.bin")).unwrap())
            .unwrap();
        let mut weights_ptr = mmap.as_ptr() as *const u8;
        let rms_final = ptr_to_slice::<dim>(&mut weights_ptr);
        let qembeddings = QuantizedTensor::from_ptr::<{ vocab_size * dim }>(&mut weights_ptr);
        let mut embeddings = vec![0f32; vocab_size * dim];
        dequantize(&mut embeddings, &qembeddings);
        let output = QuantizedTensor::from_ptr::<{ vocab_size * dim }>(&mut weights_ptr);
        let mut layers = Vec::new();
        for _ in 0..32 {
            let rms_attention = ptr_to_slice::<dim>(&mut weights_ptr);
            let rms_feedforward = ptr_to_slice::<dim>(&mut weights_ptr);
            let query = QuantizedTensor::from_ptr::<{ dim * dim }>(&mut weights_ptr);
            let key =
                QuantizedTensor::from_ptr::<{ dim * n_kv_heads * head_size }>(&mut weights_ptr);
            let value =
                QuantizedTensor::from_ptr::<{ dim * n_kv_heads * head_size }>(&mut weights_ptr);
            let heads = QuantizedTensor::from_ptr::<{ dim * dim }>(&mut weights_ptr);
            let ff1 = QuantizedTensor::from_ptr::<{ dim * hidden_dim }>(&mut weights_ptr);
            let ff2 = QuantizedTensor::from_ptr::<{ hidden_dim * dim }>(&mut weights_ptr);
            let swiglu = QuantizedTensor::from_ptr::<{ dim * hidden_dim }>(&mut weights_ptr);

            layers.push(Layer {
                rms_attention,
                rms_feedforward,
                query,
                key,
                value,
                heads,
                ff1,
                ff2,
                swiglu,
            });
        }

        Model {
            state: vec![0f32; dim],
            buffer: Buffer {
                state: vec![0f32; dim],
                qstate: QuantizedBuffer {
                    values: vec![0i8; dim],
                    scales: vec![0f32; dim / q_group_size],
                },
                query: vec![0f32; dim],
                attention: vec![0f32; state_size],
                swiglu: vec![0f32; hidden_dim],
                ff_hidden: vec![0f32; hidden_dim],
                qhidden: QuantizedBuffer {
                    values: vec![0i8; hidden_dim],
                    scales: vec![0f32; hidden_dim / q_group_size],
                },
                logits: vec![0f32; vocab_size],
            },
            cache: Cache {
                key: vec![0f32; n_layers * state_size * kv_dim],
                value: vec![0f32; n_layers * state_size * kv_dim],
            },
            weights: Weights {
                embeddings,
                layers,
                rms_final,
                output,
            },
            position: 0,
            tokenizer: Tokenizer::from_file(path.join("tokenizer.json")).unwrap(),
            mmap: mmap,
        }
    }

    fn forward(&mut self, token: usize) {
        let pos = self.position;
        self.state
            .copy_from_slice(&self.weights.embeddings[token * dim..(token + 1) * dim]);

        for ((weights, layer_key_cache), layer_value_cache) in self
            .weights
            .layers
            .iter()
            .zip(self.cache.key.chunks_exact_mut(state_size * kv_dim))
            .zip(self.cache.value.chunks_exact_mut(state_size * kv_dim))
        {
            rmsnorm(&mut self.buffer.state, &self.state, weights.rms_attention);

            quantize(&mut self.buffer.qstate, &self.buffer.state);
            let mut qstate_tensor = self.buffer.qstate.to_tensor();

            matmul(&mut self.buffer.query, &weights.query, &qstate_tensor);
            let offset = pos * kv_dim;
            let key_cache = &mut layer_key_cache[offset..offset + kv_dim];
            let value_cache = &mut layer_value_cache[offset..offset + kv_dim];
            matmul(key_cache, &weights.key, &qstate_tensor);
            matmul(value_cache, &weights.value, &qstate_tensor);

            let mut fcrs = [0f32; head_size];
            let mut fcis = [0f32; head_size];
            for (i, (fcr, fci)) in fcrs.iter_mut().zip(fcis.iter_mut()).enumerate() {
                let frequency = 1f32 / 1000000f32.powf((i * 2) as f32 / head_size as f32);
                let value = pos as f32 * frequency;
                *fcr = value.cos();
                *fci = value.sin();
            }
            for query_head in self.buffer.query.chunks_exact_mut(head_size) {
                for ((query_pair, fcr), fci) in query_head
                    .chunks_exact_mut(2)
                    .zip(fcrs.iter())
                    .zip(fcis.iter())
                {
                    query_pair.copy_from_slice(&[
                        query_pair[0] * fcr - query_pair[1] * fci,
                        query_pair[0] * fci + query_pair[1] * fcr,
                    ]);
                }
            }
            for key_head in key_cache.chunks_exact_mut(head_size) {
                for ((key_pair, fcr), fci) in key_head
                    .chunks_exact_mut(2)
                    .zip(fcrs.iter())
                    .zip(fcis.iter())
                {
                    key_pair.copy_from_slice(&[
                        key_pair[0] * fcr - key_pair[1] * fci,
                        key_pair[0] * fci + key_pair[1] * fcr,
                    ]);
                }
            }
 
            self.buffer.state.fill(0f32);
            for (h, (state_head, query_head)) in self
                .buffer
                .state
                .chunks_exact_mut(head_size)
                .zip(self.buffer.query.chunks_exact(head_size))
                .enumerate()
            {
                let offset = h * n_kv_heads / n_heads * head_size;
                for (attention_x, pos_key_cache) in self.buffer.attention[0..=pos]
                    .iter_mut()
                    .zip(layer_key_cache.chunks_exact(kv_dim))
                {
                    let mut x = 0f32;
                    for (query_x, key_x) in query_head
                        .iter()
                        .zip(pos_key_cache[offset..offset + head_size].iter())
                    {
                        x += query_x * key_x
                    }
                    *attention_x = x;
                }
                smul(&mut self.buffer.attention, 1f32 / (head_size as f32).sqrt());
                softmax(&mut self.buffer.attention[..=pos]);
                for (attention_x, pos_value_cache) in self.buffer.attention[0..=pos]
                    .iter()
                    .zip(layer_value_cache.chunks_exact(kv_dim))
                {
                    for (state_x, value_x) in state_head
                        .iter_mut()
                        .zip(pos_value_cache[offset..offset + head_size].iter())
                    {
                        *state_x += *attention_x * *value_x;
                    }
                }
            }

            quantize(&mut self.buffer.qstate, &self.buffer.state);
            matmul(
                &mut self.buffer.state,
                &weights.heads,
                &self.buffer.qstate.to_tensor(),
            );
            add(&mut self.state, &self.buffer.state);

            rmsnorm(
                &mut self.buffer.state,
                &self.state,
                &weights.rms_feedforward,
            );

            quantize(&mut self.buffer.qstate, &self.buffer.state);
            qstate_tensor = self.buffer.qstate.to_tensor();
            matmul(&mut self.buffer.ff_hidden, &weights.ff1, &qstate_tensor);

            matmul(&mut self.buffer.swiglu, &weights.swiglu, &qstate_tensor);
            for (hidden_x, swiglu_x) in self
                .buffer
                .ff_hidden
                .iter_mut()
                .zip(self.buffer.swiglu.iter())
            {
                *hidden_x *= 1f32 / (1f32 + (-*hidden_x).exp());
                *hidden_x *= swiglu_x;
            }

            quantize(&mut self.buffer.qhidden, &self.buffer.ff_hidden);
            matmul(
                &mut self.buffer.state,
                &weights.ff2,
                &self.buffer.qhidden.to_tensor(),
            );
            add(&mut self.state, &self.buffer.state);
        }

        rmsnorm(&mut self.buffer.state, &self.state, &self.weights.rms_final);

        quantize(&mut self.buffer.qstate, &self.buffer.state);

        matmul(
            &mut self.buffer.logits,
            &self.weights.output,
            &self.buffer.qstate.to_tensor(),
        );
    }

    pub fn generate(
        &mut self,
        prompt: &String,
        steps: usize,
        print: bool,
        autostop: bool,
        cache: Option<&(usize, Cache)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut tokens = self
            .tokenizer
            .encode(prompt.to_owned(), cache.is_none())?
            .get_ids()
            .to_vec();
        let prompt_len = tokens.len();
        if let Some((position, cache)) = cache {
            self.position = *position;
            self.cache.key.copy_from_slice(&cache.key);
            self.cache.value.copy_from_slice(&cache.value);
        } else {
            self.position = 0;
        }
        let mut start = Instant::now();
        for i in 0..(prompt_len + steps) {
            self.forward(tokens[i] as usize);
            if print {
                print!(
                    "{}",
                    self.tokenizer
                        .id_to_token(tokens[i] as u32)
                        .ok_or("print token error")?
                        .replace("▁", " ")
                );
                std::io::stdout().flush()?;
            }
            if i == tokens.len() - 1 {
                let token = self
                    .buffer
                    .logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, logit1), (_, logit2)| logit1.total_cmp(&logit2))
                    .ok_or("max logits error")?
                    .0;
                tokens.push(token as u32);
                if autostop && token == 13 {
                    break;
                }
            }
            if i == 0 {
                start = Instant::now();
            }
            self.position += 1;
        }
        if print {
            println!(
                "\ntokens/sec: {}",
                (tokens.len() - 1) as f32 / start.elapsed().as_secs_f32()
            );
        }
        Ok(self.tokenizer.decode(&tokens[prompt_len..], false)?)
    }

    pub fn compile(&mut self, prompt: &String) -> (usize, Cache) {
        self.generate(prompt, 0, true, false, None).unwrap();
        (
            self.position,
            Cache {
                key: self.cache.key.to_vec(),
                value: self.cache.value.to_vec(),
            },
        )
    }
}
