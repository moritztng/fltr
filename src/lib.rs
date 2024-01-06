use core::slice::from_raw_parts;
use memmap2::{MmapOptions, MmapRaw};
use rayon::prelude::*;
use std::{fs::File, io::Write, path::Path, time::Instant};
use tokenizers::tokenizer::Tokenizer;

const VOCAB_SIZE: usize = 32000;
const STATE_SIZE: usize = 2048;
const N_LAYERS: usize = 32;
const N_KV_HEADS: usize = 8;
const N_HEADS: usize = 32;
const HEAD_SIZE: usize = 128;
const HIDDEN_DIM: usize = 14336;
const N_EXPERTS: usize = 8;
const N_EXPERTS_PER_TOKEN: usize = 2;
const DIM: usize = HEAD_SIZE * N_HEADS;
const KV_DIM: usize = HEAD_SIZE * N_KV_HEADS;
const Q_GROUP_SIZE: usize = 64;

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
                scales: from_raw_parts(ptr.add(A) as *const f32, A / Q_GROUP_SIZE),
            };
            *ptr = ptr.add(A + 4 * (A / Q_GROUP_SIZE));
            tensor
        }
    }
}

struct Expert {
    ff1: QuantizedTensor<'static>,
    ff2: QuantizedTensor<'static>,
    swiglu: QuantizedTensor<'static>,
}

struct Layer {
    query: QuantizedTensor<'static>,
    key: QuantizedTensor<'static>,
    value: QuantizedTensor<'static>,
    heads: QuantizedTensor<'static>,
    rms_attention: &'static [f32],
    rms_feedforward: &'static [f32],
    gate: QuantizedTensor<'static>,
    experts: Vec<Expert>,
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
    #[allow(dead_code)]
    mmap: MmapRaw,
}

fn quantize(out: &mut QuantizedBuffer, x: &[f32]) {
    const Q_MAX: f32 = 127f32;
    for ((out_group, x_group), scale) in out
        .values
        .chunks_exact_mut(Q_GROUP_SIZE)
        .zip(x.chunks_exact(Q_GROUP_SIZE))
        .zip(out.scales.iter_mut())
    {
        let group_max = x_group.iter().map(|x| x.abs()).reduce(f32::max).unwrap();
        *scale = group_max / Q_MAX;
        for (out_x, x_x) in out_group.iter_mut().zip(x_group.iter()) {
            *out_x = (*x_x / *scale).round() as i8;
        }
    }
}

fn dequantize(out: &mut [f32], x: &QuantizedTensor) {
    for ((out_group, x_group), scale) in out
        .chunks_exact_mut(Q_GROUP_SIZE)
        .zip(x.values.chunks_exact(Q_GROUP_SIZE))
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
        .zip_eq(a.scales.par_chunks_exact(b.values.len() / Q_GROUP_SIZE))
        .for_each(|((out_x, a_row), a_row_scales)| {
            let mut x = 0f32;
            for (((a_row_group, b_group), a_row_scale), b_scale) in a_row
                .chunks_exact(Q_GROUP_SIZE)
                .zip(b.values.chunks_exact(Q_GROUP_SIZE))
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

    rms = 1f32 / (rms / DIM as f32 + 1e-5).sqrt();
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
        let rms_final = ptr_to_slice::<DIM>(&mut weights_ptr);
        let qembeddings = QuantizedTensor::from_ptr::<{ VOCAB_SIZE * DIM }>(&mut weights_ptr);
        let mut embeddings = vec![0f32; VOCAB_SIZE * DIM];
        dequantize(&mut embeddings, &qembeddings);
        let output = QuantizedTensor::from_ptr::<{ VOCAB_SIZE * DIM }>(&mut weights_ptr);
        let mut layers = Vec::new();
        for _ in 0..32 {
            let rms_attention = ptr_to_slice::<DIM>(&mut weights_ptr);
            let rms_feedforward = ptr_to_slice::<DIM>(&mut weights_ptr);
            let query = QuantizedTensor::from_ptr::<{ DIM * DIM }>(&mut weights_ptr);
            let key =
                QuantizedTensor::from_ptr::<{ DIM * N_KV_HEADS * HEAD_SIZE }>(&mut weights_ptr);
            let value =
                QuantizedTensor::from_ptr::<{ DIM * N_KV_HEADS * HEAD_SIZE }>(&mut weights_ptr);
            let heads = QuantizedTensor::from_ptr::<{ DIM * DIM }>(&mut weights_ptr);
            let gate = QuantizedTensor::from_ptr::<{ DIM * N_EXPERTS }>(&mut weights_ptr);
            let mut experts = Vec::new();
            for _ in 0..N_EXPERTS {
                experts.push(Expert {
                    ff1: QuantizedTensor::from_ptr::<{ DIM * HIDDEN_DIM }>(&mut weights_ptr),
                    ff2: QuantizedTensor::from_ptr::<{ HIDDEN_DIM * DIM }>(&mut weights_ptr),
                    swiglu: QuantizedTensor::from_ptr::<{ DIM * HIDDEN_DIM }>(&mut weights_ptr),
                })
            }

            layers.push(Layer {
                rms_attention,
                rms_feedforward,
                query,
                key,
                value,
                heads,
                gate,
                experts,
            });
        }

        Model {
            state: vec![0f32; DIM],
            buffer: Buffer {
                state: vec![0f32; DIM],
                qstate: QuantizedBuffer {
                    values: vec![0i8; DIM],
                    scales: vec![0f32; DIM / Q_GROUP_SIZE],
                },
                query: vec![0f32; DIM],
                attention: vec![0f32; STATE_SIZE],
                swiglu: vec![0f32; HIDDEN_DIM],
                ff_hidden: vec![0f32; HIDDEN_DIM],
                qhidden: QuantizedBuffer {
                    values: vec![0i8; HIDDEN_DIM],
                    scales: vec![0f32; HIDDEN_DIM / Q_GROUP_SIZE],
                },
                logits: vec![0f32; VOCAB_SIZE],
            },
            cache: Cache {
                key: vec![0f32; N_LAYERS * STATE_SIZE * KV_DIM],
                value: vec![0f32; N_LAYERS * STATE_SIZE * KV_DIM],
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
            .copy_from_slice(&self.weights.embeddings[token * DIM..(token + 1) * DIM]);

        for ((weights, layer_key_cache), layer_value_cache) in self
            .weights
            .layers
            .iter()
            .zip(self.cache.key.chunks_exact_mut(STATE_SIZE * KV_DIM))
            .zip(self.cache.value.chunks_exact_mut(STATE_SIZE * KV_DIM))
        {
            rmsnorm(&mut self.buffer.state, &self.state, weights.rms_attention);
            
            quantize(&mut self.buffer.qstate, &self.buffer.state);
            let mut qstate_tensor = self.buffer.qstate.to_tensor();
            matmul(&mut self.buffer.query, &weights.query, &qstate_tensor);
            let offset = pos * KV_DIM;
            let key_cache = &mut layer_key_cache[offset..offset + KV_DIM];
            let value_cache = &mut layer_value_cache[offset..offset + KV_DIM];
            matmul(key_cache, &weights.key, &qstate_tensor);
            matmul(value_cache, &weights.value, &qstate_tensor);

            let mut fcrs = [0f32; HEAD_SIZE];
            let mut fcis = [0f32; HEAD_SIZE];
            for (i, (fcr, fci)) in fcrs.iter_mut().zip(fcis.iter_mut()).enumerate() {
                let frequency = 1f32 / 1000000f32.powf((i * 2) as f32 / HEAD_SIZE as f32);
                let value = pos as f32 * frequency;
                *fcr = value.cos();
                *fci = value.sin();
            }
            for query_head in self.buffer.query.chunks_exact_mut(HEAD_SIZE) {
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
            for key_head in key_cache.chunks_exact_mut(HEAD_SIZE) {
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
                .chunks_exact_mut(HEAD_SIZE)
                .zip(self.buffer.query.chunks_exact(HEAD_SIZE))
                .enumerate()
            {
                let offset = h * N_KV_HEADS / N_HEADS * HEAD_SIZE;
                for (attention_x, pos_key_cache) in self.buffer.attention[0..=pos]
                    .iter_mut()
                    .zip(layer_key_cache.chunks_exact(KV_DIM))
                {
                    let mut x = 0f32;
                    for (query_x, key_x) in query_head
                        .iter()
                        .zip(pos_key_cache[offset..offset + HEAD_SIZE].iter())
                    {
                        x += query_x * key_x
                    }
                    *attention_x = x;
                }
                smul(&mut self.buffer.attention, 1f32 / (HEAD_SIZE as f32).sqrt());
                softmax(&mut self.buffer.attention[..=pos]);
                for (attention_x, pos_value_cache) in self.buffer.attention[0..=pos]
                    .iter()
                    .zip(layer_value_cache.chunks_exact(KV_DIM))
                {
                    for (state_x, value_x) in state_head
                        .iter_mut()
                        .zip(pos_value_cache[offset..offset + HEAD_SIZE].iter())
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
            let mut expert_logits = [0f32; N_EXPERTS];
            matmul(&mut expert_logits, &weights.gate, &qstate_tensor);
            let mut indices_logits: Vec<_> = expert_logits.iter().enumerate().collect();
            indices_logits.sort_unstable_by(|(_, logit1), (_, logit2)| logit2.total_cmp(logit1));
            let (expert_indices, mut expert_weights): (Vec<_>, Vec<_>) =
                indices_logits.into_iter().take(N_EXPERTS_PER_TOKEN).unzip();
            softmax(&mut expert_weights);

            for (expert_index, expert_weight) in expert_indices.iter().zip(expert_weights) {
                let expert = &weights.experts[*expert_index];
                matmul(&mut self.buffer.ff_hidden, &expert.ff1, &qstate_tensor);
                matmul(&mut self.buffer.swiglu, &expert.swiglu, &qstate_tensor);
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
                    &expert.ff2,
                    &self.buffer.qhidden.to_tensor(),
                );
                smul(&mut self.buffer.state, expert_weight);
                add(&mut self.state, &self.buffer.state);
            }
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
    ) -> String {
        let mut tokens = self
            .tokenizer
            .encode(prompt.to_owned(), cache.is_none()).unwrap()
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
                        .unwrap()
                        .replace("▁", " ")
                );
                std::io::stdout().flush().unwrap();
            }
            if i == tokens.len() - 1 {
                let token = self
                    .buffer
                    .logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, logit1), (_, logit2)| logit1.total_cmp(&logit2))
                    .unwrap()
                    .0;
                tokens.push(token as u32);
                if autostop && token == 2 {
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
        self.tokenizer.decode(&tokens[prompt_len..], false).unwrap()
    }

    pub fn compile(&mut self, prompt: &String) -> (usize, Cache) {
        self.generate(prompt, 0, true, false, None);
        (
            self.position,
            Cache {
                key: self.cache.key.to_vec(),
                value: self.cache.value.to_vec(),
            },
        )
    }
}
