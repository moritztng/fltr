use core::slice::from_raw_parts;
use memmap2::{MmapOptions, MmapRaw};
#[cfg(not(feature = "cuda"))]
use rayon::prelude::*;
use std::{
    fs::File,
    io::{stdout, Write},
    path::Path,
    time::Instant,
};
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
type QuantizedSlice<'a> = Quantized<&'a [i8], &'a [f32]>;
type QuantizedSliceMut<'a> = Quantized<&'a mut [i8], &'a mut [f32]>;
type QuantizedVector = Quantized<Vec<i8>, Vec<f32>>;

impl QuantizedVector {
    fn slice_full(&self) -> QuantizedSlice<'_> {
        QuantizedSlice {
            values: &self.values,
            scales: &self.scales,
        }
    }
    fn slice(&self, start: usize, end: usize) -> QuantizedSlice<'_> {
        QuantizedSlice {
            values: &self.values[start..end],
            scales: &self.scales[start / Q_GROUP_SIZE..end / Q_GROUP_SIZE],
        }
    }
    fn slice_mut(&mut self, start: usize, end: usize) -> QuantizedSliceMut<'_> {
        QuantizedSliceMut {
            values: &mut self.values[start..end],
            scales: &mut self.scales[start / Q_GROUP_SIZE..end / Q_GROUP_SIZE],
        }
    }
}

impl QuantizedSlice<'_> {
    fn from_ptr<const A: usize>(ptr: &mut *const u8) -> QuantizedSlice<'static> {
        unsafe {
            let tensor = QuantizedSlice {
                values: from_raw_parts(*ptr as *const i8, A),
                scales: from_raw_parts(ptr.add(A) as *const f32, A / Q_GROUP_SIZE),
            };
            *ptr = ptr.add(A + 4 * (A / Q_GROUP_SIZE));
            tensor
        }
    }
}

struct Expert {
    ff1: QuantizedSlice<'static>,
    ff2: QuantizedSlice<'static>,
    swiglu: QuantizedSlice<'static>,
}

struct Layer {
    query: QuantizedSlice<'static>,
    key: QuantizedSlice<'static>,
    value: QuantizedSlice<'static>,
    heads: QuantizedSlice<'static>,
    rms_attention: &'static [f32],
    rms_feedforward: &'static [f32],
    gate: QuantizedSlice<'static>,
    experts: Vec<Expert>,
}

struct Weights {
    embeddings: Vec<f32>,
    layers: Vec<Layer>,
    rms_final: &'static [f32],
    output: QuantizedSlice<'static>,
}

pub struct Cache {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct Buffer {
    state: Vec<f32>,
    state2: Vec<f32>,
    qstate: QuantizedVector,
    qstate2: QuantizedVector,
    query: Vec<f32>,
    attention: Vec<f32>,
    swiglu: Vec<f32>,
    ff_hidden: Vec<f32>,
    qhidden: QuantizedVector,
    expert_logits: Vec<f32>,
    logits: Vec<f32>,
}

pub struct Model {
    cache: Cache,
    weights: Weights,
    position: usize,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    mmap: MmapRaw,
}

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_init();
    fn cuda_matmul(
        out: *const f32,
        a_h_quants: *const i8,
        b_h_quants: *const i8,
        a_h_scales: *const f32,
        b_h_scales: *const f32,
        a: u32,
        b: u32,
        n: u32,
        group_size: u16,
    );
}

fn quantize(out: &mut QuantizedVector, x: &[f32]) {
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

fn dequantize(out: &mut [f32], x: &QuantizedSlice) {
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

fn matmul<const N: usize, const B: usize>(out: &mut [f32], a: &QuantizedSlice, b: &QuantizedSlice) {
    let batch_size = a.values.len() / N;
    let mut out_t = vec![0f32; out.len()];
    #[cfg(feature = "cuda")]
    unsafe {
        cuda_matmul(
            out_t.as_mut_ptr(),
            a.values.as_ptr(),
            b.values.as_ptr(),
            a.scales.as_ptr(),
            b.scales.as_ptr(),
            (a.values.len() / N) as u32,
            B as u32,
            N as u32,
            Q_GROUP_SIZE as u16,
        );
    };
    #[cfg(not(feature = "cuda"))]
    out_t
        .par_chunks_exact_mut(batch_size)
        .zip_eq(b.values.par_chunks_exact(N))
        .zip_eq(b.scales.par_chunks_exact(N / Q_GROUP_SIZE))
        .for_each(|((out_column, b_column), b_column_scales)| {
            for ((out_x, a_row), a_row_scales) in out_column
                .iter_mut()
                .zip(a.values.chunks_exact(N))
                .zip(a.scales.chunks_exact(N / Q_GROUP_SIZE))
            {
                let mut x = 0f32;
                for (((a_group, b_group), a_scale), b_scale) in a_row
                    .chunks_exact(Q_GROUP_SIZE)
                    .zip(b_column.chunks_exact(Q_GROUP_SIZE))
                    .zip(a_row_scales.iter())
                    .zip(b_column_scales.iter())
                {
                    let mut gx = 0i32;
                    for (a_x, b_x) in a_group.iter().zip(b_group.iter()) {
                        gx += *a_x as i32 * *b_x as i32;
                    }
                    x += gx as f32 * a_scale * b_scale;
                }
                *out_x = x;
            }
        });

    for (out_t_column_i, out_row) in out.chunks_exact_mut(B).enumerate() {
        for (out_x, out_t_row) in out_row.iter_mut().zip(out_t.chunks_exact(batch_size)) {
            *out_x = out_t_row[out_t_column_i];
        }
    }
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

fn rmsnorm(out: &mut [f32], x: &[f32], weights: &[f32], dim: usize) {
    for (out, x) in out.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
        let mut rms = x.iter().fold(0f32, |acc, x| acc + x.powi(2));

        rms = 1f32 / (rms / dim as f32 + 1e-5).sqrt();
        for ((out_x, weights_x), x_x) in out.iter_mut().zip(weights.iter()).zip(x.iter()) {
            *out_x = weights_x * (rms * x_x);
        }
    }
}

fn ptr_to_slice<const A: usize>(ptr: &mut *const u8) -> &'static [f32] {
    unsafe {
        let slice = from_raw_parts(*ptr as *const f32, A);
        *ptr = ptr.add(4 * A);
        slice
    }
}

impl Buffer {
    fn new(batch_size: usize) -> Buffer {
        Buffer {
            state: vec![0f32; batch_size * DIM],
            state2: vec![0f32; batch_size * DIM],
            qstate: QuantizedVector {
                values: vec![0i8; batch_size * DIM],
                scales: vec![0f32; batch_size * (DIM / Q_GROUP_SIZE)],
            },
            qstate2: QuantizedVector {
                values: vec![0i8; batch_size * DIM],
                scales: vec![0f32; batch_size * (DIM / Q_GROUP_SIZE)],
            },
            query: vec![0f32; batch_size * DIM],
            attention: vec![0f32; STATE_SIZE],
            swiglu: vec![0f32; batch_size * HIDDEN_DIM],
            ff_hidden: vec![0f32; batch_size * HIDDEN_DIM],
            qhidden: QuantizedVector {
                values: vec![0i8; batch_size * HIDDEN_DIM],
                scales: vec![0f32; batch_size * (HIDDEN_DIM / Q_GROUP_SIZE)],
            },
            expert_logits: vec![0f32; batch_size * N_EXPERTS],
            logits: vec![0f32; batch_size * VOCAB_SIZE],
        }
    }
}

impl Model {
    pub fn from_dir(path: &Path) -> Model {
        #[cfg(feature = "cuda")]
        unsafe {
            cuda_init()
        };
        let mmap: MmapRaw = MmapOptions::new()
            .map_raw_read_only(&File::open(path.join("weights.bin")).unwrap())
            .unwrap();
        let mut weights_ptr = mmap.as_ptr() as *const u8;
        let rms_final = ptr_to_slice::<DIM>(&mut weights_ptr);
        let qembeddings = QuantizedSlice::from_ptr::<{ VOCAB_SIZE * DIM }>(&mut weights_ptr);
        let mut embeddings = vec![0f32; VOCAB_SIZE * DIM];
        dequantize(&mut embeddings, &qembeddings);
        let output = QuantizedSlice::from_ptr::<{ VOCAB_SIZE * DIM }>(&mut weights_ptr);
        let mut layers = Vec::new();
        for _ in 0..32 {
            let rms_attention = ptr_to_slice::<DIM>(&mut weights_ptr);
            let rms_feedforward = ptr_to_slice::<DIM>(&mut weights_ptr);
            let query = QuantizedSlice::from_ptr::<{ DIM * DIM }>(&mut weights_ptr);
            let key =
                QuantizedSlice::from_ptr::<{ DIM * N_KV_HEADS * HEAD_SIZE }>(&mut weights_ptr);
            let value =
                QuantizedSlice::from_ptr::<{ DIM * N_KV_HEADS * HEAD_SIZE }>(&mut weights_ptr);
            let heads = QuantizedSlice::from_ptr::<{ DIM * DIM }>(&mut weights_ptr);
            let gate = QuantizedSlice::from_ptr::<{ DIM * N_EXPERTS }>(&mut weights_ptr);
            let mut experts = Vec::new();
            for _ in 0..N_EXPERTS {
                experts.push(Expert {
                    ff1: QuantizedSlice::from_ptr::<{ DIM * HIDDEN_DIM }>(&mut weights_ptr),
                    ff2: QuantizedSlice::from_ptr::<{ HIDDEN_DIM * DIM }>(&mut weights_ptr),
                    swiglu: QuantizedSlice::from_ptr::<{ DIM * HIDDEN_DIM }>(&mut weights_ptr),
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

    fn forward(&mut self, tokens: &[u32], buffer: &mut Buffer) {
        let pos = self.position;
        let batch_size = tokens.len();

        for (state, token) in buffer
            .state
            .chunks_exact_mut(DIM)
            .zip(tokens.iter().map(|&t| t as usize))
        {
            state.copy_from_slice(
                &self
                    .weights
                    .embeddings
                    .chunks_exact(DIM)
                    .nth(token)
                    .unwrap(),
            );
        }

        for ((weights, layer_key_cache), layer_value_cache) in self
            .weights
            .layers
            .iter()
            .zip(self.cache.key.chunks_exact_mut(STATE_SIZE * KV_DIM))
            .zip(self.cache.value.chunks_exact_mut(STATE_SIZE * KV_DIM))
        {
            rmsnorm(
                &mut buffer.state2,
                &buffer.state,
                weights.rms_attention,
                DIM,
            );

            quantize(&mut buffer.qstate, &buffer.state2);
            let qstate_tensor = buffer.qstate.slice_full();
            matmul::<DIM, DIM>(&mut buffer.query, &qstate_tensor, &weights.query);
            let offset = pos * KV_DIM;
            let batch_kv_dim = batch_size * KV_DIM;
            let key_cache = &mut layer_key_cache[offset..offset + batch_kv_dim];
            let value_cache = &mut layer_value_cache[offset..offset + batch_kv_dim];
            matmul::<DIM, KV_DIM>(key_cache, &qstate_tensor, &weights.key);
            matmul::<DIM, KV_DIM>(value_cache, &qstate_tensor, &weights.value);
            for ((p, token_query), token_key_cache) in buffer
                .query
                .chunks_exact_mut(DIM)
                .enumerate()
                .zip(key_cache.chunks_exact_mut(KV_DIM))
            {
                let mut fcrs = [0f32; HEAD_SIZE];
                let mut fcis = [0f32; HEAD_SIZE];
                for (i, (fcr, fci)) in fcrs.iter_mut().zip(fcis.iter_mut()).enumerate() {
                    let frequency = 1f32 / 1000000f32.powf((i * 2) as f32 / HEAD_SIZE as f32);
                    let value = (pos + p) as f32 * frequency;
                    *fcr = value.cos();
                    *fci = value.sin();
                }
                for query_head in token_query.chunks_exact_mut(HEAD_SIZE) {
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
                for key_head in token_key_cache.chunks_exact_mut(HEAD_SIZE) {
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
            }

            buffer.state2.fill(0f32);
            for ((p, token_state), token_query) in buffer
                .state2
                .chunks_exact_mut(DIM)
                .enumerate()
                .zip(buffer.query.chunks_exact(DIM))
            {
                let pos = pos + p;
                for (h, (state_head, query_head)) in token_state
                    .chunks_exact_mut(HEAD_SIZE)
                    .zip(token_query.chunks_exact(HEAD_SIZE))
                    .enumerate()
                {
                    let offset = h * N_KV_HEADS / N_HEADS * HEAD_SIZE;
                    for (attention_x, pos_key_cache) in buffer.attention[0..=pos]
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
                    smul(&mut buffer.attention, 1f32 / (HEAD_SIZE as f32).sqrt());
                    softmax(&mut buffer.attention[..=pos]);
                    for (attention_x, pos_value_cache) in buffer.attention[0..=pos]
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
            }

            quantize(&mut buffer.qstate, &buffer.state2);
            matmul::<DIM, DIM>(
                &mut buffer.state2,
                &buffer.qstate.slice_full(),
                &weights.heads,
            );
            add(&mut buffer.state, &buffer.state2);

            rmsnorm(
                &mut buffer.state2,
                &buffer.state,
                &weights.rms_feedforward,
                DIM,
            );

            quantize(&mut buffer.qstate, &buffer.state2);
            matmul::<DIM, N_EXPERTS>(
                &mut buffer.expert_logits,
                &buffer.qstate.slice_full(),
                &weights.gate,
            );

            let mut expert_tokens: [Vec<(usize, f32)>; 8] = Default::default();
            for (p, token_expert_logits) in
                buffer.expert_logits.chunks_exact_mut(N_EXPERTS).enumerate()
            {
                let mut indices_logits: Vec<_> = token_expert_logits.iter().enumerate().collect();
                indices_logits
                    .sort_unstable_by(|(_, logit1), (_, logit2)| logit2.total_cmp(logit1));
                let (expert_indices, mut expert_weights): (Vec<_>, Vec<_>) =
                    indices_logits.into_iter().take(N_EXPERTS_PER_TOKEN).unzip();
                softmax(&mut expert_weights);
                for (expert_index, expert_weight) in
                    expert_indices.iter().zip(expert_weights.iter())
                {
                    expert_tokens[*expert_index].push((p, *expert_weight));
                }
            }

            for (expert_index, token_weights) in expert_tokens.iter().enumerate() {
                if token_weights.is_empty() {
                    continue;
                }

                let expert = &weights.experts[expert_index];
                let n_tokens = token_weights.len();
                let expert_qstate = buffer.qstate2.slice_mut(0, n_tokens * DIM);
                for ((state_values, state_scales), (token_index, _)) in expert_qstate
                    .values
                    .chunks_exact_mut(DIM)
                    .zip(expert_qstate.scales.chunks_exact_mut(DIM / Q_GROUP_SIZE))
                    .zip(token_weights.iter())
                {
                    state_values.copy_from_slice(
                        &buffer
                            .qstate
                            .values
                            .chunks_exact(DIM)
                            .nth(*token_index)
                            .unwrap(),
                    );
                    state_scales.copy_from_slice(
                        &buffer
                            .qstate
                            .scales
                            .chunks_exact(DIM / Q_GROUP_SIZE)
                            .nth(*token_index)
                            .unwrap(),
                    );
                }
                let expert_qstate = buffer.qstate2.slice(0, n_tokens * DIM);
                let expert_ff_hidden = &mut buffer.ff_hidden[..n_tokens * HIDDEN_DIM];
                let expert_swiglu = &mut buffer.swiglu[..n_tokens * HIDDEN_DIM];
                matmul::<DIM, HIDDEN_DIM>(expert_ff_hidden, &expert_qstate, &expert.ff1);
                matmul::<DIM, HIDDEN_DIM>(expert_swiglu, &expert_qstate, &expert.swiglu);
                for (hidden_x, swiglu_x) in expert_ff_hidden.iter_mut().zip(expert_swiglu.iter()) {
                    *hidden_x *= 1f32 / (1f32 + (-*hidden_x).exp());
                    *hidden_x *= swiglu_x;
                }
                quantize(&mut buffer.qhidden, &expert_ff_hidden);
                matmul::<HIDDEN_DIM, DIM>(
                    &mut buffer.state2[..n_tokens * DIM],
                    &buffer.qhidden.slice(0, n_tokens * HIDDEN_DIM),
                    &expert.ff2,
                );
                for (token_state, (token_index, weight)) in buffer.state2[..n_tokens * DIM]
                    .chunks_exact_mut(DIM)
                    .zip(token_weights.iter())
                {
                    smul(token_state, *weight);
                    add(
                        &mut buffer
                            .state
                            .chunks_exact_mut(DIM)
                            .nth(*token_index)
                            .unwrap(),
                        token_state,
                    );
                }
            }
        }

        rmsnorm(
            &mut buffer.state2,
            &buffer.state,
            &self.weights.rms_final,
            DIM,
        );

        quantize(&mut buffer.qstate, &buffer.state2);
        matmul::<DIM, VOCAB_SIZE>(
            &mut buffer.logits,
            &buffer.qstate.slice_full(),
            &self.weights.output,
        );

        self.position += batch_size;
    }

    pub fn generate(
        &mut self,
        prompt: &String,
        steps: usize,
        print: bool,
        autostop: bool,
        cache: Option<&(usize, Cache)>,
    ) -> String {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt.to_owned(), cache.is_none())
            .unwrap()
            .get_ids()
            .to_vec();
        let batch_size = prompt_tokens.len();

        if let Some((position, cache)) = cache {
            self.position = *position;
            self.cache.key.copy_from_slice(&cache.key);
            self.cache.value.copy_from_slice(&cache.value);
        } else {
            self.position = 0;
        }

        let mut buffer = Buffer::new(batch_size);
        let mut start_time = Instant::now();
        self.forward(&prompt_tokens, &mut buffer);
        if print {
            print!(
                "prompt prefill tokens/sec: {}\n{}",
                batch_size as f32 / start_time.elapsed().as_secs_f32(),
                self.tokenizer.decode(&prompt_tokens, false).unwrap()
            );
            stdout().flush().unwrap();
        }

        let mut token = buffer.logits[(batch_size - 1) * VOCAB_SIZE..]
            .iter()
            .enumerate()
            .max_by(|(_, logit1), (_, logit2)| logit1.total_cmp(&logit2))
            .unwrap()
            .0 as u32;
        if print {
            print!(
                "{}",
                self.tokenizer.id_to_token(token).unwrap().replace("▁", " ")
            );
            stdout().flush().unwrap();
        }
        let mut output_tokens = Vec::new();
        output_tokens.push(token);

        buffer = Buffer::new(1);

        start_time = Instant::now();
        for _ in 0..steps {
            if autostop && token == 2 {
                break;
            }

            self.forward(&[token], &mut buffer);
            token = buffer.logits[..VOCAB_SIZE]
                .iter()
                .enumerate()
                .max_by(|(_, logit1), (_, logit2)| logit1.total_cmp(&logit2))
                .unwrap()
                .0 as u32;
            if print {
                print!(
                    "{}",
                    self.tokenizer.id_to_token(token).unwrap().replace("▁", " ")
                );
                stdout().flush().unwrap();
            }
            output_tokens.push(token);
        }

        if print {
            if steps > 0 {
                print!(
                    "\ndecode tokens/sec: {}",
                    steps as f32 / start_time.elapsed().as_secs_f32()
                );
            }
            print!("\n\n");
        }

        self.tokenizer.decode(&output_tokens, false).unwrap()
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
