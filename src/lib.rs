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
    lengths: Vec<usize>,
}

struct Buffer {
    state: Vec<f32>,
    state2: Vec<f32>,
    qstate: QuantizedVector,
    qstate2: QuantizedVector,
    query: Vec<f32>,
    key_value: Vec<f32>,
    attention: Vec<f32>,
    swiglu: Vec<f32>,
    ff_hidden: Vec<f32>,
    qhidden: QuantizedVector,
    expert_logits: Vec<f32>,
    logits: Vec<f32>,
}

pub struct Model {
    weights: Weights,
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

fn transpose<A: Copy>(out: &mut [A], x: &[A], dim: usize) {
    let dim2 = x.len() / dim;
    for (x_column_i, out_row) in out.chunks_exact_mut(dim).enumerate() {
        for (out_x, x_row) in out_row.iter_mut().zip(x.chunks_exact(dim2)) {
            *out_x = x_row[x_column_i];
        }
    }
}

fn matmul<const N: usize, const B: usize>(out: &mut [f32], a: &QuantizedSlice, b: &QuantizedSlice) {
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
    {
        let batch_size = a.values.len() / N;
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
    }
    transpose(out, &out_t, B);
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
    fn new(batch_size: usize, max_prompt_len: usize) -> Buffer {
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
            key_value: vec![0f32; batch_size * KV_DIM],
            attention: vec![0f32; max_prompt_len],
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

fn cache_kv(cache: &mut [f32], kv: &[f32], cache_lens: &[usize], kv_lens: &[usize]) {
    let mut i_cache = 0;
    let mut i_kv = 0;
    for (cache_len, kv_len) in cache_lens.iter().zip(kv_lens) {
        i_cache += *cache_len * KV_DIM;
        cache[i_cache..i_cache + kv_len * KV_DIM]
            .copy_from_slice(&kv[i_kv..i_kv + kv_len * KV_DIM]);
        i_cache += kv_len * KV_DIM;
        i_kv += kv_len * KV_DIM;
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
            weights: Weights {
                embeddings,
                layers,
                rms_final,
                output,
            },
            tokenizer: Tokenizer::from_file(path.join("tokenizer.json")).unwrap(),
            mmap: mmap,
        }
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        prompt_lens: &[usize],
        buffer: &mut Buffer,
        cache: &mut Cache,
    ) {
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

        let layer_size = cache.key.len() / N_LAYERS;
        for ((weights, layer_key_cache), layer_value_cache) in self
            .weights
            .layers
            .iter()
            .zip(cache.key.chunks_exact_mut(layer_size))
            .zip(cache.value.chunks_exact_mut(layer_size))
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
            matmul::<DIM, KV_DIM>(&mut buffer.key_value, &qstate_tensor, &weights.key);
            cache_kv(
                layer_key_cache,
                &buffer.key_value,
                &cache.lengths,
                prompt_lens,
            );
            matmul::<DIM, KV_DIM>(&mut buffer.key_value, &qstate_tensor, &weights.value);
            cache_kv(
                layer_value_cache,
                &buffer.key_value,
                &cache.lengths,
                prompt_lens,
            );

            let mut i_prompt_query = 0;
            let mut i_prompt_key = 0;
            for (prompt_len, prompt_len_cache) in prompt_lens.iter().zip(cache.lengths.iter()) {
                i_prompt_key += prompt_len_cache * KV_DIM;
                for ((p, token_query), token_key_cache) in buffer.query
                    [i_prompt_query..i_prompt_query + prompt_len * DIM]
                    .chunks_exact_mut(DIM)
                    .enumerate()
                    .zip(
                        layer_key_cache[i_prompt_key..i_prompt_key + prompt_len * KV_DIM]
                            .chunks_exact_mut(KV_DIM),
                    )
                {
                    let mut fcrs = [0f32; HEAD_SIZE];
                    let mut fcis = [0f32; HEAD_SIZE];
                    for (i, (fcr, fci)) in fcrs.iter_mut().zip(fcis.iter_mut()).enumerate() {
                        let frequency = 1f32 / 1000000f32.powf((i * 2) as f32 / HEAD_SIZE as f32);
                        let value = (prompt_len_cache + p) as f32 * frequency;
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
                i_prompt_query += prompt_len * DIM;
                i_prompt_key += prompt_len * KV_DIM;
            }

            buffer.state2.fill(0f32);
            let mut i_prompt_state = 0;
            let mut i_prompt_cache = 0;
            for (prompt_len, prompt_len_cache) in prompt_lens.iter().zip(cache.lengths.iter()) {
                for ((p, token_state), token_query) in buffer.state2
                    [i_prompt_state..i_prompt_state + prompt_len * DIM]
                    .chunks_exact_mut(DIM)
                    .enumerate()
                    .zip(
                        buffer.query[i_prompt_state..i_prompt_state + prompt_len * DIM]
                            .chunks_exact(DIM),
                    )
                {
                    let pos = prompt_len_cache + p;
                    for (h, (state_head, query_head)) in token_state
                        .chunks_exact_mut(HEAD_SIZE)
                        .zip(token_query.chunks_exact(HEAD_SIZE))
                        .enumerate()
                    {
                        let offset = h * N_KV_HEADS / N_HEADS * HEAD_SIZE;
                        for (attention_x, pos_key_cache) in
                            buffer.attention[0..=pos].iter_mut().zip(
                                layer_key_cache[i_prompt_cache
                                    ..i_prompt_cache + (prompt_len_cache + prompt_len) * KV_DIM]
                                    .chunks_exact(KV_DIM),
                            )
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
                        for (attention_x, pos_value_cache) in buffer.attention[0..=pos].iter().zip(
                            layer_value_cache[i_prompt_cache
                                ..i_prompt_cache + (prompt_len_cache + prompt_len) * KV_DIM]
                                .chunks_exact(KV_DIM),
                        ) {
                            for (state_x, value_x) in state_head
                                .iter_mut()
                                .zip(pos_value_cache[offset..offset + HEAD_SIZE].iter())
                            {
                                *state_x += *attention_x * *value_x;
                            }
                        }
                    }
                }
                i_prompt_state += prompt_len * DIM;
                i_prompt_cache += (prompt_len_cache + prompt_len) * KV_DIM;
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

        for (prompt_len_cache, prompt_len) in cache.lengths.iter_mut().zip(prompt_lens) {
            *prompt_len_cache += prompt_len;
        }
    }

    pub fn generate(
        &mut self,
        prompts: &[String],
        steps: usize,
        print: bool,
        autostop: bool,
        cache: Option<&Cache>,
    ) -> (Cache, Vec<String>) {
        let mut tokens = Vec::new();
        let mut prompt_lens = Vec::new();
        for prompt in prompts {
            let prompt_tokens = self
                .tokenizer
                .encode(prompt.to_owned(), cache.is_none())
                .unwrap();
            tokens.extend(prompt_tokens.get_ids());
            prompt_lens.push(prompt_tokens.len());
        }

        let n_prompts = prompts.len();
        let n_tokens = tokens.len();
        let mut cache = if let Some(cache) = cache {
            let mut new_cache = Cache {
                key: vec![
                    0f32;
                    N_LAYERS * (n_tokens + n_prompts * (cache.lengths[0] + steps)) * KV_DIM
                ],
                value: vec![
                    0f32;
                    N_LAYERS * (n_tokens + n_prompts * (cache.lengths[0] + steps)) * KV_DIM
                ],
                lengths: vec![cache.lengths[0]; n_prompts],
            };
            let layer_size = cache.key.len() / N_LAYERS;
            let new_layer_size = new_cache.key.len() / N_LAYERS;
            for (((new_layer_key, new_layer_value), layer_key), layer_value) in new_cache
                .key
                .chunks_exact_mut(new_layer_size)
                .zip(new_cache.value.chunks_exact_mut(new_layer_size))
                .zip(cache.key.chunks_exact(layer_size))
                .zip(cache.value.chunks_exact(layer_size))
            {
                let mut i_prompt = 0;
                for prompt_len in prompt_lens.iter() {
                    new_layer_key[i_prompt..i_prompt + cache.lengths[0] * KV_DIM]
                        .copy_from_slice(&layer_key);
                    new_layer_value[i_prompt..i_prompt + cache.lengths[0] * KV_DIM]
                        .copy_from_slice(&layer_value);
                    i_prompt += (cache.lengths[0] + prompt_len + steps) * KV_DIM;
                }
            }
            new_cache
        } else {
            Cache {
                key: vec![0f32; N_LAYERS * (n_tokens + n_prompts * steps) * KV_DIM],
                value: vec![0f32; N_LAYERS * (n_tokens + n_prompts * steps) * KV_DIM],
                lengths: vec![0; n_prompts],
            }
        };

        let max_prompt_len = cache
            .lengths
            .iter()
            .zip(prompt_lens.iter())
            .map(|(x, y)| x + y)
            .max()
            .unwrap();
        let mut buffer = Buffer::new(n_tokens, max_prompt_len);
        let mut start_time = Instant::now();
        self.forward(&tokens, &prompt_lens, &mut buffer, &mut cache);
        if print {
            print!(
                "prompt prefill tokens/sec: {}\n{}",
                n_tokens as f32 / start_time.elapsed().as_secs_f32(),
                self.tokenizer.decode(&tokens, false).unwrap()
            );
            stdout().flush().unwrap();
        }
        let mut output_tokens = Vec::new();
        let mut i_prompt_logits = 0;
        for prompt_len in prompt_lens {
            i_prompt_logits += (prompt_len - 1) * VOCAB_SIZE;
            let token = buffer.logits[i_prompt_logits..i_prompt_logits + VOCAB_SIZE]
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
            i_prompt_logits += VOCAB_SIZE;
        }

        buffer = Buffer::new(n_prompts, max_prompt_len + steps);
        start_time = Instant::now();
        for _ in 0..steps {
            let last_tokens = &output_tokens[output_tokens.len() - n_prompts..];

            if autostop && last_tokens.contains(&2) {
                break;
            }

            self.forward(last_tokens, &vec![1; n_prompts], &mut buffer, &mut cache);
            for logits in buffer.logits.chunks_exact(VOCAB_SIZE) {
                let token = logits
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
        }

        if print {
            if steps > 0 {
                print!(
                    "\ndecode tokens/sec: {}",
                    (n_prompts * steps) as f32 / start_time.elapsed().as_secs_f32()
                );
            }
            print!("\n\n");
        }

        let mut output_tokens_t = vec![0; output_tokens.len()];
        transpose(&mut output_tokens_t, &output_tokens, steps + 1);
        let mut output_strings = Vec::new();
        for prompt_tokens in output_tokens_t.chunks_exact(steps + 1) {
            output_strings.push(self.tokenizer.decode(prompt_tokens, false).unwrap())
        }

        return (cache, output_strings);
    }
}
