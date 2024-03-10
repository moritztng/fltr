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
    gate: Option<QuantizedSlice<'static>>,
    experts: Vec<Expert>,
}

struct Weights {
    embeddings: Vec<f32>,
    layers: Vec<Layer>,
    rms_final: &'static [f32],
    output: QuantizedSlice<'static>,
}

struct Buffer {
    state: Vec<f32>,
    state2: Vec<f32>,
    qstate: QuantizedVector,
    qstate2: QuantizedVector,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
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

fn matmul(out: &mut [f32], a: &QuantizedSlice, b: &QuantizedSlice, dim: usize) {
    let b_dim = b.values.len() / dim;
    let mut out_t = vec![0f32; out.len()];
    #[cfg(feature = "cuda")]
    unsafe {
        cuda_matmul(
            out_t.as_mut_ptr(),
            a.values.as_ptr(),
            b.values.as_ptr(),
            a.scales.as_ptr(),
            b.scales.as_ptr(),
            (a.values.len() / dim) as u32,
            b_dim as u32,
            dim as u32,
            Q_GROUP_SIZE as u16,
        );
    };
    #[cfg(not(feature = "cuda"))]
    {
        let a_dim = a.values.len() / dim;
        out_t
            .par_chunks_exact_mut(a_dim)
            .zip_eq(b.values.par_chunks_exact(dim))
            .zip_eq(b.scales.par_chunks_exact(dim / Q_GROUP_SIZE))
            .for_each(|((out_column, b_column), b_column_scales)| {
                for ((out_x, a_row), a_row_scales) in out_column
                    .iter_mut()
                    .zip(a.values.chunks_exact(dim))
                    .zip(a.scales.chunks_exact(dim / Q_GROUP_SIZE))
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
    transpose(out, &out_t, b_dim);
}

fn smul(matrix: &mut [f32], scalar: f32) {
    for matrix_x in matrix.iter_mut() {
        *matrix_x *= scalar;
    }
}

fn softmax(x: &mut [f32], dim: usize) {
    for row in x.chunks_exact_mut(dim) {
        let max = *row.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        let mut sum = 0f32;
        for x_x in row.iter_mut() {
            *x_x = (*x_x - max).exp();
            sum += *x_x;
        }

        for x_x in row.iter_mut() {
            *x_x /= sum;
        }
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

fn print_debug(message: &str, print: bool) {
    if print {
        println!("{}", message);
    }
}

impl Buffer {
    fn new(n_tokens: usize) -> Buffer {
        Buffer {
            state: vec![0f32; n_tokens * DIM],
            state2: vec![0f32; n_tokens * DIM],
            qstate: QuantizedVector {
                values: vec![0i8; n_tokens * DIM],
                scales: vec![0f32; n_tokens * (DIM / Q_GROUP_SIZE)],
            },
            qstate2: QuantizedVector {
                values: vec![0i8; n_tokens * DIM],
                scales: vec![0f32; n_tokens * (DIM / Q_GROUP_SIZE)],
            },
            query: vec![0f32; n_tokens * DIM],
            key: vec![0f32; n_tokens * KV_DIM],
            value: vec![0f32; n_tokens * KV_DIM],
            swiglu: vec![0f32; n_tokens * HIDDEN_DIM],
            ff_hidden: vec![0f32; n_tokens * HIDDEN_DIM],
            qhidden: QuantizedVector {
                values: vec![0i8; n_tokens * HIDDEN_DIM],
                scales: vec![0f32; n_tokens * (HIDDEN_DIM / Q_GROUP_SIZE)],
            },
            expert_logits: vec![0f32; n_tokens * N_EXPERTS],
            logits: vec![0f32; n_tokens * VOCAB_SIZE],
        }
    }
}

impl Model {
    pub fn from_dir(path: &Path, multiple_experts: bool) -> Model {
        #[cfg(feature = "cuda")]
        unsafe {
            cuda_init()
        };
        let mmap: MmapRaw = MmapOptions::new()
            .map_raw_read_only(
                &File::open(path.join(format!(
                    "{}.bin",
                    if multiple_experts { "large" } else { "small" }
                )))
                .unwrap(),
            )
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
            let gate = if multiple_experts {
                Some(QuantizedSlice::from_ptr::<{ DIM * N_EXPERTS }>(
                    &mut weights_ptr,
                ))
            } else {
                None
            };
            let mut experts = Vec::new();
            for _ in 0..(if multiple_experts { N_EXPERTS } else { 1 }) {
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

    fn forward(&mut self, tokens: &[u32], prompt_lens: &[usize], buffer: &mut Buffer, print: bool) {
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

        for weights in self.weights.layers.iter() {
            print_debug("query key value", print);
            rmsnorm(
                &mut buffer.state2,
                &buffer.state,
                weights.rms_attention,
                DIM,
            );

            quantize(&mut buffer.qstate, &buffer.state2);
            let qstate_tensor = buffer.qstate.slice_full();
            matmul(&mut buffer.query, &qstate_tensor, &weights.query, DIM);
            matmul(&mut buffer.key, &qstate_tensor, &weights.key, DIM);
            matmul(&mut buffer.value, &qstate_tensor, &weights.value, DIM);

            print_debug("postional encoding", print);
            let mut i_prompt_query = 0;
            let mut i_prompt_key = 0;
            for prompt_len in prompt_lens.iter() {
                for ((p, token_query), token_key) in buffer.query
                    [i_prompt_query..i_prompt_query + prompt_len * DIM]
                    .chunks_exact_mut(DIM)
                    .enumerate()
                    .zip(
                        buffer.key[i_prompt_key..i_prompt_key + prompt_len * KV_DIM]
                            .chunks_exact_mut(KV_DIM),
                    )
                {
                    let mut fcrs = [0f32; HEAD_SIZE];
                    let mut fcis = [0f32; HEAD_SIZE];
                    for (i, (fcr, fci)) in fcrs.iter_mut().zip(fcis.iter_mut()).enumerate() {
                        let frequency = 1f32 / 1000000f32.powf((i * 2) as f32 / HEAD_SIZE as f32);
                        let value = p as f32 * frequency;
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
                    for key_head in token_key.chunks_exact_mut(HEAD_SIZE) {
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

            print_debug("attention", print);
            let mut i_prompt_query = 0;
            let mut i_prompt_kv = 0;
            for prompt_len in prompt_lens.iter() {
                for h in 0..N_HEADS {
                    let head_prompt_size = prompt_len * HEAD_SIZE;
                    let mut head_queries = vec![0f32; head_prompt_size];
                    let mut head_keys = vec![0f32; head_prompt_size];
                    let mut head_values = vec![0f32; head_prompt_size];
                    for (
                        (
                            (
                                ((head_queries_token, head_keys_token), head_values_token),
                                queries_token,
                            ),
                            keys_token,
                        ),
                        values_token,
                    ) in head_queries
                        .chunks_exact_mut(HEAD_SIZE)
                        .zip(head_keys.chunks_exact_mut(HEAD_SIZE))
                        .zip(head_values.chunks_exact_mut(HEAD_SIZE))
                        .zip(
                            buffer.query[i_prompt_query..i_prompt_query + prompt_len * DIM]
                                .chunks_exact(DIM),
                        )
                        .zip(
                            buffer.key[i_prompt_kv..i_prompt_kv + prompt_len * KV_DIM]
                                .chunks_exact(KV_DIM),
                        )
                        .zip(
                            buffer.value[i_prompt_kv..i_prompt_kv + prompt_len * KV_DIM]
                                .chunks_exact(KV_DIM),
                        )
                    {
                        let query_offset = h * HEAD_SIZE;
                        let kv_offset = h * N_KV_HEADS / N_HEADS * HEAD_SIZE;
                        head_queries_token.copy_from_slice(
                            &queries_token[query_offset..query_offset + HEAD_SIZE],
                        );
                        head_keys_token
                            .copy_from_slice(&keys_token[kv_offset..kv_offset + HEAD_SIZE]);
                        head_values_token
                            .copy_from_slice(&values_token[kv_offset..kv_offset + HEAD_SIZE]);
                    }
                    let mut attention_scores = vec![0f32; prompt_len * prompt_len];
                    let mut qhead_queries = QuantizedVector {
                        values: vec![0i8; head_queries.len()],
                        scales: vec![0f32; head_queries.len() / Q_GROUP_SIZE],
                    };
                    let mut qhead_keys = QuantizedVector {
                        values: vec![0i8; head_keys.len()],
                        scales: vec![0f32; head_keys.len() / Q_GROUP_SIZE],
                    };
                    quantize(&mut qhead_queries, &head_queries);
                    quantize(&mut qhead_keys, &head_keys);
                    matmul(
                        &mut attention_scores,
                        &qhead_queries.slice_full(),
                        &qhead_keys.slice_full(),
                        HEAD_SIZE,
                    );
                    smul(&mut attention_scores, 1f32 / (HEAD_SIZE as f32).sqrt());
                    for (i, attention_token) in
                        attention_scores.chunks_exact_mut(*prompt_len).enumerate()
                    {
                        attention_token[i + 1..].fill(f32::NEG_INFINITY);
                    }
                    softmax(&mut attention_scores, *prompt_len);
                    let padded_dim = (prompt_len + Q_GROUP_SIZE - 1) / Q_GROUP_SIZE * Q_GROUP_SIZE;
                    let mut attention_scores_padded = vec![0f32; prompt_len * padded_dim];
                    for (padded, unpadded) in attention_scores_padded
                        .chunks_exact_mut(padded_dim)
                        .zip(attention_scores.chunks_exact(*prompt_len))
                    {
                        padded[..unpadded.len()].copy_from_slice(unpadded);
                    }
                    let mut head_values_padded = vec![0f32; padded_dim * HEAD_SIZE];
                    head_values_padded[..head_values.len()].copy_from_slice(&head_values);
                    let mut head_values_padded_t = vec![0f32; head_values_padded.len()];
                    transpose(&mut head_values_padded_t, &head_values_padded, padded_dim);
                    let mut qattention_scores = QuantizedVector {
                        values: vec![0i8; attention_scores_padded.len()],
                        scales: vec![0f32; attention_scores_padded.len() / Q_GROUP_SIZE],
                    };
                    let mut qhead_values = QuantizedVector {
                        values: vec![0i8; head_values_padded.len()],
                        scales: vec![0f32; head_values_padded.len() / Q_GROUP_SIZE],
                    };
                    quantize(&mut qattention_scores, &attention_scores_padded);
                    quantize(&mut qhead_values, &head_values_padded_t);
                    matmul(
                        &mut head_values,
                        &qattention_scores.slice_full(),
                        &qhead_values.slice_full(),
                        padded_dim,
                    );
                    let offset = h * HEAD_SIZE;
                    for (state_token, head_value) in buffer.state2
                        [i_prompt_query..i_prompt_query + prompt_len * DIM]
                        .chunks_exact_mut(DIM)
                        .zip(head_values.chunks_exact(HEAD_SIZE))
                    {
                        state_token[offset..offset + HEAD_SIZE].copy_from_slice(head_value);
                    }
                }
                i_prompt_query += prompt_len * DIM;
                i_prompt_kv += prompt_len * KV_DIM;
            }

            print_debug("add attention", print);
            quantize(&mut buffer.qstate, &buffer.state2);
            matmul(
                &mut buffer.state2,
                &buffer.qstate.slice_full(),
                &weights.heads,
                DIM,
            );
            add(&mut buffer.state, &buffer.state2);

            print_debug("feed forward", print);
            rmsnorm(
                &mut buffer.state2,
                &buffer.state,
                &weights.rms_feedforward,
                DIM,
            );
            quantize(&mut buffer.qstate, &buffer.state2);
            if let Some(gate_weights) = &weights.gate {
                matmul(
                    &mut buffer.expert_logits,
                    &buffer.qstate.slice_full(),
                    gate_weights,
                    DIM,
                );
                let mut expert_tokens: [Vec<(usize, f32)>; 8] = Default::default();
                for (p, token_expert_logits) in
                    buffer.expert_logits.chunks_exact_mut(N_EXPERTS).enumerate()
                {
                    let mut indices_logits: Vec<_> =
                        token_expert_logits.iter().enumerate().collect();
                    indices_logits
                        .sort_unstable_by(|(_, logit1), (_, logit2)| logit2.total_cmp(logit1));
                    let (expert_indices, mut expert_weights): (Vec<_>, Vec<_>) =
                        indices_logits.into_iter().take(N_EXPERTS_PER_TOKEN).unzip();
                    let dim = expert_weights.len();
                    softmax(&mut expert_weights, dim);
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
                    matmul(expert_ff_hidden, &expert_qstate, &expert.ff1, DIM);
                    matmul(expert_swiglu, &expert_qstate, &expert.swiglu, DIM);
                    for (hidden_x, swiglu_x) in
                        expert_ff_hidden.iter_mut().zip(expert_swiglu.iter())
                    {
                        *hidden_x *= 1f32 / (1f32 + (-*hidden_x).exp());
                        *hidden_x *= swiglu_x;
                    }
                    quantize(&mut buffer.qhidden, &expert_ff_hidden);
                    matmul(
                        &mut buffer.state2[..n_tokens * DIM],
                        &buffer.qhidden.slice(0, n_tokens * HIDDEN_DIM),
                        &expert.ff2,
                        HIDDEN_DIM,
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
            } else {
                matmul(
                    &mut buffer.ff_hidden,
                    &buffer.qstate.slice_full(),
                    &weights.experts[0].ff1,
                    DIM,
                );
                matmul(
                    &mut buffer.swiglu,
                    &buffer.qstate.slice_full(),
                    &weights.experts[0].swiglu,
                    DIM,
                );
                for (hidden_x, swiglu_x) in buffer.ff_hidden.iter_mut().zip(buffer.swiglu.iter()) {
                    *hidden_x *= 1f32 / (1f32 + (-*hidden_x).exp());
                    *hidden_x *= swiglu_x;
                }
                quantize(&mut buffer.qhidden, &buffer.ff_hidden);
                matmul(
                    &mut buffer.state2,
                    &buffer.qhidden.slice_full(),
                    &weights.experts[0].ff2,
                    HIDDEN_DIM,
                );
                add(&mut buffer.state, &buffer.state2);
            }
        }

        print_debug("logits", print);
        rmsnorm(
            &mut buffer.state2,
            &buffer.state,
            &self.weights.rms_final,
            DIM,
        );

        quantize(&mut buffer.qstate, &buffer.state2);
        matmul(
            &mut buffer.logits,
            &buffer.qstate.slice_full(),
            &self.weights.output,
            DIM,
        );
    }

    pub fn generate(&mut self, prompts: &[String], print: bool) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut prompt_lens = Vec::new();
        for prompt in prompts {
            let prompt_tokens = self.tokenizer.encode(prompt.to_owned(), true).unwrap();
            tokens.extend(prompt_tokens.get_ids());
            prompt_lens.push(prompt_tokens.len());
        }

        let mut buffer = Buffer::new(tokens.len());
        let start_time = Instant::now();
        self.forward(&tokens, &prompt_lens, &mut buffer, print);
        if print {
            print!(
                "tokens/sec: {}\n{}",
                tokens.len() as f32 / start_time.elapsed().as_secs_f32(),
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
            let output_token = self.tokenizer.id_to_token(token).unwrap().replace("▁", "");
            if print {
                print!("{}", output_token);
                stdout().flush().unwrap();
            }
            output_tokens.push(output_token);
            i_prompt_logits += VOCAB_SIZE;
        }

        return output_tokens;
    }
}
