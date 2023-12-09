use core::slice::from_raw_parts;
use memmap2::{MmapOptions, MmapRaw};
use std::{collections::VecDeque, error::Error, fs::File, io::Write};
use tokenizers::tokenizer::Tokenizer;

const vocab_size: usize = 32000;
const state_size: usize = 256;
const n_layers: usize = 6;
const n_kv_heads: usize = 6;
const n_heads: usize = 6;
const head_size: usize = 48;
const hidden_dim: usize = 768;
const dim: usize = head_size * n_heads;
const kv_dim: usize = head_size * n_kv_heads;
const q_group_size: usize = 32;

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
    qembeddings: QuantizedTensor<'static>,
    layers: Vec<Layer>,
    rms_final: &'static [f32],
}

struct Cache {
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
}

struct Model {
    state: Vec<f32>,
    buffer: Buffer,
    cache: Cache,
    weights: Weights,
}

fn quantize<const A: usize>(out: &mut QuantizedBuffer, x: &[f32]) {
    let n_groups = A / q_group_size;
    const q_max: f32 = 127f32;

    for group in 0..n_groups {
        let start = group * q_group_size;
        let x_max = x[start..start + q_group_size]
            .iter()
            .map(|x| x.abs())
            .reduce(f32::max)
            .unwrap();

        let scale = x_max / q_max;
        out.scales[group] = scale;

        for i in start..start + q_group_size {
            out.values[i] = (x[i] / scale).round() as i8;
        }
    }
}

fn dequantize<const A: usize>(out: &mut [f32], x: &QuantizedTensor) {
    for i in 0..A {
        out[i] = (x.values[i] as f32) * x.scales[i / q_group_size];
    }
}

fn matmul<const A: usize, const B: usize>(
    out: &mut [f32],
    a: &QuantizedTensor,
    b: &QuantizedTensor,
) {
    out[0..A].fill(0f32);
    for i in 0..A {
        let ib = i * B;
        for j in (0..=B - q_group_size).step_by(q_group_size) {
            let mut gx = 0i32;
            let ibj = ib + j;
            for k in 0..q_group_size {
                gx += a.values[ibj + k] as i32 * b.values[j + k] as i32;
            }
            out[i] += (gx as f32) * a.scales[ibj / q_group_size] * b.scales[j / q_group_size];
        }
    }
}

fn smul<const A: usize>(matrix: &mut [f32], scalar: f32) {
    for i in 0..A {
        matrix[i] *= scalar;
    }
}

fn softmax(x: &mut [f32], size: usize) {
    let mut max = x[0];
    for i in 0..size {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut sum = 0f32;
    for i in 0..size {
        x[i] = (x[i] - max).exp();
        sum += x[i];
    }

    for i in 0..size {
        x[i] /= sum;
    }
}

fn add<const A: usize>(a: &mut [f32], b: &[f32]) {
    for i in 0..A {
        a[i] += b[i];
    }
}

fn rmsnorm<const A: usize>(out: &mut [f32], x: &[f32], weights: &[f32]) {
    let mut rms = 0f32;
    for i in 0..A {
        rms += x[i].powi(2);
    }

    rms = 1f32 / ((rms / dim as f32) + 1e-5).sqrt();
    for i in 0..dim {
        out[i] = weights[i] * (rms * x[i]);
    }
}

impl Model {
    fn forward(&mut self, out: &mut [f32], token: usize, pos: usize) {
        self.state
            .copy_from_slice(&self.weights.embeddings[token * dim..(token + 1) * dim]);

        for l in 0..n_layers {
            rmsnorm::<dim>(
                &mut self.buffer.state,
                &self.state,
                &self.weights.layers[l].rms_attention,
            );

            quantize::<dim>(&mut self.buffer.qstate, &self.buffer.state);
            let mut qstate_tensor = self.buffer.qstate.to_tensor();

            matmul::<dim, dim>(
                &mut self.buffer.query,
                &self.weights.layers[l].query,
                &qstate_tensor,
            );
            let offset = l * state_size * kv_dim + pos * kv_dim;
            matmul::<kv_dim, dim>(
                &mut self.cache.key[offset..],
                &self.weights.layers[l].key,
                &qstate_tensor,
            );
            matmul::<kv_dim, dim>(
                &mut self.cache.value[offset..],
                &self.weights.layers[l].value,
                &qstate_tensor,
            );

            for i in (0..dim).step_by(2) {
                let head_i = i % head_size;
                let frequency = 1f32 / 10000f32.powf(head_i as f32 / head_size as f32);
                let value = pos as f32 * frequency;
                let fcr = value.cos();
                let fci = value.sin();
                let mut tmp = [self.buffer.query[i], self.buffer.query[i + 1]];
                self.buffer.query[i] = tmp[0] * fcr - tmp[1] * fci;
                self.buffer.query[i + 1] = tmp[0] * fci + tmp[1] * fcr;
                tmp.copy_from_slice(&self.cache.key[offset + i..offset + i + 2]);
                self.cache.key[offset + i] = tmp[0] * fcr - tmp[1] * fci;
                self.cache.key[offset + i + 1] = tmp[0] * fci + tmp[1] * fcr;
            }

            self.buffer.state.fill(0f32);
            for h in 0..n_heads {
                self.buffer.attention.fill(0f32);
                for p in 0..=pos {
                    for k in 0..head_size {
                        self.buffer.attention[p] += self.cache.key
                            [l * state_size * kv_dim + p * kv_dim + h * head_size + k]
                            * self.buffer.query[h * head_size + k];
                    }
                }
                smul::<state_size>(&mut self.buffer.attention, 1f32 / (head_size as f32).sqrt());
                softmax(&mut self.buffer.attention, pos + 1);
                for p in 0..=pos {
                    for i in 0..head_size {
                        self.buffer.state[h * head_size + i] += self.buffer.attention[p]
                            * &self.cache.value
                                [l * state_size * kv_dim + p * kv_dim + h * head_size + i]
                    }
                }
            }

            quantize::<dim>(&mut self.buffer.qstate, &self.buffer.state);
            matmul::<dim, dim>(
                &mut self.buffer.state,
                &self.weights.layers[l].heads,
                &self.buffer.qstate.to_tensor(),
            );
            add::<dim>(&mut self.state, &self.buffer.state);

            rmsnorm::<dim>(
                &mut self.buffer.state,
                &self.state,
                &self.weights.layers[l].rms_feedforward,
            );

            quantize::<dim>(&mut self.buffer.qstate, &self.buffer.state);
            qstate_tensor = self.buffer.qstate.to_tensor();
            matmul::<hidden_dim, dim>(
                &mut self.buffer.ff_hidden,
                &self.weights.layers[l].ff1,
                &qstate_tensor,
            );

            matmul::<hidden_dim, dim>(
                &mut self.buffer.swiglu,
                &self.weights.layers[l].swiglu,
                &qstate_tensor,
            );
            for i in 0..hidden_dim {
                let mut x = self.buffer.ff_hidden[i];
                x *= 1f32 / (1f32 + (-x).exp());
                x *= self.buffer.swiglu[i];
                self.buffer.ff_hidden[i] = x;
            }

            quantize::<hidden_dim>(&mut self.buffer.qhidden, &self.buffer.ff_hidden);
            matmul::<dim, hidden_dim>(
                &mut self.buffer.state,
                &self.weights.layers[l].ff2,
                &self.buffer.qhidden.to_tensor(),
            );
            add::<dim>(&mut self.state, &self.buffer.state);
        }

        rmsnorm::<dim>(&mut self.buffer.state, &self.state, &self.weights.rms_final);

        quantize::<dim>(&mut self.buffer.qstate, &self.buffer.state);
        matmul::<vocab_size, dim>(
            out,
            &self.weights.qembeddings,
            &self.buffer.qstate.to_tensor(),
        )
    }
}

fn init_quantized_tensors<const N: usize, const S: usize>(
    data: &mut *const i8,
) -> VecDeque<QuantizedTensor<'static>> {
    let mut tensors = VecDeque::new();
    for _ in 0..N {
        tensors.push_back(unsafe {
            let tensor = QuantizedTensor {
                values: from_raw_parts(*data, S),
                scales: from_raw_parts(data.add(S) as *const f32, S / q_group_size),
            };
            *data = data.add(S + 4 * (S / q_group_size));
            tensor
        });
    }
    tensors
}

pub fn generate(
    weights_pth: String,
    prompt: String,
    steps: usize,
    print: bool,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mmap: MmapRaw = MmapOptions::new()
        .offset(256)
        .map_raw_read_only(&File::open(weights_pth)?)?;
    let mut weights_ptr = mmap.as_ptr() as *const i8;
    let mut weights_ptr_f = weights_ptr as *const f32;

    let (rms_attention, rms_feedforward, rms_final) = unsafe {
        let rms_attention = from_raw_parts(weights_ptr_f, n_layers * dim);
        weights_ptr_f = weights_ptr_f.add(n_layers * dim);
        let rms_feedforward = from_raw_parts(weights_ptr_f, n_layers * dim);
        weights_ptr_f = weights_ptr_f.add(n_layers * dim);
        let rms_final = from_raw_parts(weights_ptr_f, dim);
        weights_ptr_f = weights_ptr_f.add(dim);
        (rms_attention, rms_feedforward, rms_final)
    };

    weights_ptr = weights_ptr_f as *const i8;
    let qembeddings = init_quantized_tensors::<1, { vocab_size * dim }>(&mut weights_ptr)
        .pop_back()
        .unwrap();
    let mut embeddings = vec![0f32; vocab_size * dim];
    dequantize::<{ vocab_size * dim }>(&mut embeddings, &qembeddings);
    let mut query = init_quantized_tensors::<n_layers, { dim * dim }>(&mut weights_ptr);
    let mut key =
        init_quantized_tensors::<n_layers, { dim * n_kv_heads * head_size }>(&mut weights_ptr);
    let mut value =
        init_quantized_tensors::<n_layers, { dim * n_kv_heads * head_size }>(&mut weights_ptr);
    let mut heads = init_quantized_tensors::<n_layers, { dim * dim }>(&mut weights_ptr);
    let mut ff1 = init_quantized_tensors::<n_layers, { dim * hidden_dim }>(&mut weights_ptr);
    let mut ff2 = init_quantized_tensors::<n_layers, { hidden_dim * dim }>(&mut weights_ptr);
    let mut swiglu = init_quantized_tensors::<n_layers, { dim * hidden_dim }>(&mut weights_ptr);

    let mut layers = Vec::new();
    for l in 0..n_layers {
        layers.push(Layer {
            rms_attention: &rms_attention[l * dim..],
            rms_feedforward: &rms_feedforward[l * dim..],
            query: query.pop_front().unwrap(),
            key: key.pop_front().unwrap(),
            value: value.pop_front().unwrap(),
            heads: heads.pop_front().unwrap(),
            ff1: ff1.pop_front().unwrap(),
            ff2: ff2.pop_front().unwrap(),
            swiglu: swiglu.pop_front().unwrap(),
        });
    }

    let mut model = Model {
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
        },
        cache: Cache {
            key: vec![0f32; n_layers * state_size * kv_dim],
            value: vec![0f32; n_layers * state_size * kv_dim],
        },
        weights: Weights {
            embeddings,
            qembeddings,
            layers,
            rms_final,
        },
    };

    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    let mut logits = vec![0f32; vocab_size];

    for pos in 0..steps {
        model.forward(&mut logits, tokens[pos] as usize, pos);

        if print {
            print!(
                "{}",
                tokenizer
                    .id_to_token(tokens[pos] as u32)
                    .ok_or("print token error")?
                    .replace("▁", " ")
            );
            std::io::stdout().flush()?;
        }

        if pos == tokens.len() - 1 {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, logit1), (_, logit2)| logit1.total_cmp(&logit2))
                .ok_or("max logits error")?
                .0;
            tokens.push(token as u32);
        }
    }
    Ok(tokenizer.decode(&tokens, false)?)
}
