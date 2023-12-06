use core::slice::from_raw_parts;
use memmap2::{MmapOptions, MmapRaw};
use std::{error::Error, fs::File, io::Write};
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

struct Weights {
    embeddings: &'static [f32],
    query: &'static [f32],
    key: &'static [f32],
    value: &'static [f32],
    heads: &'static [f32],
    rms_attention: &'static [f32],
    rms_feedforward: &'static [f32],
    rms_final: &'static [f32],
    ff1: &'static [f32],
    ff2: &'static [f32],
    swiglu: &'static [f32],
}

struct Cache {
    key: Vec<f32>,
    value: Vec<f32>,
}

struct Buffer {
    state0: Vec<f32>,
    state1: Vec<f32>,
    query: Vec<f32>,
    attention: Vec<f32>,
    swiglu: Vec<f32>,
    ff_hidden: Vec<f32>,
}

struct Model {
    state: Vec<f32>,
    buffer: Buffer,
    cache: Cache,
    weights: Weights,
}

fn matmul<const A: usize, const B: usize, const C: usize>(out: &mut [f32], a: &[f32], b: &[f32]) {
    out[0..A * C].fill(0f32);
    for i in 0..A {
        for j in 0..B {
            for k in 0..C {
                out[i * C + k] += a[i * B + j] * b[j * C + k];
            }
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
                &mut self.buffer.state0,
                &self.state,
                &self.weights.rms_attention[l * dim..],
            );

            matmul::<dim, dim, 1>(
                &mut self.buffer.query,
                &self.weights.query[l * dim * dim..],
                &self.buffer.state0,
            );
            let offset = l * state_size * kv_dim + pos * kv_dim;
            matmul::<kv_dim, dim, 1>(
                &mut self.cache.key[offset..],
                &self.weights.key[l * dim * kv_dim..],
                &self.buffer.state0,
            );
            matmul::<kv_dim, dim, 1>(
                &mut self.cache.value[offset..],
                &self.weights.value[l * dim * kv_dim..],
                &self.buffer.state0,
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

            self.buffer.state1.fill(0f32);
            for h in 0..n_heads {
                for p in 0..=pos {
                    matmul::<1, head_size, 1>(
                        &mut self.buffer.attention[p..],
                        &self.cache.key[l * state_size * kv_dim + p * kv_dim + h * head_size..],
                        &self.buffer.query[h * head_size..],
                    );
                }
                smul::<state_size>(&mut self.buffer.attention, 1f32 / (head_size as f32).sqrt());
                softmax(&mut self.buffer.attention, pos + 1);
                for p in 0..=pos {
                    for i in 0..head_size {
                        self.buffer.state1[h * head_size + i] += self.buffer.attention[p]
                            * &self.cache.value
                                [l * state_size * kv_dim + p * kv_dim + h * head_size + i]
                    }
                }
            }

            matmul::<dim, dim, 1>(
                &mut self.buffer.state0,
                &self.weights.heads[l * dim * dim..],
                &self.buffer.state1,
            );
            add::<dim>(&mut self.state, &self.buffer.state0);

            rmsnorm::<dim>(
                &mut self.buffer.state0,
                &self.state,
                &self.weights.rms_feedforward[l * dim..],
            );

            matmul::<hidden_dim, dim, 1>(
                &mut self.buffer.ff_hidden,
                &self.weights.ff1[l * dim * hidden_dim..],
                &self.buffer.state0,
            );
            matmul::<hidden_dim, dim, 1>(
                &mut self.buffer.swiglu,
                &self.weights.swiglu[l * dim * hidden_dim..],
                &self.buffer.state0,
            );
            for i in 0..hidden_dim {
                let mut x = self.buffer.ff_hidden[i];
                x *= 1f32 / (1f32 + (-x).exp());
                x *= self.buffer.swiglu[i];
                self.buffer.ff_hidden[i] = x;
            }

            matmul::<dim, hidden_dim, 1>(
                &mut self.buffer.state0,
                &self.weights.ff2[l * hidden_dim * dim..],
                &self.buffer.ff_hidden,
            );
            add::<dim>(&mut self.state, &self.buffer.state0);
        }

        rmsnorm::<dim>(
            &mut self.buffer.state0,
            &self.state,
            &self.weights.rms_final,
        );
        matmul::<vocab_size, dim, 1>(out, &self.weights.embeddings, &self.buffer.state0)
    }
}

pub fn generate(
    weights_pth: String,
    prompt: String,
    steps: usize,
    print: bool,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mmap: MmapRaw = MmapOptions::new()
        .offset(28)
        .map_raw_read_only(&File::open(weights_pth)?)?;
    let weights: &[f32] = unsafe { from_raw_parts(mmap.as_ptr() as *const f32, mmap.len()) };

    let mut index = 0;
    let mut length = vocab_size * dim;
    let embeddings = &weights[index..index + length];
    index += length;
    length = n_layers * dim;
    let rms_attention = &weights[index..index + length];
    index += length;
    length = n_layers * dim * n_heads * head_size;
    let query = &weights[index..index + length];
    index += length;
    length = n_layers * dim * n_kv_heads * head_size;
    let key = &weights[index..index + length];
    index += length;
    length = n_layers * dim * n_kv_heads * head_size;
    let value = &weights[index..index + length];
    index += length;
    length = n_layers * n_heads * head_size * dim;
    let heads = &weights[index..index + length];
    index += length;
    length = n_layers * dim;
    let rms_feedforward = &weights[index..index + length];
    index += length;
    length = n_layers * dim * hidden_dim;
    let ff1 = &weights[index..index + length];
    index += length;
    length = n_layers * hidden_dim * dim;
    let ff2 = &weights[index..index + length];
    index += length;
    length = n_layers * dim * hidden_dim;
    let swiglu = &weights[index..index + length];
    index += length;
    length = n_layers * dim * hidden_dim;
    let rms_final = &weights[index..index + length];

    let mut model = Model {
        state: vec![0f32; dim],
        buffer: Buffer {
            state0: vec![0f32; dim],
            state1: vec![0f32; dim],
            query: vec![0f32; dim],
            attention: vec![0f32; state_size],
            swiglu: vec![0f32; hidden_dim],
            ff_hidden: vec![0f32; hidden_dim],
        },
        cache: Cache {
            key: vec![0f32; n_layers * state_size * kv_dim],
            value: vec![0f32; n_layers * state_size * kv_dim],
        },
        weights: Weights {
            embeddings,
            query,
            key,
            value,
            heads,
            rms_attention,
            rms_feedforward,
            rms_final,
            ff1,
            ff2,
            swiglu,
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
