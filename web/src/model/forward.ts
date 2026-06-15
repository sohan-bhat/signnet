import type { ModelWeights } from "./weights";

export interface Tensor3 {
  data: Float32Array;
  h: number;
  w: number;
  c: number;
}

export function conv(
  input: Tensor3,
  weights: Float32Array,
  biases: Float32Array,
  numFilters: number,
  kernel: number
): Tensor3 {
  const { data: x, h: H, w: W, c: C } = input;
  const OH = H - kernel + 1;
  const OW = W - kernel + 1;
  const out = new Float32Array(OH * OW * numFilters);

  for (let oh = 0; oh < OH; oh++) {
    for (let ow = 0; ow < OW; ow++) {
      const outBase = (oh * OW + ow) * numFilters;
      for (let f = 0; f < numFilters; f++) {
        let sum = biases[f];

        const wFilterBase = f * kernel * kernel * C;
        for (let i = 0; i < kernel; i++) {
          const inRow = ((oh + i) * W) * C;
          const wRow = wFilterBase + i * kernel * C;
          for (let j = 0; j < kernel; j++) {
            const inBase = inRow + (ow + j) * C;
            const wBase = wRow + j * C;
            for (let c = 0; c < C; c++) {
              sum += x[inBase + c] * weights[wBase + c];
            }
          }
        }
        out[outBase + f] = sum;
      }
    }
  }
  return { data: out, h: OH, w: OW, c: numFilters };
}

export function relu(t: Tensor3): Tensor3 {
  const d = t.data;
  for (let i = 0; i < d.length; i++) {
    if (d[i] < 0) d[i] = 0;
  }
  return t;
}

export function maxpool2(input: Tensor3): Tensor3 {
  const { data: x, h: H, w: W, c: C } = input;
  const OH = Math.floor(H / 2);
  const OW = Math.floor(W / 2);
  const out = new Float32Array(OH * OW * C);

  for (let oh = 0; oh < OH; oh++) {
    for (let ow = 0; ow < OW; ow++) {
      const outBase = (oh * OW + ow) * C;
      const h0 = oh * 2;
      const w0 = ow * 2;
      for (let c = 0; c < C; c++) {
        let m = -Infinity;
        for (let dh = 0; dh < 2; dh++) {
          const rowBase = ((h0 + dh) * W + w0) * C;
          for (let dw = 0; dw < 2; dw++) {
            const v = x[rowBase + dw * C + c];
            if (v > m) m = v;
          }
        }
        out[outBase + c] = m;
      }
    }
  }
  return { data: out, h: OH, w: OW, c: C };
}

export function flatten(t: Tensor3): Float32Array {
  return t.data;
}

export function dense(
  input: Float32Array,
  weights: Float32Array,
  biases: Float32Array,
  nOut: number
): Float32Array {
  const nIn = input.length;
  const out = new Float32Array(nOut);
  for (let m = 0; m < nOut; m++) out[m] = biases[m];
  for (let n = 0; n < nIn; n++) {
    const xn = input[n];
    if (xn === 0) continue;
    const wRow = n * nOut;
    for (let m = 0; m < nOut; m++) {
      out[m] += xn * weights[wRow + m];
    }
  }
  return out;
}

export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export interface Prediction {
  logits: Float32Array;
  probs: Float32Array;
  pred: number;
}

export function predict(input32: Float32Array, weights: ModelWeights): Prediction {
  let t: Tensor3 = { data: input32, h: 32, w: 32, c: 3 };

  t = conv(t, weights.w0, weights.b0, 32, 3);
  relu(t);
  t = maxpool2(t);

  t = conv(t, weights.w1, weights.b1, 64, 3);
  relu(t);
  t = maxpool2(t);

  const flat = flatten(t);

  let v = dense(flat, weights.w2, weights.b2, 128);

  for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0;

  const logits = dense(v, weights.w3, weights.b3, 43);
  const probs = softmax(logits);

  let pred = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[pred]) pred = i;

  return { logits, probs, pred };
}

export function topK(probs: Float32Array, k: number): number[] {
  return Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, k);
}
