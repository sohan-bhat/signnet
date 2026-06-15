# SignNet — Web Demo

A fully static, serverless web demo for the from-scratch NumPy CNN that classifies
German traffic signs (GTSRB, 43 classes) at ~90% test accuracy. The exact forward
pass is reimplemented in TypeScript and runs entirely in the browser — no backend,
no ML frameworks.

Built with **Vite + React + TypeScript**. Deployable to Netlify as static files.

## Quick start

```bash
# from the repo root, regenerate weights + samples + verification fixtures
python3 export_weights.py

# then build/run the web app
cd web
npm install
npm run verify   # check the TS forward pass matches the Python model
npm run dev      # local dev server
npm run build    # production build -> web/dist
npm run preview  # serve the production build
```

`export_weights.py` (repo root) reads `models/model_90.npz` and writes everything
the app needs into `web/public/`:

| Output | Contents |
| --- | --- |
| `public/weights.bin` | All params concatenated as little-endian **float32**, C-order (~1.3 MB) |
| `public/weights.json` | Manifest: each param's `shape`, element `offset`, `count` |
| `public/samples/<id>.png` | One held-out GTSRB test image per class (0–42) |
| `public/samples/manifest.json` | `[{classId, file}]` for the gallery |
| `web/verify/fixtures.json` | Pre-resized 32×32 inputs + Python `logits`/`probs`/`pred` for verification (not shipped to `dist`) |

## Architecture (reimplemented in `src/model/forward.ts`)

Tensors are stored flat as `Float32Array` in **HWC, C-order**:
`index(h, w, c) = (h * width + w) * channels + c` — matching NumPy's row-major
layout for an `(H, W, C)` array.

```
Input 32×32×3  (RGB, ÷255)
 → Conv 32×(3×3) valid, no flip      → 30×30×32   W:(32,3,3,3)  b:(32,)
 → ReLU
 → MaxPool 2×2 stride 2              → 15×15×32
 → Conv 64×(3×3) valid, no flip      → 13×13×64   W:(64,3,3,32) b:(64,)
 → ReLU
 → MaxPool 2×2 stride 2             →  6×6×64
 → Flatten (C-order h,w,c)           → 2304
 → Dense 2304→128 → ReLU            W:(2304,128) b:(1,128)
 → Dense 128→43                      W:(128,43)   b:(1,43)
 → Softmax → argmax
```

Key correctness details that mirror the Python (`layers.py`):

- **Valid cross-correlation, no kernel flip:** `out[oh,ow,f] = Σ_{i,j,c} in[oh+i,ow+j,c]·W[f,i,j,c] + b[f]`.
  The patch flatten order `(i,j,c)` matches `weights.reshape(num_filters, -1)`.
- **Flatten** is a no-op reorder: tensors are already HWC C-order, identical to
  `arr.reshape(batch, -1)` over `(h, w, c)`.
- **Softmax** subtracts the row max for numerical stability (matches `losses.py`).

## Verification & discrepancies

`npm run verify` runs `src/model/verify.ts`, which feeds the **identical**
pre-resized 32×32 inputs from `fixtures.json` to the TS `predict()` and compares
against the Python outputs. Latest run:

```
PASS  0.png    py_pred=0 js_pred=0  maxProbDiff=2.65e-6
PASS  1.png    py_pred=1 js_pred=1  maxProbDiff=1.78e-8
PASS  2.png    py_pred=2 js_pred=2  maxProbDiff=3.10e-12
PASS  3.png    py_pred=3 js_pred=3  maxProbDiff=1.64e-6
PASS  4.png    py_pred=4 js_pred=4  maxProbDiff=2.60e-6
✓ ALL FIXTURES PASSED
```

On identical inputs the predicted class always matches and probabilities agree to
**~1e-6**. The only source of difference is the weights being stored as float32
(Python trains/infers in float64); this is far below any decision boundary.

**Live-image caveat (documented, expected):** when you upload an image or click a
gallery sample, the browser resizes it to 32×32 with the canvas's **bilinear**
filter, whereas Python's `PIL.Image.resize` defaults to **bicubic**. This yields
slightly different pixel values and therefore slightly different probabilities than
a Python run on the same source file. The predicted class is almost always
identical; only genuinely borderline images may flip. The verification above
deliberately bypasses this by comparing on identical pre-resized tensors, isolating
forward-pass math from the resize filter.

## Deployment (Netlify)

`netlify.toml` is configured for the `web/` subdirectory:

```toml
[build]
  base = "web"
  command = "npm run build"
  publish = "dist"
```

Everything in `public/` (weights, samples) is copied verbatim into `dist/`, so the
deployed site is fully static. Point Netlify at this repo (or drag-and-drop
`web/dist`) and it just works.
