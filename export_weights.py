"""Export trained SignNet weights, sample images, and verification fixtures
for the static web demo in web/.

This script is additive only: it reads models/model_90.npz and data/Test*,
reuses the existing layers/network code unchanged, and writes everything the
browser app needs into web/public/.

Outputs:
  web/public/weights.bin            all params concatenated, little-endian float32, C-order
  web/public/weights.json           manifest: key -> {dtype, shape, offset, count}
  web/public/samples/<classId>.png  one display image per class (0..42)
  web/public/samples/manifest.json  [{classId, file}]
  web/public/verify/fixtures.json   [{file, classId, input32, logits, probs, pred}]

Run:  python3 export_weights.py
"""
import json
import os
import struct

import numpy as np
import pandas as pd
from PIL import Image

from layers import Conv, Dense, Flatten, MaxPool, ReLU
from losses import Softmax
from network import Network

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "model_90.npz")
TEST_CSV = os.path.join(ROOT, "data", "Test.csv")
DATA_DIR = os.path.join(ROOT, "data")

WEB_DIR = os.path.join(ROOT, "web")
WEB_PUBLIC = os.path.join(WEB_DIR, "public")
SAMPLES_DIR = os.path.join(WEB_PUBLIC, "samples")
# Verification fixtures live outside public/ so they are not shipped in the build;
# only `npm run verify` reads them.
VERIFY_DIR = os.path.join(WEB_DIR, "verify")

NUM_CLASSES = 43
DISPLAY_SIZE = 128          # size saved for the gallery thumbnails
NUM_FIXTURES = 5            # how many images to pin for JS<->Python verification


def export_weights():
    """Concatenate every param into one float32 blob + a shape manifest."""
    data = np.load(MODEL_PATH)
    # Preserve the exact save order from Network.save: w0,b0,w1,b1,...
    keys = [f"{p}{i}" for i in range(len(data.files) // 2) for p in ("w", "b")]

    manifest = {}
    blob = bytearray()
    offset = 0  # in float32 elements
    for key in keys:
        arr = np.ascontiguousarray(data[key], dtype="<f4")  # C-order little-endian float32
        flat = arr.reshape(-1)
        manifest[key] = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "offset": offset,         # element offset into the blob
            "count": int(flat.size),
        }
        blob.extend(flat.tobytes())
        offset += int(flat.size)

    os.makedirs(WEB_PUBLIC, exist_ok=True)
    with open(os.path.join(WEB_PUBLIC, "weights.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(WEB_PUBLIC, "weights.json"), "w") as f:
        json.dump({"order": keys, "totalFloats": offset, "params": manifest}, f, indent=2)

    print(f"weights.bin: {len(blob)} bytes ({offset} float32 values)")
    for k, m in manifest.items():
        print(f"  {k:>3}  shape={m['shape']}  offset={m['offset']}  count={m['count']}")
    return keys


def build_network():
    """Same architecture as train.py, loaded with the trained weights."""
    layers = [
        Conv(32, 3, 3), ReLU(), MaxPool(2),
        Conv(64, 3, 32), ReLU(), MaxPool(2),
        Flatten(),
        Dense(2304, 128), ReLU(),
        Dense(128, 43),
    ]
    net = Network(layers)
    net.load(MODEL_PATH)
    return net


def export_samples_and_fixtures():
    """Pick one test image per class for the gallery, and dump verification
    fixtures (identical 32x32 inputs + Python outputs) for the first few."""
    df = pd.read_csv(TEST_CSV)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(VERIFY_DIR, exist_ok=True)

    net = build_network()
    softmax = Softmax()

    sample_manifest = []
    fixtures = []
    for class_id in range(NUM_CLASSES):
        rows = df[df["ClassId"] == class_id]
        if rows.empty:
            print(f"  WARNING: no test image for class {class_id}")
            continue
        rel_path = rows.iloc[0]["Path"]
        src = os.path.join(DATA_DIR, rel_path)
        img = Image.open(src).convert("RGB")

        # Display thumbnail for the gallery.
        out_name = f"{class_id}.png"
        img.resize((DISPLAY_SIZE, DISPLAY_SIZE)).save(os.path.join(SAMPLES_DIR, out_name))
        sample_manifest.append({"classId": class_id, "file": out_name})

        # Verification fixture: replicate the exact training-time preprocessing.
        if len(fixtures) < NUM_FIXTURES:
            arr = np.array(img.resize((32, 32)), dtype=np.float64) / 255.0  # 32x32x3, HWC, RGB
            logits = net.forward(arr[None, ...])[0]                          # (43,)
            probs = softmax.forward(logits[None, :])[0]                      # (43,)
            fixtures.append({
                "file": out_name,
                "classId": class_id,
                "input32": arr.reshape(-1).tolist(),   # C-order: h,w,c
                "logits": logits.tolist(),
                "probs": probs.tolist(),
                "pred": int(np.argmax(logits)),
            })

    with open(os.path.join(SAMPLES_DIR, "manifest.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)
    with open(os.path.join(VERIFY_DIR, "fixtures.json"), "w") as f:
        json.dump(fixtures, f)

    print(f"samples: wrote {len(sample_manifest)} images to {SAMPLES_DIR}")
    print(f"fixtures: wrote {len(fixtures)} entries to {VERIFY_DIR}/fixtures.json")
    for fx in fixtures:
        print(f"  class {fx['classId']:>2} -> pred {fx['pred']:>2}  p={max(fx['probs']):.4f}")


if __name__ == "__main__":
    print("Exporting weights...")
    export_weights()
    print("\nExporting samples + verification fixtures...")
    export_samples_and_fixtures()
    print("\nDone.")
