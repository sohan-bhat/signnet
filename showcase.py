"""
showcase.py - loads the trained SignNet model and prints a clean,
colored table of predictions on real GTSRB test images.

Run from the signnet/ folder:  python3 showcase.py
Reuses your existing data.py / layers.py / network.py so the
forward math is identical to what was trained.
"""

import numpy as np
from data import get_data
from network import Network
from layers import Conv, Dense, Flatten, MaxPool, ReLU

MODEL_PATH = "models/model_90.npz"
SAMPLE_COUNT = 14          # how many example signs to show in the table
BATCH = 64                 # batch size for memory-safe evaluation

# ---- the 43 GTSRB class names (index -> readable label) ----
SIGN_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing >3.5t", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles >3.5t prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing >3.5t",
}

# ---- ANSI truecolor styling (SignNet orange theme) ----
O = "\033[38;2;240;140;50m"    # orange accent
G = "\033[38;2;45;205;160m"    # emerald / blue-green (correct)
R = "\033[38;2;235;95;95m"     # red (wrong)
D = "\033[38;2;120;120;130m"   # dim gray
W = "\033[38;2;235;235;240m"   # near-white
B = "\033[1m"
X = "\033[0m"

NW = 28  # name column width


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def conf_bar(conf, width=14):
    filled = int(round(conf * width))
    # full block fills the whole cell (no half-height gap), light shade for empty
    return "\u2588" * filled + "\u2591" * (width - filled)


def trunc(s, n):
    return s if len(s) <= n else s[:n - 1] + "\u2026"


def evaluate_full(network, X, y):
    """Memory-safe full-set accuracy."""
    correct = 0
    for start in range(0, X.shape[0], BATCH):
        xb = X[start:start + BATCH]
        yb = y[start:start + BATCH]
        preds = np.argmax(network.forward(xb), axis=1)
        correct += np.sum(preds == yb)
    return correct / X.shape[0] * 100


def pick_samples(y, count):
    """Grab one image index from each of `count` evenly-spread classes."""
    chosen = []
    classes = np.linspace(0, 42, count).round().astype(int)
    for c in classes:
        idxs = np.where(y == c)[0]
        if len(idxs):
            chosen.append(int(idxs[len(idxs) // 2]))  # a middle example
    return chosen


def main():
    print(f"\n  {O}{B}loading SignNet...{X}")
    X_train, y_train, X_test, y_test = get_data()  # images stay 32x32x3 for the CNN

    layers = [
        Conv(32, 3, 3), ReLU(), MaxPool(2),
        Conv(64, 3, 32), ReLU(), MaxPool(2),
        Flatten(),
        Dense(2304, 128), ReLU(),
        Dense(128, 43),
    ]
    network = Network(layers)
    network.load(MODEL_PATH)

    # full-set accuracy (the headline number)
    test_acc = evaluate_full(network, X_test, y_test)

    # sample predictions
    idxs = pick_samples(y_test, SAMPLE_COUNT)
    probs = softmax(network.forward(X_test[idxs]))
    preds = np.argmax(probs, axis=1)
    confs = probs[np.arange(len(idxs)), preds]
    truths = y_test[idxs]

    # ---- render ----
    line = D + "\u2500" * 66 + X
    print()
    print("  " + O + B + "\u25b2 " + W + B + "SignNet" + X
          + D + "   from-scratch CNN  \u00b7  43-class traffic sign classifier" + X)
    print("  " + line)
    print("  " + D + f"{'#':<3}{'TRUE SIGN':<{NW}}{'PREDICTED':<{NW}}{'CONFIDENCE':<14}" + X)
    print("  " + line)

    shown_correct = 0
    for i, (t, p, c) in enumerate(zip(truths, preds, confs), 1):
        ok = (t == p)
        shown_correct += ok
        mark = G + "\u2713" + X if ok else R + "\u2717" + X
        pcol = G if ok else R
        barcol = G if ok else R
        print("  "
              + D + f"{i:<3}" + X
              + W + f"{trunc(SIGN_NAMES[int(t)], NW-1):<{NW}}" + X
              + pcol + f"{trunc(SIGN_NAMES[int(p)], NW-1):<{NW}}" + X
              + barcol + conf_bar(float(c)) + X
              + W + f" {c*100:5.1f}%" + X
              + "  " + mark)
        print()  # blank line separates each bar

    print("  " + line)
    print("  " + W + B + "Test accuracy (full 12,630-image set): " + O + f"{test_acc:.2f}%" + X)
    print("  " + D + f"sample shown above: {shown_correct}/{len(idxs)} correct"
          + "   \u00b7   trained from scratch in NumPy, no ML frameworks" + X)
    print()


if __name__ == "__main__":
    main()