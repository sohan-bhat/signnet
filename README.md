# SignNet

A convolutional neural network that classifies German traffic signs into 43 categories (stop signs, speed limits, yield, and so on). The network, including backpropagation, is written directly in NumPy with no deep learning frameworks: no PyTorch, no TensorFlow, no autograd.

**Live demo:** https://signnet-cnn.netlify.app

It reaches **90.06% accuracy** on the GTSRB test set (12,630 images it never saw during training).

## How it works

Every layer is implemented by hand. The forward pass is plain matrix math; the backward pass derives and applies the gradients manually, layer by layer. The convolution is the interesting part: rather than looping over every position in Python, patches are unfolded into a matrix (im2col) so the whole convolution becomes a single matrix multiply, which is what makes training on a CPU practical.

The pipeline:

- Load GTSRB, resize images to 32×32 RGB, normalize.
- Forward through the network to get class scores.
- Softmax + cross-entropy for the loss.
- Backpropagate gradients through every layer.
- Update weights with mini-batch gradient descent, shuffling each epoch, decaying the learning rate, and keeping the best model by test accuracy.

## Architecture

```
Input 32×32×3
 → Conv 32×(3×3)  → ReLU → MaxPool 2×2
 → Conv 64×(3×3)  → ReLU → MaxPool 2×2
 → Flatten (2304)
 → Dense 2304→128 → ReLU
 → Dense 128→43   → Softmax
```

Getting from one conv block to two was what pushed accuracy from the high 80s past 90%; the second block lets the network build shapes out of the edges the first one detects.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Loads and preprocesses the GTSRB dataset |
| `layers.py` | Conv, MaxPool, Flatten, Dense, ReLU (forward + backward) |
| `losses.py` | Softmax and cross-entropy |
| `network.py` | Assembles layers, runs forward/backward, saves/loads weights |
| `train.py` | Training loop |
| `showcase.py` | Loads the trained model and prints sample predictions |
| `web/` | The browser demo (runs the trained model in JavaScript, no server) |
| `models/` | Saved weights |

## Running it

Download GTSRB from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and place the `Train/`, `Test/`, and CSV files in a `data/` folder (kept out of the repo because of its size).

```bash
python3 train.py       # trains and saves weights to models/
python3 showcase.py    # loads the trained model, prints predictions
```

The web demo loads the exported weights and runs the same forward pass in the browser, so it reflects whatever model is currently saved.

## Notes

The model overfits a little (it scores higher on training data than test data), which is expected for a network this size without data augmentation. Adding augmentation is the next step toward the mid-90s.
