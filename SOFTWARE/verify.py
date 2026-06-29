"""
SNN Weight Verification
=======================
Hardware-faithful simulation using exported int8 weights.
Matches RTL exactly: same LFSR, same LIF equations, same T=25.

Usage:
  python verify.py              # test 500 images (fast numpy)
  python verify.py --all        # test all 10000 images
  python verify.py --exact      # exact LFSR mode (slow but bit-perfect)
  python verify.py --digit 7    # show spike counts for digit 7
"""

import os
import sys
import time
import argparse
import numpy as np

# hardware parameters - must match RTL
T          = 25
V_TH       = 100
LEAK_SHIFT = 4
LFSR_SEED  = 0xA5
LFSR_TAPS  = (7, 5, 4, 3)   # poly x^8+x^6+x^5+x^4+1

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_weights():
    w1 = os.path.join(OUT_DIR, "weights_l1_int8.npy")
    w2 = os.path.join(OUT_DIR, "weights_l2_int8.npy")
    if not os.path.exists(w1):
        print("ERROR: weights not found. Run train.py first.")
        sys.exit(1)
    W1 = np.load(w1)
    W2 = np.load(w2)
    print(f"W1 : {W1.shape}  {W1.dtype}  [{W1.min()}, {W1.max()}]")
    print(f"W2 : {W2.shape}  {W2.dtype}  [{W2.min()}, {W2.max()}]")
    return W1, W2


class LFSR8:
    """8-bit LFSR matching spike_encoder.v exactly."""
    def __init__(self):
        self.s = LFSR_SEED & 0xFF

    def step(self):
        fb = 0
        for t in LFSR_TAPS:
            fb ^= (self.s >> t) & 1
        self.s = ((self.s << 1) | fb) & 0xFF

    def val(self):
        return self.s


def lif_step(v, i_syn):
    """Single LIF update matching lif_core.v."""
    v_leak = v - (v >> LEAK_SHIFT)
    u      = v_leak + i_syn
    fire   = 1 if u >= V_TH else 0
    return (0 if fire else u), fire


def infer_exact(img, W1, W2):
    """Exact LFSR inference - bit-perfect match to hardware."""
    lfsr   = LFSR8()
    vh     = [0] * 256
    vo     = [0] * 10
    counts = [0] * 10

    for _ in range(T):
        spk_h = []
        for h in range(256):
            i_syn = 0
            for j in range(784):
                sp = 1 if img[j] > lfsr.val() else 0
                lfsr.step()
                if sp:
                    i_syn += int(W1[h, j])
            vh[h], fire = lif_step(vh[h], i_syn)
            spk_h.append(fire)

        for o in range(10):
            i_syn = 0
            for h in range(256):
                if spk_h[h]:
                    i_syn += int(W2[o, h])
            vo[o], fire = lif_step(vo[o], i_syn)
            if fire:
                counts[o] += 1

    return int(np.argmax(counts)), counts


def infer_batch(imgs, labels, W1, W2, n=500):
    """Fast batched inference using numpy matmul.
    Uses random thresholds instead of LFSR - same statistics."""
    rng  = np.random.default_rng(42)
    n    = min(n, len(imgs))
    X    = imgs[:n]
    y    = labels[:n]
    W1T  = W1.astype(np.int32).T
    W2T  = W2.astype(np.int32).T
    vh   = np.zeros((n, 256), dtype=np.int32)
    vo   = np.zeros((n, 10),  dtype=np.int32)
    cnt  = np.zeros((n, 10),  dtype=np.int32)

    for _ in range(T):
        thresh = rng.integers(0, 256, (n, 784), dtype=np.uint8)
        sp1    = (X > thresh).astype(np.int32)
        u1     = vh - (vh >> LEAK_SHIFT) + sp1 @ W1T
        f1     = (u1 >= V_TH).astype(np.int32)
        vh     = np.where(f1, 0, u1)

        u2     = vo - (vo >> LEAK_SHIFT) + f1 @ W2T
        f2     = (u2 >= V_TH).astype(np.int32)
        vo     = np.where(f2, 0, u2)
        cnt   += f2

    preds = cnt.argmax(axis=1)
    return preds, y


def print_results(preds, labels):
    print(f"\n  {'Digit':>5} | {'Correct':>7} | {'Total':>6} | {'Acc':>8}")
    print("  " + "-" * 36)
    for d in range(10):
        mask    = (labels == d)
        total   = mask.sum()
        correct = (preds[mask] == d).sum()
        print(f"  {d:>5} | {correct:>7} | {total:>6} | {correct/total*100:>7.1f}%")
    overall = (preds == labels).mean() * 100
    print("  " + "-" * 36)
    print(f"  {'ALL':>5} | {'':>7} | {'':>6} | {overall:>7.2f}%")
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all',   action='store_true')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--digit', type=int, default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("SNN WEIGHT VERIFICATION")
    print("=" * 50)
    print(f"T={T}  V_TH={V_TH}  LEAK_SHIFT={LEAK_SHIFT}  SEED=0x{LFSR_SEED:02X}")
    print()

    W1, W2 = load_weights()
    print()

    try:
        from torchvision import datasets, transforms
        test_ds    = datasets.MNIST('./data', train=False, download=True,
                                    transform=transforms.ToTensor())
        imgs_raw   = test_ds.data.numpy()
        labels_all = test_ds.targets.numpy()
        imgs_flat  = imgs_raw.reshape(10000, 784)
        print(f"Test set : {len(imgs_flat)} images loaded")
    except ImportError:
        print("ERROR: torchvision not found.")
        sys.exit(1)

    print()

    # single digit demo
    d   = args.digit if args.digit is not None else 3
    idx = np.where(labels_all == d)[0][0]
    print(f"Single image demo (digit={d}):")
    pred, counts = infer_exact(imgs_flat[idx], W1, W2)
    print(f"  Spike counts : {counts}")
    print(f"  Predicted    : {pred}  ({'PASS' if pred == d else 'FAIL'})")
    print()

    # batch test
    n    = 10000 if args.all else 500
    mode = "exact LFSR" if args.exact else "batched numpy"
    print(f"Testing {n} images [{mode}]...")

    t0 = time.time()
    if args.exact:
        preds_list = []
        for i in range(n):
            p, _ = infer_exact(imgs_flat[i], W1, W2)
            preds_list.append(p)
            if (i + 1) % 50 == 0:
                acc = sum(preds_list[k] == labels_all[k]
                          for k in range(i + 1)) / (i + 1)
                print(f"  [{i+1}/{n}]  acc={acc*100:.1f}%")
        preds  = np.array(preds_list)
        labels = labels_all[:n]
    else:
        preds, labels = infer_batch(imgs_flat, labels_all, W1, W2, n=n)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({n/elapsed:.0f} img/s)")

    overall = print_results(preds, labels)

    print()
    print("=" * 50)
    if overall >= 90:
        print(f"RESULT : {overall:.2f}%  PASS (weights good)")
    elif overall >= 80:
        print(f"RESULT : {overall:.2f}%  OK   (consider more epochs)")
    else:
        print(f"RESULT : {overall:.2f}%  FAIL (retrain)")
    print("=" * 50)
    print()

    # check output files exist
    files = ["weights_l1_final.bin", "weights_l2_final.bin",
             "weights_l1_hex.txt",   "weights_l2_hex.txt",
             "digit0_pix.txt"]
    print("Output files:")
    for fname in files:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            print(f"  OK  {fname}  ({os.path.getsize(path)} bytes)")
        else:
            print(f"  MISSING  {fname}")


if __name__ == "__main__":
    main()
