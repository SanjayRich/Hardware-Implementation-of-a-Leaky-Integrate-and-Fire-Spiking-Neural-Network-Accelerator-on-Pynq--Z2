"""
PHASE 1 — Weight Verification
==============================
Runs a hardware-faithful SNN simulation using the exported int8 weights.
This is your GOLDEN REFERENCE — it simulates exactly what the Verilog
hardware does: same LIF equations, same LFSR encoder, same T=25 timesteps.

Run AFTER train.py has completed and output/ folder exists.

Usage:
  python verify.py            (tests 200 images, prints per-digit table)
  python verify.py --all      (tests all 10000 test images, takes ~5 min)
"""

import os, sys, argparse
import numpy as np

# ── Hardware-matched parameters (must match RTL + train.py) ──────────────────
T          = 25       # timesteps
V_TH       = 100      # LIF threshold  (lif_core.v  V_TH=100)
LEAK_SHIFT = 4        # LIF leak shift (lif_core.v  LEAK_SHIFT=4)
LFSR_SEED  = 0xA5     # spike_encoder.v seed
# LFSR poly x^8+x^6+x^5+x^4+1  (taps on bits 7,5,4,3)
LFSR_TAPS  = (7, 5, 4, 3)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── Load weights ──────────────────────────────────────────────────────────────
def load_weights():
    w1_path = os.path.join(OUT_DIR, "weights_l1_int8.npy")
    w2_path = os.path.join(OUT_DIR, "weights_l2_int8.npy")
    if not os.path.exists(w1_path):
        print(f"ERROR: {w1_path} not found. Run train.py first!")
        sys.exit(1)
    W1 = np.load(w1_path)   # (256, 784) int8
    W2 = np.load(w2_path)   # (10, 256)  int8
    print(f"  W1 loaded: shape={W1.shape} dtype={W1.dtype} range=[{W1.min()},{W1.max()}]")
    print(f"  W2 loaded: shape={W2.shape} dtype={W2.dtype} range=[{W2.min()},{W2.max()}]")
    return W1, W2

# ── Hardware-faithful LFSR (matches spike_encoder.v exactly) ─────────────────
class LFSR8:
    def __init__(self):
        self.s = LFSR_SEED & 0xFF
    def step(self):
        fb = 0
        for t in LFSR_TAPS:
            fb ^= (self.s >> t) & 1
        self.s = ((self.s << 1) | fb) & 0xFF
    def val(self):
        return self.s

# ── Hardware-faithful LIF neuron (matches lif_core.v exactly) ────────────────
def lif_update(v, i_syn):
    """
    v_leak = v_mem - (v_mem >> LEAK_SHIFT)   arithmetic right-shift
    u      = v_leak + i_syn
    fire   = (u >= V_TH)
    v_next = 0 if fire else u
    """
    v_leak = v - (v >> LEAK_SHIFT)
    u      = v_leak + i_syn
    fire   = 1 if u >= V_TH else 0
    v_next = 0 if fire else u
    return v_next, fire

# ── Single image inference (exact hardware model) ─────────────────────────────
def infer_single(img_uint8, W1, W2):
    """
    img_uint8 : (784,) uint8 pixels [0..255]
    W1        : (256, 784) int8
    W2        : (10, 256)  int8
    Returns   : predicted class (int), spike_counts (list of 10)
    """
    lfsr   = LFSR8()
    v_h    = [0] * 256   # hidden membrane potentials
    v_o    = [0] * 10    # output membrane potentials
    counts = [0] * 10

    for _ in range(T):
        # ── Layer 1: encode + MAC + LIF ─────────────────────────────────────
        spikes_h = []
        for h in range(256):
            i_syn = 0
            for j in range(784):
                sp = 1 if img_uint8[j] > lfsr.val() else 0
                lfsr.step()
                if sp:
                    i_syn += int(W1[h, j])
            v_h[h], fire = lif_update(v_h[h], i_syn)
            spikes_h.append(fire)

        # ── Layer 2: hidden spikes + MAC + LIF ──────────────────────────────
        for o in range(10):
            i_syn = 0
            for h in range(256):
                if spikes_h[h]:
                    i_syn += int(W2[o, h])
            v_o[o], fire = lif_update(v_o[o], i_syn)
            if fire:
                counts[o] += 1

    pred = int(np.argmax(counts))
    return pred, counts

# ── Batched numpy inference (fast, ~500 images/min) ──────────────────────────
def infer_batch(imgs_uint8, labels, W1, W2, n_samples=200):
    """Fast batched version using numpy matmul + random thresholds.
    Statistically equivalent to the LFSR version for accuracy estimation."""
    rng = np.random.default_rng(42)
    n   = min(n_samples, len(imgs_uint8))
    imgs   = imgs_uint8[:n]
    labels = labels[:n]

    W1T = W1.astype(np.int32).T   # (784, 256)
    W2T = W2.astype(np.int32).T   # (256, 10)

    v_h    = np.zeros((n, 256), dtype=np.int32)
    v_o    = np.zeros((n, 10),  dtype=np.int32)
    counts = np.zeros((n, 10),  dtype=np.int32)

    for _ in range(T):
        # L1
        thresh1 = rng.integers(0, 256, (n, 784), dtype=np.uint8)
        sp1     = (imgs > thresh1).astype(np.int32)
        i_syn1  = sp1 @ W1T                               # (n, 256)
        u1      = v_h - (v_h >> LEAK_SHIFT) + i_syn1
        fire1   = (u1 >= V_TH).astype(np.int32)
        v_h     = np.where(fire1, 0, u1)

        # L2
        i_syn2  = fire1 @ W2T                             # (n, 10)
        u2      = v_o - (v_o >> LEAK_SHIFT) + i_syn2
        fire2   = (u2 >= V_TH).astype(np.int32)
        v_o     = np.where(fire2, 0, u2)
        counts += fire2

    preds = counts.argmax(axis=1)
    return preds, labels, counts

# ── Per-digit accuracy table ──────────────────────────────────────────────────
def print_table(preds, labels):
    print(f"\n  {'Digit':>5} | {'Correct':>7} | {'Total':>6} | {'Accuracy':>9}")
    print("  " + "─" * 38)
    for d in range(10):
        mask    = (labels == d)
        total   = mask.sum()
        correct = (preds[mask] == d).sum()
        bar     = "█" * int(correct/total*20) if total else ""
        print(f"  {d:>5} | {correct:>7} | {total:>6} | {correct/total*100:>8.1f}%  {bar}")
    overall = (preds == labels).mean()
    print("  " + "─" * 38)
    print(f"  {'ALL':>5} | {'':>7} | {'':>6} | {overall*100:>8.2f}%")
    return overall

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all',    action='store_true', help='test all 10000 images (slow)')
    parser.add_argument('--exact',  action='store_true', help='use exact LFSR mode (very slow)')
    parser.add_argument('--digit',  type=int, default=None, help='show spike counts for one digit')
    args = parser.parse_args()

    print("=" * 55)
    print("  PHASE 1 — WEIGHT VERIFICATION")
    print("=" * 55)
    print(f"  T={T}  V_TH={V_TH}  LEAK_SHIFT={LEAK_SHIFT}  LFSR_SEED=0x{LFSR_SEED:02X}")
    print()

    # Load weights
    W1, W2 = load_weights()
    print()

    # Load MNIST test set
    try:
        from torchvision import datasets, transforms
        test_ds    = datasets.MNIST('./data', train=False, download=True,
                                    transform=transforms.ToTensor())
        imgs_raw   = test_ds.data.numpy()             # (10000, 28, 28) uint8
        labels_all = test_ds.targets.numpy()
        imgs_flat  = imgs_raw.reshape(10000, 784)     # (10000, 784) uint8
        print(f"  Test set loaded: {len(imgs_flat)} images")
    except ImportError:
        print("ERROR: torchvision not found. Install with: pip install torchvision")
        sys.exit(1)
    print()

    # ── Single digit spike-count demo ────────────────────────────────────────
    d = args.digit if args.digit is not None else 3
    idx = np.where(labels_all == d)[0][0]
    print(f"  Single image demo  (digit={d}, exact LFSR):")
    pred, counts = infer_single(imgs_flat[idx], W1, W2)
    print(f"  Spike counts : {counts}")
    print(f"  Predicted    : {pred}   ({'PASS' if pred == d else 'FAIL'})")
    print()

    # ── Batch accuracy ────────────────────────────────────────────────────────
    n = 10000 if args.all else 500
    mode = "exact LFSR" if args.exact else "batched numpy (fast)"
    print(f"  Running SNN on {n} test images  [{mode}] ...")

    import time
    t0 = time.time()
    if args.exact:
        preds_list = []
        for i in range(n):
            p, _ = infer_single(imgs_flat[i], W1, W2)
            preds_list.append(p)
            if (i+1) % 50 == 0:
                acc_so_far = sum(preds_list[k]==labels_all[k] for k in range(i+1)) / (i+1)
                print(f"    [{i+1:>5}/{n}]  running acc: {acc_so_far*100:.1f}%")
        preds  = np.array(preds_list)
        labels = labels_all[:n]
    else:
        preds, labels, _ = infer_batch(imgs_flat, labels_all, W1, W2, n_samples=n)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({n/elapsed:.0f} images/s)")

    overall = print_table(preds, labels)

    print()
    print("=" * 55)
    if overall >= 0.90:
        print(f"  RESULT: {overall*100:.2f}%  ✓  (>=90% — weights are good!)")
    elif overall >= 0.80:
        print(f"  RESULT: {overall*100:.2f}%  ~  (80-90% — acceptable, consider more epochs)")
    else:
        print(f"  RESULT: {overall*100:.2f}%  ✗  (<80% — retrain with more epochs)")
    print("=" * 55)
    print()
    print("  Output files ready for hardware:")
    for f in ["weights_l1_final.bin","weights_l2_final.bin",
              "weights_l1_hex.txt","weights_l2_hex.txt",
              "digit0_pix.txt"]:
        path = os.path.join(OUT_DIR, f)
        if os.path.exists(path):
            print(f"    ✓ {f}  ({os.path.getsize(path)} bytes)")
        else:
            print(f"    ✗ {f}  MISSING — run train.py first")

if __name__ == "__main__":
    main()
