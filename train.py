"""
PHASE 1 — SNN MNIST Training + Weight Export
=============================================
Architecture  : 784 -> 256 -> 10
Framework     : snnTorch + PyTorch (surrogate gradient)
Hardware params (must match your RTL exactly):
  T           = 25      timesteps
  BETA        = 0.9375  (= 1 - 1/16, matches LEAK_SHIFT=4 in lif_core.v)
  SCALE       = 64      (quantization scale factor)
  CLAMP       = 10      (weight clamp +-10, matches adder_tree.v)
  V_TH        = 100     (hardware threshold in lif_core.v)

OUTPUT FILES  (all saved to ./output/)
──────────────────────────────────────
  mnist_snn_best.pt        best trained PyTorch model (reload anytime)

  Hardware / RTL testbench:
    weights_l1_final.bin   W1 int8 raw binary [256x784] = 200704 bytes
    weights_l2_final.bin   W2 int8 raw binary [10x256]  = 2560 bytes
    weights_l1_hex.txt     W1 hex text for $readmemh (one byte per line)
    weights_l2_hex.txt     W2 hex text for $readmemh (one byte per line)
    digit0_pix.txt ...     one hex pixel file per digit for $readmemh

  Software / Python:
    weights_l1.npy         float32 W1 shape (256, 784)
    weights_l2.npy         float32 W2 shape (10, 256)
    weights_l1_int8.npy    int8    W1 shape (256, 784)
    weights_l2_int8.npy    int8    W2 shape (10, 256)
"""

import os, time
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hardware-matched parameters ───────────────────────────────────────────────
T      = 25       # timesteps  (must match RTL parameter T)
BETA   = 0.9375   # 1 - 1/2^4 = 0.9375  (matches LEAK_SHIFT=4 in lif_core.v)
SCALE  = 64       # quantisation scale factor
CLAMP  = 10       # weight clamp +-10   (matches adder_tree.v original)

# ── Training parameters ───────────────────────────────────────────────────────
BATCH  = 128
EPOCHS = 25
LR     = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 55)
print("  PHASE 1 — SNN MNIST TRAINING")
print("=" * 55)
print(f"  Device : {DEVICE}")
print(f"  T={T}  BETA={BETA}  SCALE={SCALE}  CLAMP=+-{CLAMP}")
print(f"  Epochs={EPOCHS}  Batch={BATCH}  LR={LR}")
print()

# ── Data ──────────────────────────────────────────────────────────────────────
# No normalisation — pixel values stay in [0,1] as spike probability
transform    = transforms.Compose([transforms.ToTensor()])
train_ds     = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds      = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=512,   shuffle=False, num_workers=0)

print(f"  Train samples : {len(train_ds)}")
print(f"  Test  samples : {len(test_ds)}")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
spike_grad = surrogate.fast_sigmoid(slope=25)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 256, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True)
        self.fc2  = nn.Linear(256, 10,  bias=False)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        # x shape: (batch, 784)
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        spk2_sum = torch.zeros(x.size(0), 10, device=x.device)

        for _ in range(T):
            # Rate encode: compare pixel value vs uniform random threshold
            inp = (x > torch.rand_like(x)).float()
            spk1 = self.lif1(self.fc1(inp))
            spk2 = self.lif2(self.fc2(spk1))
            spk2_sum += spk2

        return spk2_sum   # spike counts over T timesteps

model     = SNN().to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()

print(f"  Model parameters : {sum(p.numel() for p in model.parameters()):,}")
print()

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Test Acc':>8}  {'Time':>6}")
print("─" * 50)

best_test_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in train_loader:
        imgs   = imgs.view(-1, 784).to(DEVICE)
        labels = labels.to(DEVICE)

        optimiser.zero_grad()
        spike_counts = model(imgs)
        loss = criterion(spike_counts, labels)
        loss.backward()
        optimiser.step()

        total_loss    += loss.item() * imgs.size(0)
        total_correct += (spike_counts.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    scheduler.step()

    # ── Test accuracy ─────────────────────────────────────────────────────────
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)
            preds  = model(imgs).argmax(1)
            test_correct += (preds == labels).sum().item()

    train_loss = total_loss    / total_samples
    train_acc  = total_correct / total_samples
    test_acc   = test_correct  / len(test_ds)
    elapsed    = time.time() - t0

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
          f"{test_acc*100:>7.2f}%  {elapsed:>5.1f}s")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(),
                   os.path.join(OUT_DIR, "mnist_snn_best.pt"))

print()
print(f"  Best test accuracy : {best_test_acc*100:.2f}%")
print()

# ── Load best model for export ────────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "mnist_snn_best.pt"),
                                  map_location='cpu'))
model.eval()

W1_float = model.fc1.weight.detach().cpu().numpy()  # shape (256, 784)
W2_float = model.fc2.weight.detach().cpu().numpy()  # shape (10, 256)

print(f"  W1 float shape : {W1_float.shape}  range [{W1_float.min():.4f}, {W1_float.max():.4f}]")
print(f"  W2 float shape : {W2_float.shape}  range [{W2_float.min():.4f}, {W2_float.max():.4f}]")

# ── Quantise to int8 (SCALE=64, CLAMP=+-10) ──────────────────────────────────
def quantise(W, scale=SCALE, clamp=CLAMP):
    W_scaled = np.round(W * scale).astype(np.int32)
    W_clamped = np.clip(W_scaled, -clamp, clamp).astype(np.int8)
    return W_clamped

W1_int8 = quantise(W1_float)   # (256, 784)
W2_int8 = quantise(W2_float)   # (10, 256)

print()
print(f"  W1 int8 shape  : {W1_int8.shape}  range [{W1_int8.min()}, {W1_int8.max()}]")
print(f"  W2 int8 shape  : {W2_int8.shape}  range [{W2_int8.min()}, {W2_int8.max()}]")

# ── Export 1: .bin files (raw int8 binary, for DMA / PYNQ) ───────────────────
p = os.path.join(OUT_DIR, "weights_l1_final.bin")
W1_int8.flatten().tobytes().__class__  # just to verify
with open(p, 'wb') as f:
    f.write(W1_int8.flatten().tobytes())
print(f"\n  [BIN] weights_l1_final.bin  {os.path.getsize(p):>8} bytes  -> {p}")

p = os.path.join(OUT_DIR, "weights_l2_final.bin")
with open(p, 'wb') as f:
    f.write(W2_int8.flatten().tobytes())
print(f"  [BIN] weights_l2_final.bin  {os.path.getsize(p):>8} bytes  -> {p}")

# ── Export 2: .npy files (for software / verify.py) ──────────────────────────
np.save(os.path.join(OUT_DIR, "weights_l1.npy"),      W1_float)
np.save(os.path.join(OUT_DIR, "weights_l2.npy"),      W2_float)
np.save(os.path.join(OUT_DIR, "weights_l1_int8.npy"), W1_int8)
np.save(os.path.join(OUT_DIR, "weights_l2_int8.npy"), W2_int8)
print(f"  [NPY] weights_l1.npy / weights_l1_int8.npy -> {OUT_DIR}/")
print(f"  [NPY] weights_l2.npy / weights_l2_int8.npy -> {OUT_DIR}/")

# ── Export 3: hex text files (for RTL $readmemh in testbench) ────────────────
def write_hex(W_int8, path):
    flat = W_int8.flatten().view(np.uint8)  # reinterpret signed bytes as uint8
    with open(path, 'w') as f:
        for b in flat:
            f.write(f"{int(b):02x}\n")

write_hex(W1_int8, os.path.join(OUT_DIR, "weights_l1_hex.txt"))
write_hex(W2_int8, os.path.join(OUT_DIR, "weights_l2_hex.txt"))
print(f"  [HEX] weights_l1_hex.txt / weights_l2_hex.txt -> {OUT_DIR}/")

# ── Export 4: one pixel image per digit (for testbench) ──────────────────────
# Pick first occurrence of each digit from test set
test_imgs_all   = test_ds.data.numpy()    # (10000, 28, 28) uint8
test_labels_all = test_ds.targets.numpy()

for d in range(10):
    idx = np.where(test_labels_all == d)[0][0]
    pix = test_imgs_all[idx].flatten()   # 784 uint8 values [0..255]
    path = os.path.join(OUT_DIR, f"digit{d}_pix.txt")
    with open(path, 'w') as f:
        for b in pix:
            f.write(f"{int(b):02x}\n")

print(f"  [HEX] digit0_pix.txt .. digit9_pix.txt -> {OUT_DIR}/")

print()
print("=" * 55)
print("  EXPORT COMPLETE — Phase 1 done!")
print("=" * 55)
print()
print("  Next steps:")
print("  1. Run verify.py to confirm weights work in software SNN")
print("  2. Copy hex files + digit*_pix.txt to your Verilog sim folder")
print("  3. Copy .bin files to PYNQ board for hardware inference")
print()