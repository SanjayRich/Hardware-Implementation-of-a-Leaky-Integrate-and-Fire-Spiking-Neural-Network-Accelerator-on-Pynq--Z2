"""
SNN MNIST Training Script
=========================
Architecture : 784 -> 256 -> 10 (LIF neurons)
Framework    : PyTorch + snnTorch

Hardware parameters (must match RTL):
  T      = 25       timesteps
  BETA   = 0.9375   leak factor (1 - 1/16, same as LEAK_SHIFT=4 in lif_core.v)
  SCALE  = 64       quantization scale
  CLAMP  = 10       weight clamp range

Output files saved to ./output/
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

# hardware matched parameters
T     = 25
BETA  = 0.9375
SCALE = 64
CLAMP = 10

# training settings
EPOCHS = 25
BATCH  = 128
LR     = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 50)
print("SNN MNIST TRAINING")
print("=" * 50)
print(f"Device : {DEVICE}")
print(f"T={T}  BETA={BETA}  SCALE={SCALE}  CLAMP={CLAMP}")
print(f"Epochs={EPOCHS}  Batch={BATCH}  LR={LR}")
print()

# load MNIST - no normalization, pixels stay in [0,1] for rate coding
transform    = transforms.Compose([transforms.ToTensor()])
train_ds     = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds      = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=512,   shuffle=False, num_workers=0)

print(f"Train : {len(train_ds)} images")
print(f"Test  : {len(test_ds)} images")
print()


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1  = nn.Linear(784, 256, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True)
        self.fc2  = nn.Linear(256, 10,  bias=False)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        out = torch.zeros(x.size(0), 10, device=x.device)
        for _ in range(T):
            inp  = (x > torch.rand_like(x)).float()
            spk1 = self.lif1(self.fc1(inp))
            spk2 = self.lif2(self.fc2(spk1))
            out += spk2
        return out


model     = SNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_fn   = nn.CrossEntropyLoss()

print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
print()
print(f"{'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Test Acc':>9}  {'Time':>6}")
print("-" * 48)

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs   = imgs.view(-1, 784).to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    scheduler.step()

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)
            test_correct += (model(imgs).argmax(1) == labels).sum().item()

    tr_loss = total_loss / total
    tr_acc  = correct / total * 100
    te_acc  = test_correct / len(test_ds) * 100
    elapsed = time.time() - t0

    print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>9.2f}%  {te_acc:>8.2f}%  {elapsed:>5.1f}s")

    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "mnist_snn_best.pt"))

print()
print(f"Best test accuracy : {best_acc:.2f}%")
print()

# load best weights for export
model.load_state_dict(torch.load(
    os.path.join(OUT_DIR, "mnist_snn_best.pt"), map_location='cpu'))
model.eval()

W1 = model.fc1.weight.detach().cpu().numpy()   # (256, 784)
W2 = model.fc2.weight.detach().cpu().numpy()   # (10,  256)

print(f"W1 : {W1.shape}  [{W1.min():.4f}, {W1.max():.4f}]")
print(f"W2 : {W2.shape}  [{W2.min():.4f}, {W2.max():.4f}]")


def quantize(W):
    W_scaled  = np.round(W * SCALE).astype(np.int32)
    W_clamped = np.clip(W_scaled, -CLAMP, CLAMP).astype(np.int8)
    return W_clamped


W1_int8 = quantize(W1)
W2_int8 = quantize(W2)

print()
print(f"W1 int8 : {W1_int8.shape}  [{W1_int8.min()}, {W1_int8.max()}]")
print(f"W2 int8 : {W2_int8.shape}  [{W2_int8.min()}, {W2_int8.max()}]")

# --- binary files for DMA loading on PYNQ ---
p = os.path.join(OUT_DIR, "weights_l1_final.bin")
with open(p, 'wb') as f:
    f.write(W1_int8.flatten().tobytes())
print(f"\n[BIN] weights_l1_final.bin  {os.path.getsize(p)} bytes")

p = os.path.join(OUT_DIR, "weights_l2_final.bin")
with open(p, 'wb') as f:
    f.write(W2_int8.flatten().tobytes())
print(f"[BIN] weights_l2_final.bin  {os.path.getsize(p)} bytes")

# --- npy files for verify.py ---
np.save(os.path.join(OUT_DIR, "weights_l1.npy"),      W1)
np.save(os.path.join(OUT_DIR, "weights_l2.npy"),      W2)
np.save(os.path.join(OUT_DIR, "weights_l1_int8.npy"), W1_int8)
np.save(os.path.join(OUT_DIR, "weights_l2_int8.npy"), W2_int8)
print("[NPY] weights_l1.npy  weights_l2.npy  (float32)")
print("[NPY] weights_l1_int8.npy  weights_l2_int8.npy  (int8)")

# --- hex files for Vivado $readmemh ---
def write_hex(W_int8, path):
    flat = W_int8.flatten().view(np.uint8)
    with open(path, 'w') as f:
        for b in flat:
            f.write(f"{int(b):02x}\n")

write_hex(W1_int8, os.path.join(OUT_DIR, "weights_l1_hex.txt"))
write_hex(W2_int8, os.path.join(OUT_DIR, "weights_l2_hex.txt"))
print("[HEX] weights_l1_hex.txt  weights_l2_hex.txt")

# --- one pixel file per digit for RTL testbench ---
imgs_all   = test_ds.data.numpy()
labels_all = test_ds.targets.numpy()

for d in range(10):
    idx  = np.where(labels_all == d)[0][0]
    pix  = imgs_all[idx].flatten()
    path = os.path.join(OUT_DIR, f"digit{d}_pix.txt")
    with open(path, 'w') as f:
        for b in pix:
            f.write(f"{int(b):02x}\n")

print("[HEX] digit0_pix.txt .. digit9_pix.txt")
