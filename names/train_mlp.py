# names/train_mlp.py  (T18, T19)
# Run from the repo root:  uv run python names/train_mlp.py sweep    (T18 Step 2: 1,000 steps with the learning rate climbing, drawn)
#                     or:  uv run python names/train_mlp.py          (T18 Step 3: train on the training split, score all three splits, save)
#                     or:  uv run python names/train_mlp.py big      (T18 extension: the same with a hidden layer of 300)
# The video's training loop on YOUR names: the names shuffled with the video's seed and cut 80/10/10 into training,
# dev and test splits, minibatches of 32, a learning rate of 0.1 and then 0.01. The trained parameters go to
# names/mlp.pt for embed_plot.py and demo.py. Nothing to edit in this file.
import sys
import random
sys.path.insert(0, "names")
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlp
from mlp import build_dataset, make_parameters, forward, SEED, words

BATCH = 32
STEPS_FAST = 10000                                 # at learning rate 0.1
STEPS_SLOW = 5000                                  # then at 0.01, the "learning rate decay" from the video

random.seed(42)                                    # the video's shuffle seed
names = words[:]
random.shuffle(names)
n1, n2 = int(0.8 * len(names)), int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1])               # 80%: what the network trains on
Xdev, Ydev = build_dataset(names[n1:n2])           # 10%: for choosing settings, never trained on
Xte, Yte = build_dataset(names[n2:])               # 10%: looked at once, at the very end

# In: the parameters and a split.  Out: the loss on the whole split, as a number.
def split_loss(parameters, X, Y):
    with torch.no_grad():                          # no gradients needed, just the number
        return F.cross_entropy(forward(parameters, X), Y).item()

mode = sys.argv[1] if len(sys.argv) > 1 else "train"
if mode == "big":
    mlp.HIDDEN = 300                               # the extension: three times the hidden neurons
g = torch.Generator().manual_seed(SEED)
parameters = make_parameters(g)
print(f"{len(names)} names: train {n1} / dev {n2 - n1} / test {len(names) - n2}   examples {Xtr.shape[0]} / {Xdev.shape[0]} / {Xte.shape[0]}")
print(f"parameters: {sum(p.nelement() for p in parameters)}")

if mode == "sweep":
    lre = torch.linspace(-3, 0, 1000)              # exponents from -3 to 0
    lrs = 10 ** lre                                # learning rates from 0.001 to 1
    lri, lossi = [], []
    for i in range(1000):
        ix = torch.randint(0, Xtr.shape[0], (BATCH,), generator=g)
        loss = F.cross_entropy(forward(parameters, Xtr[ix]), Ytr[ix])
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -lrs[i] * p.grad
        lri.append(lre[i].item()); lossi.append(loss.item())
    plt.figure(figsize=(7, 4))
    plt.plot(lri, lossi)
    plt.xlabel("log10(learning rate)"); plt.ylabel("minibatch loss"); plt.title("1,000 steps, learning rate climbing from 0.001 to 1")
    plt.tight_layout(); plt.savefig("names/lr_sweep.png", dpi=110)
    smooth = [sum(lossi[max(0, i - 25):i + 25]) / len(lossi[max(0, i - 25):i + 25]) for i in range(1000)]
    best = min(range(1000), key=lambda i: smooth[i])
    print(f"drawn: names/lr_sweep.png   lowest smoothed loss {smooth[best]:.3f} at log10(lr) = {lri[best]:.2f}, learning rate {10 ** lri[best]:.3f}")
    for e in (-3, -2, -1, -0.5, 0):
        i = min(range(1000), key=lambda i: abs(lri[i] - e))
        print(f"   log10(lr) {e:5}   loss around there {smooth[i]:.3f}")
    sys.exit()

losses = []
for step in range(STEPS_FAST + STEPS_SLOW):
    ix = torch.randint(0, Xtr.shape[0], (BATCH,), generator=g)      # a minibatch: 32 random examples
    loss = F.cross_entropy(forward(parameters, Xtr[ix]), Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if step < STEPS_FAST else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    losses.append(loss.item())
    if step % 2500 == 0 or step == STEPS_FAST + STEPS_SLOW - 1:
        print(f"step {step:5d}   lr {lr}   minibatch loss {loss.item():.4f}   train {split_loss(parameters, Xtr, Ytr):.4f}   dev {split_loss(parameters, Xdev, Ydev):.4f}")

print(f"final   train {split_loss(parameters, Xtr, Ytr):.4f}   dev {split_loss(parameters, Xdev, Ydev):.4f}   test {split_loss(parameters, Xte, Yte):.4f}")
plt.figure(figsize=(7, 4))
plt.plot(losses, linewidth=0.5)
plt.xlabel("step"); plt.ylabel("minibatch loss"); plt.title(f"hidden {mlp.HIDDEN}: {STEPS_FAST} steps at 0.1, then {STEPS_SLOW} at 0.01")
plt.tight_layout(); plt.savefig("names/train_curve.png" if mode == "train" else "names/train_curve_big.png", dpi=110)
if mode == "train":
    torch.save([p.detach() for p in parameters], "names/mlp.pt")
    print("saved: names/mlp.pt   drawn: names/train_curve.png")
else:
    print("drawn: names/train_curve_big.png   (not saved: mlp.pt keeps the hidden-100 model)")
