# names/net.py  (T16)
# Run from the repo root:  uv run python names/net.py
# The second half of the makemore video on YOUR names: one layer of 27 neurons with no bias and no tanh, fed one-hot
# characters, its outputs turned into probabilities by softmax, trained by gradient descent to make the real next
# characters likely. It ends up learning the same matrix the counts gave for free, and the script shows how close.
# Nothing to edit in this file.
import sys
sys.path.insert(0, "names")
import torch
import torch.nn.functional as F
from counts import N, V, itos, stoi, words, row_probabilities
from sample import sample_name, nll, SEED

STEPS = 100
LR = 50                                            # the video's learning rate for this model, large because the loss is an average

# 1. The dataset: every bigram as an input character index and a target character index
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs, ys = torch.tensor(xs), torch.tensor(ys)
num = xs.nelement()
print(f"{num} bigrams from {len(words)} names")

# 2. The network: W is V x V, one row per input character, one column per output character
g = torch.Generator().manual_seed(SEED)
W = torch.randn((V, V), generator=g, requires_grad=True)
xenc = F.one_hot(xs, num_classes=V).float()        # each input character as a row of zeros with a single 1

for step in range(STEPS):
    logits = xenc @ W                              # the forward pass: picking out the row of W for each input character
    counts = logits.exp()                          # softmax, step 1: make every number positive
    probs = counts / counts.sum(1, keepdim=True)   # softmax, step 2: make every row add up to 1
    loss = -probs[torch.arange(num), ys].log().mean()      # the probability the net gave the real next character, averaged
    W.grad = None                                  # zero the gradient (PyTorch's way)
    loss.backward()
    W.data += -LR * W.grad
    if step % 10 == 0 or step == STEPS - 1:
        print(f"step {step:3d}   loss {loss.item():.4f}")

# 3. What it learned, next to the counts
P_counts = row_probabilities(N)
P_net = (W.exp() / W.exp().sum(1, keepdim=True)).detach()
diff = (P_net - P_counts).abs()
r = stoi["u"] if "u" in stoi else 1
print(f"counts loss {nll(P_counts):.4f}   net loss {nll(P_net):.4f}")
print(f"largest difference between the net's probabilities and the counts': {diff.max().item():.4f}   average {diff.mean().item():.4f}")
i, j = divmod(int(diff.argmax()), V)
print(f"   at row {itos[i]!r} column {itos[j]!r}: counts say {P_counts[i, j].item():.4f}, net says {P_net[i, j].item():.4f}, N there is {int(N[i, j])}")

g = torch.Generator().manual_seed(SEED)
print("twenty names from the net:   ", " ".join(sample_name(P_net, g) for _ in range(20)))
g = torch.Generator().manual_seed(SEED)
print("the same twenty from counts: ", " ".join(sample_name(P_counts, g) for _ in range(20)))
