# names/embed_plot.py  (T19)
# Run from the repo root:  uv run python names/embed_plot.py
# Draws the trained embedding table, 27 characters as 27 points in the plane, the video's picture on your names,
# and samples twenty names from the trained MLP with the video's seed. Needs names/mlp.pt from train_mlp.py.
# Nothing to edit in this file.
import sys
sys.path.insert(0, "names")
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp import forward, BLOCK, SEED
from counts import V, itos, words

parameters = torch.load("names/mlp.pt")
C = parameters[0]

plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0], C[:, 1], s=200)
for i in range(V):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white")
plt.grid(True, alpha=0.3)
plt.title("the embedding table: where the net put each character")
plt.savefig("names/embeddings.png", dpi=110, bbox_inches="tight")
print("drawn: names/embeddings.png")

# In: nothing.  Out: how far apart the vowels sit, on average, against how far apart every pair of letters sits.
def spread(indices):
    pts = C[indices]
    d = torch.cdist(pts, pts)
    return d.sum().item() / (len(indices) * (len(indices) - 1))
vowels = [i for i in range(1, V) if itos[i] in "aeiou"]
letters = list(range(1, V))
print(f"average distance between two vowels {spread(vowels):.3f}   between any two letters {spread(letters):.3f}")
centre = C[vowels].mean(0)                         # the middle of the five vowels
far = max(vowels, key=lambda i: (C[i] - centre).norm().item())
rest = [i for i in vowels if i != far]
print(f"the vowel farthest from the others: {itos[far]!r}, {(C[far] - centre).norm().item():.3f} from their middle; the other four average {spread(rest):.3f} apart")

# In: a generator.  Out: one name sampled from the trained MLP, one character at a time.
def sample_name(g):
    out = []
    context = [0] * BLOCK
    while True:
        logits = forward(parameters, torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return "".join(out)

g = torch.Generator().manual_seed(SEED + 10)
samples = [sample_name(g) for _ in range(20)]
print("twenty names from the MLP:", " ".join(samples))
print(f"{sum(s in words for s in samples)} of the twenty are real names from your list")
