# names/demo.py  (T20, the Semester 1 Practical)
# Run from the repo root:  uv run python names/demo.py            (ten names from the trained model, a new batch every run)
#                     or:  uv run python names/demo.py <seed>     (the same ten every time for that seed)
# The one command that shows the name model working. Loads names/mlp.pt, the model train_mlp.py saved, and samples
# from it. If mlp.pt is missing it says so and stops: train first. Nothing to edit in this file.
import sys
import os
import time
sys.path.insert(0, "names")
import torch
import torch.nn.functional as F
from mlp import forward, BLOCK
from counts import itos, words

if not os.path.exists("names/mlp.pt"):
    print("names/mlp.pt is missing: run  uv run python names/train_mlp.py  first")
    sys.exit(1)
parameters = torch.load("names/mlp.pt")

seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(time.time())
g = torch.Generator().manual_seed(seed)

# In: a generator.  Out: one name from the model, one character at a time until it says "."
def sample_name(g):
    out = []
    context = [0] * BLOCK
    while True:
        probs = F.softmax(forward(parameters, torch.tensor([context])), dim=1)
        ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    return "".join(out)

print(f"name model trained on {len(words)} names mined from data/corpus.txt   seed {seed}")
for _ in range(10):
    name = sample_name(g)
    print(f"   {name:20} {'(a real name from the corpus)' if name in words else ''}")
