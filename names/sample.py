# names/sample.py  (T15)
# Run from the repo root:  uv run python names/sample.py            (twenty names sampled from the counts, then the loss)
#                     or:  uv run python names/sample.py smooth     (T15 extension: the loss with 1, 10 and 100 fake counts added)
# The bigram model is the probability matrix from names/counts.py. Sampling walks it one character at a time from
# ".". The loss is the average negative log likelihood of your real names under it, the video's measure of how
# well the model explains the names it was built from. Nothing to edit in this file.
import sys
sys.path.insert(0, "names")
import torch
from counts import N, V, itos, stoi, words, row_probabilities

SEED = 2147483647                                  # the video's seed, so the samples repeat run to run

# In: a probability matrix and a generator.  Out: one name, sampled a character at a time until "." comes up.
def sample_name(P, g):
    out = []
    ix = 0                                         # start at the "." row
    while True:
        p = P[ix]                                  # the row for the character we are on: what comes next, and how likely
        ix = int(torch.multinomial(p, num_samples=1, replacement=True, generator=g).item())
        if ix == 0:                                # "." means the name ended
            break
        out.append(itos[ix])
    return "".join(out)

# In: a probability matrix.  Out: the average negative log likelihood of every bigram in every real name.
def nll(P):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]]).item()
            n += 1
    return -log_likelihood / n

mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
if __name__ != "__main__":
    pass                                           # imported by net.py for sample_name and nll: print nothing
elif mode == "sample":
    P = row_probabilities(N)
    g = torch.Generator().manual_seed(SEED)
    samples = [sample_name(P, g) for _ in range(20)]
    print("twenty names from the counts:", " ".join(samples))
    real = sum(s in words for s in samples)
    print(f"{real} of the twenty are real names from your list")
    print(f"average negative log likelihood of your {len(words)} names: {nll(P):.4f}")
    print(f"a model that guessed every character uniformly would score {torch.log(torch.tensor(float(V))).item():.4f}")
    # one name's bigrams, the way the video prints them, so the loss has faces
    w = words[0]
    chs = ["."] + list(w) + ["."]
    print(f"{w!r}, bigram by bigram:")
    for ch1, ch2 in zip(chs, chs[1:]):
        p = P[stoi[ch1], stoi[ch2]].item()
        print(f"   {ch1}{ch2}: probability {p:.4f}   -log {-torch.log(torch.tensor(p)).item():.4f}")

elif mode == "smooth":
    for fake in (0, 1, 10, 100):
        P = row_probabilities(N + fake)            # fake counts: every bigram gets `fake` imaginary sightings
        zeros = int((P == 0).sum())
        print(f"fake counts {fake:3d}   bigrams with probability 0: {zeros:3d}   loss {nll(P):.4f}")
