# names/counts.py  (T14, T15, T16)
# Run from the repo root:  uv run python names/counts.py          (T14: the 27x27 count matrix on your names, drawn, and one row of it)
#                     or:  uv run python names/counts.py bug      (T14 extension: normalize the rows with the broadcasting bug and see what breaks)
# Also imported by sample.py and net.py for the matrix and the character tables. The one marker is "." for both
# start and end, as the video does from 20:54 on. Nothing to edit in this file.
import sys
import torch

words = open("names/names.txt", encoding="utf-8").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}     # letters are 1 to 26 (or however many your names use)
stoi["."] = 0                                      # the marker is 0, so it is the first row and column
itos = {i: s for s, i in stoi.items()}             # integer back to string
V = len(stoi)                                      # the vocabulary size: 27 for a to z plus the marker

N = torch.zeros((V, V), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# In: the count matrix.  Out: the probability matrix, every row divided by its own sum so each row adds up to 1.
def row_probabilities(N):
    P = N.float()
    P = P / P.sum(1, keepdim=True)                 # keepdim keeps the sums as a column (V x 1), so each row is divided by its own sum
    return P

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "draw"
    print(f"{len(words)} names   vocabulary {V}: {''.join(chars)} and '.'")
    print(f"N: shape {tuple(N.shape)}   sum {int(N.sum())}   N[0, 0] = {int(N[0, 0])} (a name cannot be empty, so . never follows .)")

    if mode == "draw":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 16))
        plt.imshow(N, cmap="Blues")
        for i in range(V):
            for j in range(V):
                plt.text(j, i, itos[i] + itos[j], ha="center", va="bottom", color="gray", fontsize=7)
                plt.text(j, i, int(N[i, j]), ha="center", va="top", color="gray", fontsize=7)
        plt.axis("off")
        plt.savefig("names/counts.png", dpi=80, bbox_inches="tight")
        print("drawn: names/counts.png")

        P = row_probabilities(N)
        most_common = open("names/mined_counts.txt", encoding="utf-8").readline().split()[0]   # mine.py's first line: the name seen most
        r = stoi[most_common[0]]                                         # the row for its first letter
        order = P[r].argsort(descending=True)
        print(f"most common name in your corpus: {most_common!r}, first letter {most_common[0]!r}, row {r}")
        print(f"row {most_common[0]!r} sums to {float(P[r].sum()):.4f}; after {most_common[0]!r} the five likeliest next characters:")
        for j in order[:5]:
            print(f"   {itos[int(j)]!r}  count {int(N[r, j]):4d}   probability {float(P[r, j]):.3f}")

    elif mode == "bug":
        P_right = row_probabilities(N)
        P_wrong = N.float() / N.float().sum(1)      # the bug from 48:00: without keepdim the sums are a plain vector of length V,
                                                   # which broadcasts along the rows, so every COLUMN is divided by the wrong row's sum
        print(f"right: row 1 sums to {float(P_right[1].sum()):.4f}   row 2 sums to {float(P_right[2].sum()):.4f}")
        print(f"wrong: row 1 sums to {float(P_wrong[1].sum()):.4f}   row 2 sums to {float(P_wrong[2].sum()):.4f}")
        print(f"wrong: column 1 sums to {float(P_wrong[:, 1].sum()):.4f}   (the columns add to 1 instead of the rows: the giveaway)")
        print("shapes: N.sum(1, keepdim=True) is", tuple(N.sum(1, keepdim=True).shape), "  N.sum(1) is", tuple(N.sum(1).shape))
