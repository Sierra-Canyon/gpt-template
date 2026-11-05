# names/bigrams.py  (T13)
# Run from the repo root:  uv run python names/bigrams.py
# The first makemore segment on YOUR names: every pair of neighbouring characters in every name, with <S> before the
# first letter and <E> after the last, counted in a Python dictionary, then the same counts in a torch tensor.
# Nothing to edit in this file.
import torch

words = open("names/names.txt", encoding="utf-8").read().splitlines()
print(f"{len(words)} names   shortest {min(len(w) for w in words)}   longest {max(len(w) for w in words)}")
print("first five:", words[:5])

# 1. The dictionary way, from the video: a pair of characters is the key, how often it occurs is the value.
b = {}
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]              # the start and end markers, one each
    for ch1, ch2 in zip(chs, chs[1:]):             # zip pairs each character with the next one
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1           # get(key, 0): the count so far, or 0 the first time
print(f"{len(b)} distinct bigrams, {sum(b.values())} in total")
print("the ten most common:")
for bigram, count in sorted(b.items(), key=lambda kv: -kv[1])[:10]:
    print(f"   {bigram[0]:>3} {bigram[1]:<3} {count}")

# 2. The tensor way: a row per first character, a column per second, 28 x 28 with the two markers.
chars = sorted(set("".join(words)))                # every letter that appears in your names, a to z usually
stoi = {s: i for i, s in enumerate(chars)}         # string to integer
stoi["<S>"] = len(chars)
stoi["<E>"] = len(chars) + 1
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
print(f"N: shape {tuple(N.shape)}   sum {int(N.sum())}   (the sum equals the dictionary's total)")
print(f"row <S>: starts with the most common first letter {chars[int(N[stoi['<S>'], :len(chars)].argmax())]!r} ({int(N[stoi['<S>']].max())} names)")
