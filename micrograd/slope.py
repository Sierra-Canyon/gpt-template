# micrograd/slope.py  (T07)
# Run from the repo root:  uv run python micrograd/slope.py            (one word's count moves: the slope by nudging and by rule)
#                     or:  uv run python micrograd/slope.py partials   (every word's count, one at a time: the gradient)
#                     or:  uv run python micrograd/slope.py h          (how the nudge slope depends on the size of h)
#                     or:  uv run python micrograd/slope.py far        (extension: how far the slope's straight line stays right)
# The function today is the cosine between chapter 2 and chapter 7 of your corpus, over the 50 most common words
# plus your WORD2, as the count of ONE word in chapter 2 is raised by t. Nothing to edit in this file.
import sys
sys.path.insert(0, "linear")                       # so "from chapter_vectors import ..." works from the repo root
from collections import Counter
import numpy as np
from chapter_vectors import chapters, count_vector, words, WORD2

ranked = [w for w, _ in Counter(words).most_common()]      # every word, most common first
vocab = ranked[:50]                                         # T06's top-50 band
if WORD2 not in vocab:
    vocab.append(WORD2)                                     # your word joins the vocabulary if it is not already there
vA = count_vector(chapters[2], vocab)                       # chapter 2's counts, one per vocabulary word
vB = count_vector(chapters[7], vocab)                       # chapter 7's
k = vocab.index(WORD2)                                      # the position of your word in the vector

# In: a vector a and a vector b.  Out: the cosine of the angle between them (1 means parallel).
def cosine(a, b):
    return float(a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b)))   # @ is NumPy's dot product

# In: a word position j and an amount t.  Out: the cosine after chapter 2's count of that word is raised by t.
def f(j, t):
    a = vA.copy()                                           # copy, so vA itself never changes
    a[j] += t
    return cosine(a, vB)

# In: a word position j.  Out: the exact slope of f at t = 0, by the rule (worked out in the README's Notes).
# Raising a[j] does two things: it adds b[j] to the dot product (the first term) and it makes a longer (the second).
def slope_by_rule(j):
    lenA, lenB = np.sqrt(vA @ vA), np.sqrt(vB @ vB)
    return float(vB[j] / (lenA * lenB) - (vA @ vB) * vA[j] / (lenA ** 3 * lenB))

h = 1e-6
mode = sys.argv[1] if len(sys.argv) > 1 else "one"

if mode == "one":
    print(f"vocabulary: {len(vocab)} words   WORD2 = {WORD2!r} at position {k}")
    print(f"chapter 2 count of {WORD2!r}: {vA[k]:.0f}   chapter 7: {vB[k]:.0f}")
    print(f"cosine(chapter 2, chapter 7) = {f(k, 0):.6f}")
    print(f"raise chapter 2's {WORD2!r} by 1:    {f(k, 1):.6f}")
    print(f"raise it by 100:                     {f(k, 100):.6f}")
    nudge = (f(k, h) - f(k, 0)) / h                         # Karpathy's way: nudge by h, divide by h
    print(f"slope at t = 0   by nudging {nudge: .8f}   by rule {slope_by_rule(k): .8f}")

elif mode == "partials":
    partials = np.array([(f(j, h) - f(j, 0)) / h for j in range(len(vocab))])   # one slope per word: the gradient
    order = np.argsort(partials)                            # positions sorted from most negative to most positive
    print(f"{'word':>10} {'ch2 count':>9} {'ch7 count':>9} {'slope':>12}")
    for j in list(order[-3:][::-1]) + list(order[:3]):      # the three largest, then the three smallest
        print(f"{vocab[j]:>10} {vA[j]:9.0f} {vB[j]:9.0f} {partials[j]: 12.8f}")
    print(f"{WORD2!r}: slope {partials[k]: .8f}   rank {int(np.where(order[::-1] == k)[0][0]) + 1} of {len(vocab)} (1 = raises the cosine most)")
    print(f"positive slopes: {(partials > 0).sum()}   negative: {(partials < 0).sum()}")

elif mode == "h":
    exact = slope_by_rule(k)
    print(f"rule slope for {WORD2!r}: {exact: .10f}")
    print(f"{'h':>7}  {'nudge slope':>14}  {'error':>10}")
    for hh in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12):
        nudge = (f(k, hh) - f(k, 0)) / hh
        print(f"{hh:7.0e}  {nudge: 14.10f}  {abs(nudge - exact):10.2e}")

elif mode == "far":
    slope = slope_by_rule(k)
    base = f(k, 0)
    print(f"cosine now {base:.6f}   slope {slope:.8f} per count of {WORD2!r}")
    print(f"{'t':>6}  {'straight line':>13}  {'real cosine':>12}  {'error':>10}")
    for t in (1, 10, 100, 1000, 10000):
        line = base + slope * t                              # what the slope predicts: keep going straight
        real = f(k, t)                                       # what the cosine does
        print(f"{t:6d}  {line:13.6f}  {real:12.6f}  {abs(line - real):10.6f}")
