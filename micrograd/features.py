# micrograd/features.py  (T09)
# Run from the repo root:  uv run python micrograd/features.py
# Two features per chapter, from T01's two word counts: how unusual each chapter's count of WORD1 and of WORD2 is,
# compared with the other nine chapters. A feature of 0 is an average chapter, +1 is one standard deviation above
# average, and -1 is one below. These are the two inputs the T09 neuron reads. Nothing to edit in this file.
import sys
sys.path.insert(0, "linear")
import numpy as np
from chapter_vectors import V2, size, WORD1, WORD2

per_1000 = V2 / size * 1000                        # (10, 2): each chapter's two counts per 1,000 words
mean = per_1000.mean(axis=0)                       # the average chapter, one number per word
std = per_1000.std(axis=0)                         # how much the chapters spread around it
features = (per_1000 - mean) / std                 # (10, 2): every chapter as two z-scores

# In: a chapter number.  Out: its two features, rounded to two decimals so a hand page is doable.
def chapter_features(c):
    return [round(float(v), 2) for v in features[c]]

if __name__ == "__main__":
    print(f"{'chapter':>7} {WORD1 + ' per 1000':>14} {WORD2 + ' per 1000':>14} {'x1':>7} {'x2':>7}")
    for c in range(10):
        x1, x2 = chapter_features(c)
        print(f"{c:7d} {per_1000[c, 0]:14.1f} {per_1000[c, 1]:14.1f} {x1:7.2f} {x2:7.2f}")
    print(f"mean {mean[0]:.1f} and {mean[1]:.1f} per 1,000 words; std {std[0]:.1f} and {std[1]:.1f}")
