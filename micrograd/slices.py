# micrograd/slices.py  (T11, T12)
# Run from the repo root:  uv run python micrograd/slices.py
# The dataset the micrograd network trains on: 40 slices of your corpus, each 2,000 words, four from every one of
# T01's ten chapters, spaced evenly through the chapter. Each slice is the same two features as T09 (how unusual
# its rate of WORD1 and of WORD2 is, against the chapter means and spreads from features.py), and a label: -1 if
# the slice came from the first five chapters, +1 if from the last five. The network's job is to tell the two halves
# of your corpus apart from two numbers. Nothing to edit in this file.
import sys
sys.path.insert(0, "linear")
sys.path.insert(0, "micrograd")
from chapter_vectors import chapters, count_vector, WORD1, WORD2
from features import mean, std

SLICE = 2000                                       # words per slice
PER_CHAPTER = 4                                    # slices per chapter

# In: nothing.  Out: three lists of the same length: the feature pairs, the labels, and (chapter, start) for each slice.
def load():
    xs, ys, where = [], [], []
    for c, chapter in enumerate(chapters):
        step = (len(chapter) - SLICE) // (PER_CHAPTER - 1)      # so the four slices spread from the start to the end
        for i in range(PER_CHAPTER):
            start = i * step
            words = chapter[start:start + SLICE]
            counts = count_vector(words, [WORD1, WORD2]) / SLICE * 1000     # the two rates per 1,000 words
            x = [float((counts[0] - mean[0]) / std[0]), float((counts[1] - mean[1]) / std[1])]
            xs.append(x)
            ys.append(-1.0 if c < 5 else 1.0)
            where.append((c, start))
    return xs, ys, where

if __name__ == "__main__":
    xs, ys, where = load()
    print(f"{len(xs)} slices of {SLICE} words, {PER_CHAPTER} per chapter   label -1: chapters 0-4   label +1: chapters 5-9")
    print(f"{'chapter':>7} {'start':>6} {'x1':>7} {'x2':>7} {'y':>4}")
    for (c, start), x, y in zip(where, xs, ys):
        print(f"{c:7d} {start:6d} {x[0]:7.2f} {x[1]:7.2f} {y:+4.0f}")
