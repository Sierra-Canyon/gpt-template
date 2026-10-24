# linear — T01 to T06

One script per idea, each run from the repo root with `uv run python linear/<name>.py`.
Every one of them imports your chapter vectors from `chapter_vectors.py`, so T01's `WORD2`
line is the one edit that everything else depends on.

| Script | Assignment | What it does |
|---|---|---|
| `chapter_vectors.py` | T01 | Cuts the corpus into ten chapters and counts two words in each. `ratios` mode: the slope of each chapter's arrow. |
| `landing.py` | T02 | Checks your three hand matrices, then sends the ten chapters through each one. |
| `linearity.py` | T02 | Which of four ways of processing a count vector are linear. |
| `transform_viz.py` | T03, T04 | Draws what a 2×2 matrix does to the grid; eight numbers draw `AB` against `BA`. |
| `compose3.py` | T03 | Your three matrices in all six orders. |
| `span.py` | T04 | Is chapter 5 in the span of chapters 2 and 7; a singular system from your own counts. |
| `dependent.py` | T04 | The most nearly parallel pair of chapters. |
| `linalg.py` | T05 | **You type this one**: `dot` and `matmul` from lists. |
| `test_linalg.py` | T05 | Five tests for your `linalg.py`. |
| `gram.py` | T05, T06 | Your `dot` on two chapters; `bands` mode: every pair, two vocabularies, and the timing against NumPy. |

Your own files land here too: `NOTATION.md`, the video-notes photos, `MATRICES.md`, `FINDINGS.md`.
