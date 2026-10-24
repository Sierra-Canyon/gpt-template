# linear/gram.py  (T05, T06)
# Run from the repo root:  uv run python linear/gram.py          (T05: one dot product of two chapters)
#                     or:  uv run python linear/gram.py bands    (T06: all ten chapters against each other, in two vocabulary bands, and the timing)
# Uses YOUR dot and matmul from linear/linalg.py on your chapter vectors. There is nothing to edit in this file.
import sys, time
sys.path.insert(0, "linear")
from collections import Counter
import numpy as np
from chapter_vectors import chapters, count_vector, words
from linalg import dot, matmul

ranked = [w for w, _ in Counter(words).most_common()]      # every word, most common first
vocab = ranked[:50]
vA = count_vector(chapters[2], vocab).tolist()              # .tolist(): a plain Python list, because your dot takes lists
vB = count_vector(chapters[7], vocab).tolist()

d = dot(vA, vB)
cos = d / (dot(vA, vA) ** 0.5 * dot(vB, vB) ** 0.5)         # cosine = dot / (length of A * length of B)
print(f"chapter 2 . chapter 7 = {d:,.0f}   cosine = {cos:.4f}")
print("one-row matmul:", matmul([vA], [[x] for x in vB]))  # a 1x50 times a 50x1 is the same number, wrapped as [[...]]
print("top 10 of the 50:", vocab[:10])

if len(sys.argv) > 1 and sys.argv[1] == "bands":
    N = len(chapters)

    # In: a vocabulary.  Out: the 10x10 matrix of cosines between every pair of chapters, over that vocabulary.
    def cosine_matrix(vocab):
        X = [count_vector(ch, vocab).tolist() for ch in chapters]      # 10 x V: one row per chapter
        Xt = [list(col) for col in zip(*X)]                             # V x 10: the transpose (zip(*X) pairs up the columns)
        G = matmul(X, Xt)                                               # 10 x 10: every chapter dotted with every chapter
        norms = [G[i][i] ** 0.5 for i in range(N)]                      # a chapter dotted with itself is its length squared
        return np.array([[G[i][j] / (norms[i] * norms[j]) for j in range(N)] for i in range(N)])

    for name, band in (("top 50", ranked[:50]), ("ranks 51-250", ranked[50:250])):
        C = cosine_matrix(band)
        off = C[~np.eye(N, dtype=bool)]                                 # every entry except the diagonal (a chapter against itself is always 1)
        i, j = np.unravel_index(np.argmax(C - 2 * np.eye(N)), C.shape)  # the most alike pair, with the diagonal pushed out of the running
        k, l = np.unravel_index(np.argmin(C), C.shape)                  # the least alike pair
        print(f"{name:13} min {off.min():.3f}  mean {off.mean():.3f}  max {off.max():.3f}  most alike {i},{j}  least alike {k},{l}")

    # In: a function that takes no arguments.  Out: the fastest of five timed runs, in seconds.
    def best_of_5(f):
        times = []
        for _ in range(5):
            t = time.perf_counter(); f(); times.append(time.perf_counter() - t)
        return min(times)

    for V in (50, 500, 2000):
        V = min(V, len(ranked))                                         # a corpus with fewer than V distinct words uses all of them, and the row says so
        X = [count_vector(ch, ranked[:V]).tolist() for ch in chapters]
        Xt = [list(col) for col in zip(*X)]
        A = np.array(X)
        tp = best_of_5(lambda: matmul(X, Xt))                           # your matmul, pure Python
        tn = min(best_of_5(lambda: A @ A.T) for _ in range(4))          # NumPy's; the fastest of twenty, because a few microseconds of noise would swing the ratio
        print(f"V={V:5d}  pure Python {tp * 1000:8.2f} ms   NumPy {tn * 1000:7.3f} ms   ratio {tp / tn:7.0f}x"
              f"   same answer: {np.allclose(matmul(X, Xt), A @ A.T)}")
