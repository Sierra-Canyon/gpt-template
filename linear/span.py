# linear/span.py  (T04)
# Run from the repo root:  uv run python linear/span.py
# Is chapter 5 in the span of chapters 2 and 7? With two words, two chapter vectors span the whole plane,
# so the answer is yes and solve() finds the two coefficients. With more words there are more equations
# than unknowns and lstsq finds the closest point instead. Then a singular system you built yourself.
# There is nothing to edit in this file.
import sys
sys.path.insert(0, "linear")
from collections import Counter
import numpy as np
from chapter_vectors import chapters, count_vector, words, WORD1, WORD2

A, B, C = chapters[2], chapters[7], chapters[5]

# 2 words: two equations, two unknowns
vocab = [WORD1, WORD2]
M = np.column_stack([count_vector(A, vocab), count_vector(B, vocab)])   # chapter 2 and chapter 7 as the two columns
vC = count_vector(C, vocab)
coef = np.linalg.solve(M, vC)                                            # the a and b with a*col0 + b*col1 = vC
print("2 words  coefficients", coef.round(3), " check", (M @ coef).round(3), "vs", vC)

# more words: more equations than unknowns, so ask for the closest point instead
ranked = [w for w, _ in Counter(words).most_common()]                    # every word, most common first; ranked[0] is "the", so ranked[1] is the next most common word
for vocab in ([WORD1, WORD2, ranked[1]], ranked[:10], ranked[:50]):
    M = np.column_stack([count_vector(A, vocab), count_vector(B, vocab)])
    vC = count_vector(C, vocab)
    coef, *_ = np.linalg.lstsq(M, vC, rcond=None)                        # least squares: the a, b that get closest. lstsq returns four things; *_ throws away all but the first. rcond=None silences a warning
    rel = np.linalg.norm(M @ coef - vC) / np.linalg.norm(vC)             # how far the closest point is, as a fraction of vC's length
    print(f"{len(vocab):2d} words  coefficients {coef.round(3)}  relative miss {rel:.4f}")

# a singular system you built yourself: chapter 2, and chapter 2 twice
M = np.column_stack([count_vector(A, [WORD1, WORD2]), count_vector(A + A, [WORD1, WORD2])])
print("singular system\n", M, "\n rank", np.linalg.matrix_rank(M), " det", np.linalg.det(M))
try:
    print(" solve returned", np.linalg.solve(M, count_vector(C, [WORD1, WORD2])))
except np.linalg.LinAlgError as e:
    print(" LinAlgError:", e)
