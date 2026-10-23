# linear/chapter_vectors.py  (T01)
# Run from the repo root:  uv run python linear/chapter_vectors.py
# Cuts your corpus into ten equal "chapters" by word count and turns each one into a vector of
# two word counts. The one line you edit is WORD2. Every later script in linear/ imports from here.
import re
import numpy as np

WORD1 = "the"
WORD2 = "CHANGE_ME"        # a content word that matters in your corpus and appears in all ten chapters

text = open("data/corpus.txt", encoding="utf-8").read()
words = re.findall(r"[a-z']+", text.lower())      # every run of letters or apostrophes, lowercased: the corpus as a list of words
N = 10
size = len(words) // N                            # words per chapter (// is whole-number division)
chapters = []                                     # ten lists of words
for i in range(N):
    chapters.append(words[i * size:(i + 1) * size])   # words from i*size up to (not including) (i+1)*size

# In: a list of words and a list of the words to count.  Out: a NumPy vector, one count per vocabulary word.
def count_vector(chapter, vocab):
    counts = {}                                   # word -> how many times it appears
    for w in vocab:
        counts[w] = 0
    for w in chapter:
        if w in counts:
            counts[w] += 1
    return np.array([counts[w] for w in vocab], dtype=float)   # the counts in vocab order, as a vector

V2 = np.array([count_vector(ch, [WORD1, WORD2]) for ch in chapters])   # (10, 2): one row per chapter

if __name__ == "__main__":
    import sys
    assert WORD2 != "CHANGE_ME", "open linear/chapter_vectors.py and put your word in WORD2"
    if len(sys.argv) > 1 and sys.argv[1] == "ratios":   # T01 extension: how far apart do the ten arrows point?
        ratio = V2[:, 1] / V2[:, 0]               # WORD2 count over WORD1 count, one per chapter: the slope of each arrow
        for i in range(N):
            print(f"chapter {i}: {WORD2} per {WORD1} = {ratio[i]:.4f}")
        print(f"furthest apart: chapter {ratio.argmin()} (smallest) and chapter {ratio.argmax()} (largest)")
        sys.exit()
    print(len(words), "words,", N, "chapters of", size)
    print(V2)
    if (V2[:, 1] == 0).any():                     # [:, 1] is the second column: every chapter's WORD2 count
        print("a chapter never says", repr(WORD2), ": pick a more common word")

    # Adding and scaling chapters. The count of two chapters glued together is the sum of their counts.
    from collections import Counter
    vocab50 = [w for w, _ in Counter(words).most_common(50)]   # the 50 most common words
    A, B = chapters[2], chapters[7]
    vA, vB = count_vector(A, vocab50), count_vector(B, vocab50)
    print("add:  ", np.array_equal(vA + vB, count_vector(A + B, vocab50)))   # A + B glues two lists; vA + vB adds two vectors
    print("scale:", np.array_equal(2 * vA, count_vector(A + A, vocab50)))
