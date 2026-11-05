# names/mine.py  (T13)
# Run from the repo root:  uv run python names/mine.py
# Mines the names out of your corpus and writes them to names/names.txt, one per line, lowercase, the way the
# makemore lecture's names.txt is laid out. A "name" here is a word that starts with a capital letter when it is
# NOT the first word of a sentence, has only letters after that capital, is at least MIN_LEN letters long, and shows
# up at least MIN_COUNT times that way. The rule is simple on purpose. It catches most names and some other words
# (a title like King, a place, a people), and T13 Step 3 looks at what leaked in. Nothing to edit in this file.
from collections import Counter

MIN_LEN = 3
MIN_COUNT = 3
ENDS_SENTENCE = ".!?"                              # a word ending in one of these ends a sentence
QUOTES = "\"'“”‘’"                                 # stripped off the ends of a word before looking at it

text = open("data/corpus.txt", encoding="utf-8").read()
counts = Counter()
sentence_start = True                              # the very first word of the corpus starts a sentence
for raw in text.split():                           # split on any whitespace: one token at a time
    word = raw.strip(QUOTES + ".,;:!?()[]")        # punctuation on the ends is not part of the word
    if word == "":
        continue
    if raw[0] in QUOTES:
        sentence_start = True                      # a word that opens a quotation starts the quoted sentence: he said, "The ..."
    looks_like_name = (word[0].isupper() and word[1:].islower() and word[1:].isalpha() and len(word) >= MIN_LEN)
    if looks_like_name and not sentence_start:
        counts[word.lower()] += 1
    sentence_start = raw.rstrip(QUOTES)[-1:] in ENDS_SENTENCE    # does this token end the sentence for the next one

names = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
with open("names/names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(names) + "\n")
with open("names/mined_counts.txt", "w", encoding="utf-8") as f:       # the same names with how often each was seen, most common first
    for w, c in counts.most_common():
        if c >= MIN_COUNT:
            f.write(f"{w} {c}\n")

if __name__ == "__main__":
    print(f"{len(counts)} capitalized mid-sentence words, {len(names)} seen {MIN_COUNT} or more times -> names/names.txt")
    print("most common:", [w for w, _ in counts.most_common(15)])
    lengths = [len(w) for w in names]
    print(f"shortest {min(lengths)}   longest {max(lengths)} ({max(names, key=len)})   average {sum(lengths) / len(lengths):.1f}")
    print("first ten alphabetically:", names[:10])
