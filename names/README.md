# names — T13 onward

The makemore lectures train a character model on a list of names. Yours trains on names
mined from your corpus, so the first script here is the one that finds them. Run everything
from the repo root with `uv run python names/<name>.py`. The video's own names.txt lives
in `scratch/`, for typing along; nothing in this folder reads it.

| Script | Assignment | What it does |
|---|---|---|
| `mine.py` | T13 | Finds the names in your corpus (capitalized mid-sentence, three or more letters, seen three or more times) and writes `names.txt` and `mined_counts.txt`. |
| `bigrams.py` | T13 | Every pair of neighbouring characters in your names, counted in a dictionary and in a 28×28 tensor with the two markers. |
| `counts.py` | T14 | The 27×27 count matrix with the single `.` marker, drawn to `counts.png`; one row read as a model. `bug` mode: the broadcasting mistake, measured. Imported by the scripts below. |
| `sample.py` | T15 | Twenty names sampled from the counts with the video's seed; the average negative log likelihood of your names. `smooth` mode: fake counts. |
| `net.py` | T16 | One layer of 27 neurons, one-hot in, softmax out, trained on your bigrams; its matrix against the counts; the same twenty samples from both. |

Your own files land here too: `names.txt`, `mined_counts.txt`, `counts.png`.
