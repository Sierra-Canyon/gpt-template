# GPT — Honors Software Engineering, theory track

**Section C-JD · Period C · Room U 108 · 2026–27**

One repository per project. This is the third, and it carries the theory track from
T01 to the end of the year: linear algebra on your corpus, micrograd, makemore, then the
GPT build. The foundations repo stays where it is; nothing from it moves here except the
two files that fetch your corpus.

Git hooks live in `.git/hooks/` and do not travel with a clone, so the secret guard you
installed in the foundations repo protects the foundations repo and nothing else. Every
project template ships its own `setup.sh` for exactly that reason.

---

## First run

```
./setup.sh
```

That installs Python 3.12, NumPy and Matplotlib, installs a git filter so notebook
output never reaches a commit, installs the secret guard, and runs the environment check.
It ends by asking you to try committing a fake key and watch it get refused. Do that.

Then T01 Step 4 brings your corpus over:

```
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
```

The sha256 in `SOURCE.md` has to match the file it fetched. Same corpus, byte for byte,
from September to June.

## Run every script with `uv run`, from the repo root

```
uv run python linear/chapter_vectors.py
```

From the top of the repo, not from inside `linear/`. The scripts open `data/corpus.txt`
by a path relative to where you are standing, so `cd linear && uv run python
chapter_vectors.py` fails with `FileNotFoundError` even though the file is right there.
Plain `python` uses whatever is first on your PATH and will not find NumPy; the symptom is
`ModuleNotFoundError` on a package you just watched install.

## What is where

| Path | What |
|---|---|
| `linear/` | **T01–T06.** Chapter vectors, transforms, determinants, span, the dot and matrix product you write, the Gram matrix. |
| `micrograd/` | **T07–T13.** Your own `Value` class and the training loop, from the Karpathy lecture, on your corpus. |
| `names/` | **T14–T22.** The bigram and MLP name models, trained on names mined from your corpus. |
| `evidence/T<nn>.md` | One file per assignment with a marked slot for every paste, prediction and reflection answer. `scripts/check_evidence.py T01` tells you which slots are still empty. |
| `scratch/` | Yours. Experiments, one-off scripts, notes to yourself. Committed, not graded. |
| `data/` | Your corpus. Git-ignored except `README.md` and `SOURCE.md`. |
| `scripts/check_env.py` | The environment check. Re-run it whenever something breaks. |
| `scripts/install_secret_guard.sh` | Pre-commit hook that refuses staged keys. |

Every folder above has a README saying what goes in it. Read the one for the assignment
you are on before you start it.

## Rules that apply in every repo you open this year

**Branches.** `dev/<feature>` for work. One pull request per assignment, **jd12** as the
reviewer, and the PR body says where the work is weakest. Your student log is on the
`<your-username>-track` branch from T01 to the end of the semester.

**Type the code the assignment tells you to type.** The shipped scripts are the ones you
read and run. The ones you type (`linear/linalg.py` in T05, the `Value` class in the
micrograd weeks) are the lesson, and pasting them skips it.

**AI is allowed, and you say so.** Every log entry carries an `AI use:` line. Using it
costs you nothing. Not saying so is the only version that is a problem.
