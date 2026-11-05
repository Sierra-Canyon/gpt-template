# micrograd — T07 onward

The Karpathy micrograd lecture has you build a `Value` class that records every operation
and runs backpropagation through it. That class is `engine.py`, **typed by you** as he
types it, one segment per assignment, and every other script in this folder runs on it.
Run everything from the repo root with `uv run python micrograd/<name>.py`.

| Script | Assignment | What it does |
|---|---|---|
| `engine.py` | T08–T10 | **You type this one**: `Value`, from the video. `+` and `*` in T08, `tanh` and `_backward` in T09, `backward()` and the rest of the operations in T10. |
| `slope.py` | T07 | The cosine between chapters 2 and 7 as one word's count moves: the slope by nudging and by rule; `partials`, `h` and `far` modes. |
| `viz.py` | T08 | `draw_dot` from the lecture, and `print_graph` for a machine without the `dot` program. |
| `expr.py` | T08 | T01's two-word dot product as an expression graph of your `Value`s; `grads` mode checks your hand page by nudging. |
| `features.py` | T09 | Two features per chapter from T01's counts: how unusual each chapter's rate of `WORD1` and `WORD2` is. |
| `neuron.py` | T09, T10 | One neuron reads chapter 2; `grads` calls each `_backward` by hand in order, `step` takes one optimization step, `backward` lets `o.backward()` do it. |
| `vs_torch.py` | T10 | The neuron three ways: `tanh` as one node, `tanh` from `exp`, and PyTorch. All three agree or something is wrong. |
| `nn.py` | T11 | **You type this one**: `Neuron`, `Layer`, `MLP`, from the video. |
| `slices.py` | T11, T12 | Forty 2,000-word slices of your corpus, four per chapter, as two features and a label: which half of the corpus. |
| `train.py` | T11, T12 | `before`: one forward pass of the untrained net and its loss. Plain: 100 steps of gradient descent, the curve, the misses. `noreset`: the same without zeroing the gradients. |
| `test_engine.py` | T21 | Five derivatives with paper answers, asserted against your engine. `uv run pytest micrograd -q`. |
| `chain.py` | T21 | Two neurons in a chain on chapter 2: the check for the milestone derivation page. |

Your own files land here too: the `.svg` drawings from T08, the loss curves from T12, and `MILESTONE.md` from T21.
