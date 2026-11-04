# micrograd/neuron.py  (T09, T10)
# Run from the repo root:  uv run python micrograd/neuron.py           (the forward pass: one neuron reads chapter 2)
#                     or:  uv run python micrograd/neuron.py grads     (T09 Step 4: each node's _backward called by hand, in order)
#                     or:  uv run python micrograd/neuron.py step      (T09 Step 5: one optimization step, the output moves)
#                     or:  uv run python micrograd/neuron.py backward  (T10 Step 3: o.backward() does the whole thing)
# One neuron with two inputs, the two features of chapter 2 from micrograd/features.py, built out of YOUR Value
# objects. The weights are fixed numbers today so that your hand page and the engine see the same neuron.
# Nothing to edit in this file.
import sys
sys.path.insert(0, "micrograd")
from engine import Value
from features import chapter_features, WORD1, WORD2

CHAPTER = 2
f1, f2 = chapter_features(CHAPTER)
W1, W2, B = 0.5, -0.3, 0.1                         # the weights every student uses today

# In: nothing.  Out: the neuron's nodes, built fresh, as a dictionary by label.
def build():
    x1, x2 = Value(f1, label="x1"), Value(f2, label="x2")
    w1, w2 = Value(W1, label="w1"), Value(W2, label="w2")
    b = Value(B, label="b")
    p1 = x1 * w1; p1.label = "p1"                  # x1 * w1
    p2 = x2 * w2; p2.label = "p2"                  # x2 * w2
    s = p1 + p2;  s.label = "s"
    n = s + b;    n.label = "n"                    # the raw sum
    o = n.tanh(); o.label = "o"                    # squashed to between -1 and 1
    return {"x1": x1, "x2": x2, "w1": w1, "w2": w2, "b": b, "p1": p1, "p2": p2, "s": s, "n": n, "o": o}

# In: the node dictionary.  Out: nothing. Prints every node's gradient, output first, leaves last.
def print_grads(v):
    for name in ("o", "n", "s", "b", "p1", "p2", "x1", "w1", "x2", "w2"):
        print(f"do/d{name:2} = {v[name].grad: .4f}")

if __name__ == "__main__":                         # only when run directly, not when vs_torch.py imports build()
    mode = sys.argv[1] if len(sys.argv) > 1 else "forward"
    v = build()
    print(f"chapter {CHAPTER}: x1 ({WORD1}) = {f1}   x2 ({WORD2}) = {f2}   w1 = {W1}  w2 = {W2}  b = {B}")
    print(f"p1 = {v['p1'].data:.4f}   p2 = {v['p2'].data:.4f}   s = {v['s'].data:.4f}   n = {v['n'].data:.4f}   o = tanh(n) = {v['o'].data:.4f}")

    if mode == "grads":
        v["o"].grad = 1.0                          # the output's gradient with respect to itself
        for name in ("o", "n", "s", "p1", "p2"):   # output first, then each node whose inputs still need their share
            v[name]._backward()
        print_grads(v)

    elif mode == "step":
        v["o"].grad = 1.0
        for name in ("o", "n", "s", "p1", "p2"):
            v[name]._backward()
        lr = 0.01
        for name in ("w1", "w2", "b"):
            v[name].data += lr * v[name].grad      # move each weight a little in the direction that raises o
        o_new = ((v["x1"] * v["w1"] + v["x2"] * v["w2"]) + v["b"]).tanh()
        print(f"o before {v['o'].data:.6f}   after {o_new.data:.6f}   change {o_new.data - v['o'].data:+.6f}")
        print(f"weights now: w1 = {v['w1'].data:.4f}  w2 = {v['w2'].data:.4f}  b = {v['b'].data:.4f}")

    elif mode == "backward":
        v["o"].backward()                          # T10: the engine finds the order itself
        print_grads(v)
