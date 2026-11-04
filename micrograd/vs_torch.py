# micrograd/vs_torch.py  (T10)
# Run from the repo root:  uv run python micrograd/vs_torch.py
# The T09 neuron twice more: once with tanh built out of exp and division instead of as one node, and once in
# PyTorch, which has never heard of your Value class. All three have to agree. Nothing to edit in this file.
import sys
sys.path.insert(0, "micrograd")
import torch
from engine import Value
from features import chapter_features
from neuron import CHAPTER, W1, W2, B, build

f1, f2 = chapter_features(CHAPTER)

# 1. Your engine, tanh as one node (the T09 neuron), gradients by o.backward()
v = build()
v["o"].backward()

# 2. Your engine, tanh built from exp and division: the same function, a longer graph
x1, x2 = Value(f1), Value(f2)
w1, w2, b = Value(W1), Value(W2), Value(B)
n = x1 * w1 + x2 * w2 + b
e = (2 * n).exp()
o2 = (e - 1) / (e + 1)
o2.backward()

# 3. PyTorch, in 64-bit floats so the comparison is to six decimals and not five
xt = torch.tensor([f1, f2], dtype=torch.double, requires_grad=True)
wt = torch.tensor([W1, W2], dtype=torch.double, requires_grad=True)
bt = torch.tensor(B, dtype=torch.double, requires_grad=True)
ot = torch.tanh((xt * wt).sum() + bt)
ot.backward()

rows = [("o",     v["o"].data,   o2.data,  ot.item()),
        ("do/db", v["b"].grad,   b.grad,   bt.grad.item()),
        ("do/dw1", v["w1"].grad, w1.grad,  wt.grad[0].item()),
        ("do/dw2", v["w2"].grad, w2.grad,  wt.grad[1].item()),
        ("do/dx1", v["x1"].grad, x1.grad,  xt.grad[0].item()),
        ("do/dx2", v["x2"].grad, x2.grad,  xt.grad[1].item())]
print(f"{'':7} {'tanh node':>12} {'from exp':>12} {'torch':>12}   all within 5e-7")
for name, one, two, three in rows:
    ok = abs(one - two) < 5e-7 and abs(one - three) < 5e-7
    print(f"{name:7} {one: 12.6f} {two: 12.6f} {three: 12.6f}   {ok}")
