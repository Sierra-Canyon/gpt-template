# micrograd/chain.py  (T21)
# Run from the repo root:  uv run python micrograd/chain.py
# Two neurons in a chain, the smallest network with a hidden layer: the T09 neuron feeds a second neuron with one
# input. Your milestone derivation page works every gradient of o2 by hand, from o2 back to w1, and this script
# is the check. Fixed weights, chapter 2's two features. Nothing to edit in this file.
import sys
sys.path.insert(0, "micrograd")
from engine import Value
from features import chapter_features

x1v, x2v = chapter_features(2)
x1, x2 = Value(x1v, label="x1"), Value(x2v, label="x2")
w1, w2, b1 = Value(0.5, label="w1"), Value(-0.3, label="w2"), Value(0.1, label="b1")
w3, b2 = Value(0.8, label="w3"), Value(-0.2, label="b2")

n1 = x1 * w1 + x2 * w2 + b1; n1.label = "n1"        # the T09 neuron's raw sum
o1 = n1.tanh();              o1.label = "o1"        # its output: the hidden neuron
n2 = o1 * w3 + b2;           n2.label = "n2"        # the second neuron reads it
o2 = n2.tanh();              o2.label = "o2"        # the network's output
o2.backward()

print(f"x1 = {x1v}   x2 = {x2v}   w1 = 0.5  w2 = -0.3  b1 = 0.1   w3 = 0.8  b2 = -0.2")
print(f"n1 = {n1.data:.4f}   o1 = {o1.data:.4f}   n2 = {n2.data:.4f}   o2 = {o2.data:.4f}")
for v in (o2, n2, b2, w3, o1, n1, b1, w1, w2, x1, x2):
    print(f"do2/d{v.label:2} = {v.grad: .4f}")
print(f"the two tanh factors: 1 - o2^2 = {1 - o2.data ** 2:.4f}   1 - o1^2 = {1 - o1.data ** 2:.4f}")
