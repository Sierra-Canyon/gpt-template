# micrograd/test_engine.py  (T21)
# Run from the repo root:  uv run pytest micrograd -q
# Five derivatives you can do on paper in a line each, asserted against YOUR engine. Each test states the paper answer.
# Nothing to edit in this file.
import math
import sys
sys.path.insert(0, "micrograd")
import pytest
from engine import Value

# His T08 expression: L = (a*b + c) * f. The seven gradients from the video, as one assertion.
def test_karpathy_expression():
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    L = (a * b + c) * f
    L.backward()
    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)

# a arrives at d by two paths, one through the product and one straight in: its gradient is b + 1.
def test_reuse():
    a, b = Value(2.0), Value(-3.0)
    d = a * b + a
    d.backward()
    assert a.grad == pytest.approx(b.data + 1)
    assert b.grad == pytest.approx(a.data)

# tanh as one node and tanh built from exp give the same output and the same gradient.
def test_tanh_one_node_equals_exp_form():
    n1, n2 = Value(0.7), Value(0.7)
    o1 = n1.tanh()
    e = (2 * n2).exp()
    o2 = (e - 1) / (e + 1)
    o1.backward(); o2.backward()
    assert o1.data == pytest.approx(o2.data)
    assert n1.grad == pytest.approx(1 - math.tanh(0.7) ** 2)
    assert n2.grad == pytest.approx(n1.grad)

# q = a / b: dq/da is 1/b and dq/db is -a/b^2.
def test_division():
    a, b = Value(3.0), Value(4.0)
    q = a / b
    q.backward()
    assert a.grad == pytest.approx(1 / 4)
    assert b.grad == pytest.approx(-3 / 16)

# sum() starts from the integer 0, so it needs __radd__, and every term's gradient is 1.
def test_radd_for_sum():
    vals = [Value(1.0), Value(2.0), Value(3.0)]
    s = sum(vals)
    s.backward()
    assert s.data == 6.0 and all(v.grad == 1.0 for v in vals)
