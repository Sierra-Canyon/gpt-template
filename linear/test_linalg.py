# linear/test_linalg.py  (T05)
# Run from the repo root:  uv run pytest linear -q
# Five tests of your dot and matmul in linear/linalg.py, checked against NumPy. There is nothing to edit here.
import numpy as np
import pytest
from linalg import dot, matmul

def test_dot_small():
    assert dot([1, 2, 3], [4, 5, 6]) == 32

def test_dot_perpendicular():
    assert dot([2, 0], [0, 7]) == 0

def test_square():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert np.allclose(matmul(A, B), np.array(A) @ np.array(B))

def test_non_square():
    A = [[1, 2, 3], [4, 5, 6]]                          # 2 x 3
    B = [[1, 0, 2, 1], [0, 1, 1, 3], [2, 1, 0, 1]]      # 3 x 4
    got = matmul(A, B)
    assert np.array(got).shape == (2, 4)
    assert np.allclose(got, np.array(A) @ np.array(B))

def test_mismatch():
    with pytest.raises(ValueError):                     # a 1x3 times a 2x2 has to raise, not crash somewhere inside
        matmul([[1, 2, 3]], [[1, 2], [3, 4]])
