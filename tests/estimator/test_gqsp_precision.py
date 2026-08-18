
"""Tests for GQSPTimeEvolution poly_approx_precision validation (#9618)."""
import pytest
import pennylane.estimator as qre


def test_estimate_requires_poly_approx_precision():
    """Default None precision must raise a clear error, not TypeError on None/int (#9618)."""
    with pytest.raises(ValueError, match="poly_approx_precision"):
        qre.GQSPTimeEvolution(qre.RX(0.1, wires=0), 1.0, 1.0)


def test_estimate_with_explicit_precision():
    op = qre.GQSPTimeEvolution(qre.RX(0.1, wires=0), 1.0, 1.0, 0.01)
    res = qre.estimate(op)
    assert res is not None
