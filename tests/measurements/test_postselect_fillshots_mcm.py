import pennylane as qml
from pennylane import numpy as np


def test_fillshots_preserves_correlation_rx():
    """With two correlated MCMs and postselection on the second,
    'fill-shots' must preserve correlations—i.e., '11' should
    significantly dominate '01'."""
    dev = qml.device("default.qubit")

    @qml.set_shots(1000)
    @qml.qnode(dev, postselect_mode="fill-shots")
    def circ():
        qml.H(0)
        m0 = qml.measure(0)
        qml.RX(np.pi / 4, 0)
        m1 = qml.measure(0, postselect=1)
        return qml.counts([m0, m1])

    counts = circ()
    c01 = counts.get("01", 0)
    c11 = counts.get("11", 0)

    # With 1000 shots, leave a healthy margin for randomness.
    assert c11 > c01 + 200


def test_fillshots_extreme_case_no_unphysical_01():
    """If the first MCM collapses the state, postselecting the second
    on '1' should yield almost exclusively '11' outcomes under
    'fill-shots' (no '01' mass if correlations are preserved)."""
    dev = qml.device("default.qubit")

    @qml.set_shots(1000)
    @qml.qnode(dev, postselect_mode="fill-shots")
    def circ():
        qml.H(0)
        m0 = qml.measure(0)
        m1 = qml.measure(0, postselect=1)
        return qml.counts([m0, m1])

    counts = circ()
    c01 = counts.get("01", 0)
    c11 = counts.get("11", 0)

    # Allow a tiny tolerance for randomness; main mass should be on "11".
    assert c01 <= 5
    assert c11 >= 900


def test_fillshots_single_postselect_regression():
    """A single postselected MCM should behave as before—this guards against
    unintended regressions when only one MCM is present."""
    dev = qml.device("default.qubit")

    @qml.set_shots(800)
    @qml.qnode(dev, postselect_mode="fill-shots")
    def circ():
        qml.H(0)  # 50/50 on the first measurement basis
        m1 = qml.measure(0, postselect=1)
        return qml.counts(m1)

    counts = circ()
    # With postselect on 1, nearly all kept shots should be "1".
    assert counts.get(1.0, 0) >= 760  # ~95% of 800; generous margin
