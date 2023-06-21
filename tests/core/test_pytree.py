from functools import partial
from pennylane.core.pytree import HashablePartial


def test_hashable_partial_merges_with_partial():
    def f(a, b, c, d, e, f, g):
        pass

    g = partial(f, 2, d=3)
    h = partial(g, 4, e=5)
    i = HashablePartial(h, 6, f=7)

    assert i.args == (2, 4, 6)
    assert i.keywords == {"d": 3, "e": 5, "f": 7}

    g2 = partial(f, 2, d=3)
    h2 = partial(g2, 4, e=5)
    i2 = HashablePartial(h2, 6, f=7)

    assert i == i2


def test_hashable_partial_merges_with_hashable_partial():
    def f(a, b, c):
        pass

    g = HashablePartial(f, 1)
    h = HashablePartial(g, 2)

    assert h.args == (1, 2)


def test_hashable_partial_repr():
    def f(a, b, c):
        pass

    g = HashablePartial(f, 1)
    assert isinstance(repr(g), str)
