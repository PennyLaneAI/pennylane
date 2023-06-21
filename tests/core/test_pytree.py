# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane.core.pytree.HashablePartial` class.
"""
# pylint: disable=too-many-arguments,unused-argument,

from functools import partial
from pennylane.core.pytree import HashablePartial


def test_hashable_partial_merges_with_partial():
    """
    Test nesting of partials builds the correct HashablePartial
    (this checks for a bug arising in python 3.10+ only)
    """
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
    """
    Test nesting of HashablePartials
    """
    def f(a, b, c):
        pass

    g = HashablePartial(f, 1)
    h = HashablePartial(g, 2)

    assert h.args == (1, 2)


def test_hashable_partial_repr():
    """tests the repr method does not crash"""
    def f(a, b, c):
        pass

    g = HashablePartial(f, 1)
    assert isinstance(repr(g), str)
