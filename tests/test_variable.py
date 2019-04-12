# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` utility classes :class:`ParRef`, :class:`Command`.
"""
import logging as log
from string import ascii_lowercase

import numpy as np
import numpy.random as nr

from defaults import pennylane, BaseTest
from pennylane.variable import Variable


def test_variable_str():
    """variable: Tests the positional variable reference string."""
    n = 10
    m = nr.randn(n)  # parameter multipliers
    par_fixed = nr.randn(n)  # fixed parameter values
    par_free = nr.randn(n)  # free parameter values
    Variable.free_param_values = par_free

    # test __str__()
    p = Variable(0)
    assert str(p) == "Variable 0: name = None, "
    assert str(-p) == "Variable 0: name = None,  * -1"
    assert str(1.2 * p * 0.4) == "Variable 0: name = None,  * 0.48"
    assert str(1.2 * p / 2.5) == "Variable 0: name = None,  * 0.48"


def test_variable_val():
    """variable: Tests the positional variable values.
    """
    # mapping function must yield the correct parameter values
    n = 10
    m = nr.randn(n)  # parameter multipliers
    par_fixed = nr.randn(n)  # fixed parameter values
    par_free = nr.randn(n)  # free parameter values
    Variable.free_param_values = par_free
    assert [(par_free[k] == m[k] * Variable(k)) for k in range(n)]
    assert [(par_free[k] == Variable(k) * m[k]) for k in range(n)]
    assert [(-par_free[k] * m ** 2) == m[k] * (-Variable(k)) * m[k] for k in range(n)]
    assert [(par_fixed[k] == par_fixed[k]) for k in range(n)]


def test_keyword_variable():
    """
    variable: Keyword Variable reference tests.\
    """
    n = 10
    m = nr.randn(n)  # parameter multipliers
    par_fixed = nr.randn(n)  # fixed parameter values
    par_free = nr.randn(n)  # free parameter values
    Variable.kwarg_values = {k: v for k, v in zip(ascii_lowercase, par_free)}

    # test __str__()
    p = Variable(0, name="kw1")
    assert str(p) == "Variable 0: name = kw1, "
    assert str(-p) == "Variable 0: name = kw1,  * -1"
    assert str(1.2 * p * 0.4) == "Variable 0: name = kw1,  * 0.48"
    assert str(1.2 * p / 2.5) == "Variable 0: name = kw1,  * 0.48"

    # mapping function must yield the correct parameter values
    assert (
        (par_free[k] * m[k]) == (m[k] * Variable(name=n))
        for (k, n) in zip(range(n), ascii_lowercase)
    )
    assert (
        (par_free[k] * m[k]) == (Variable(name=n) * m[k])
        for (k, n) in zip(range(n), ascii_lowercase)
    )
    assert (
        (-par_free[k] * m[k]) == (-Variable(name=n))
        for (k, n) in zip(range(n), ascii_lowercase)
    )
    assert (
        (-par_free[k] * m[k] ** 2) == (m[k] * -Variable(name=n) * m[k])
        for (k, n) in zip(range(n), ascii_lowercase)
    )
    # fixed values remain constant
    assert [(par_fixed[k] == par_fixed[k]) for k in range(n)]
