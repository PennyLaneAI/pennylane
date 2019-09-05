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
Unit tests for :mod:`pennylane.variable`.
"""
import string

import pytest
import numpy.random as nr

from pennylane.variable import Variable


# make test deterministic
nr.seed(42)


def test_variable_str():
    """variable: Tests the positional variable reference string."""
    p = Variable(0)
    assert str(p) == "Variable 0: name = None, "
    assert str(-p) == "Variable 0: name = None,  * -1"
    assert str(1.2 * p * 0.4) == "Variable 0: name = None,  * 0.48"
    assert str(1.2 * p / 2.5) == "Variable 0: name = None,  * 0.48"

    p = Variable(0, name="kw1")
    assert str(p) == "Variable 0: name = kw1, "
    assert str(-p) == "Variable 0: name = kw1,  * -1"
    assert str(1.2 * p * 0.4) == "Variable 0: name = kw1,  * 0.48"
    assert str(1.2 * p / 2.5) == "Variable 0: name = kw1,  * 0.48"


def test_variable_val():
    """variable: Tests the positional variable values.
    """
    # mapping function must yield the correct parameter values
    n = 10
    mul = nr.randn(n)  # parameter multipliers
    par_free = nr.randn(n)  # free parameter values
    Variable.free_param_values = par_free
    var = [Variable(k) for k in range(n)]

    assert all([p == v.val for p, v in zip(par_free, var)])  # basic evaluation
    assert all([m * p == (m * v).val for p, v, m in zip(par_free, var, mul)])  # left scalar mul
    assert all([m * p == (v * m).val for p, v, m in zip(par_free, var, mul)])  # right scalar mul
    assert all([p / m == pytest.approx((v / m).val) for p, v, m in zip(par_free, var, mul)])  # right scalar div
    assert all([-p == (-v).val for p, v in zip(par_free, var)])  # unary minus
    assert all([-m**2 * p == (m * -v * m).val for p, v, m in zip(par_free, var, mul)])  # compound expression


def test_keyword_variable():
    """
    variable: Keyword Variable reference tests.
    """
    # Check for a single kwarg_value
    Variable.kwarg_values = {"kw1": 1.0}
    assert Variable(0, name="kw1").val == 1.0

    names = string.ascii_lowercase
    n = 10
    mul = nr.randn(n)  # parameter multipliers
    par_free = nr.randn(n)  # free parameter values
    Variable.kwarg_values = {k: v for k, v in zip(names, par_free)}
    var = [Variable(0, name=n) for n in names]

    assert all([p == v.val for p, v in zip(par_free, var)])  # basic evaluation
    assert all([m * p == (m * v).val for p, v, m in zip(par_free, var, mul)])  # left scalar mul
    assert all([m * p == (v * m).val for p, v, m in zip(par_free, var, mul)])  # right scalar mul
    assert all([p / m == pytest.approx((v / m).val) for p, v, m in zip(par_free, var, mul)])  # right scalar div
    assert all([-p == (-v).val for p, v in zip(par_free, var)])  # unary minus
    assert all([-m**2 * p == (m * -v * m).val for p, v, m in zip(par_free, var, mul)])  # compound expression
