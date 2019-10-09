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
import pytest
import numpy.random as nr

from pennylane.variable import Variable


# make test deterministic
nr.seed(42)

n = 10
keyword_par_names = ['foo', 'bar']
par_inds = [0, 9]
par_mults = [1, 0.4, -2.7]


@pytest.fixture(scope="function")
def par_positional():
    "QNode: positional parameters"
    temp = nr.randn(n)
    Variable.free_param_values = temp  # set the values
    return temp

@pytest.fixture(scope="function")
def par_keyword():
    "QNode: keyword parameters"
    temp = {name: nr.randn(n) for name in keyword_par_names}
    Variable.kwarg_values = temp  # set the values
    return temp


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


def variable_eval_asserts(v, p, m, tol):
    """Check that variable evaluation (with scalar multiplication) yields the expected results."""
    assert v.val == p  # normal evaluation
    assert (m * v).val == m * p  # left scalar mul
    assert (v * m).val == m * p  # right scalar mul
    assert (v / m).val == pytest.approx(p / m, abs=tol)  # right scalar div
    assert (-v).val == -p   # unary minus
    assert (m * -v * m).val == -m**2 * p  # compound expression


@pytest.mark.parametrize("ind", par_inds)
@pytest.mark.parametrize("mult", par_mults)
def test_variable_val(par_positional, ind, mult, tol):
    """Positional variable evaluation."""
    v = Variable(ind)

    assert v.name is None
    assert v.mult == 1
    assert v.idx == ind
    variable_eval_asserts(v, par_positional[ind], mult, tol)


@pytest.mark.parametrize("ind", par_inds)
@pytest.mark.parametrize("mult", par_mults)
@pytest.mark.parametrize("name", keyword_par_names)
def test_keyword_variable(par_keyword, name, ind, mult, tol):
    """Keyword variable evaluation."""
    v = Variable(ind, name)

    assert v.name == name
    assert v.mult == 1
    assert v.idx == ind
    variable_eval_asserts(v, par_keyword[name][ind], mult, tol)
