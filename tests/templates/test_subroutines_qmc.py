# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import pennylane as qml
from pennylane.templates.subroutines.qmc import probs_to_unitary, func_to_unitary, _make_V, _make_Z, make_Q

class TestProbsToUnitary:
    """Tests for the probs_to_unitary function"""

    def test_non_flat(self):
        """Test if a ValueError is raised when a non-flat array is input"""
        p = np.ones((4, 1))
        with pytest.raises(ValueError, match="The probability distribution must be specified as a"):
            probs_to_unitary(p)

    def test_invalid_distribution_sum_to_not_one(self):
        """Test if a ValueError is raised when a distribution that does not sum to one is input"""
        p = np.ones(4)
        with pytest.raises(ValueError, match="A valid probability distribution of non-negative"):
            probs_to_unitary(p)

    def test_invalid_distribution_negative(self):
        """Test if a ValueError is raised when a distribution with a negative value is input"""
        p = [2, 0, 0, -1]
        with pytest.raises(ValueError, match="A valid probability distribution of non-negative"):
            probs_to_unitary(p)

    ps = [
        [0.46085261032920616, 0.5391473896707938],
        [0.2111821738452515, 0.4235979103670337, 0.36521991578771484],
        [0.3167916924190049, 0.2651843704361695, 0.1871934980886578, 0.23083043905616774],
        [0.8123242419241959, 0.07990911578859018, 0.07983919018902215, 0.027927452098191852],
    ]

    @pytest.mark.parametrize("p", ps)
    def test_fixed_examples(self, p):
        """Test if the correct unitary is returned for fixed input examples. A correct unitary has
        its first column equal to the square root of the distribution and satisfies
        U @ U.T = U.T @ U = I."""
        unitary = probs_to_unitary(p)
        assert np.allclose(np.sqrt(p), unitary[:, 0])
        assert np.allclose(unitary @ unitary.T, np.eye(len(unitary)))
        assert np.allclose(unitary.T @ unitary, np.eye(len(unitary)))


class TestFuncToUnitary:
    """Tests for the func_to_unitary function"""

    def test_not_bounded_func(self):
        """Test if a ValueError is raised if a function that evaluates outside of the [0, 1]
        interval is provided"""
        func = lambda i: np.sin(i)

        with pytest.raises(ValueError, match="func must be bounded within the interval"):
            func_to_unitary(func, 8)

    def test_one_dimensional(self):
        """Test for a one-dimensional example if the returned unitary maps input states to the
        expected output state as well as if the unitary satisfies U @ U.T = U.T @ U = I."""
        M = 8
        func = lambda i: np.sin(i) ** 2

        r = func_to_unitary(func, M)

        for i in range(M):
            # The control qubit is the last qubit, so we have to look at every other term
            # using [::2].
            output_state = r[::2][i]
            output_0 = output_state[::2]
            output_1 = output_state[1::2]
            assert np.allclose(output_0[i], np.sqrt(1 - func(i)))
            assert np.allclose(output_1[i], np.sqrt(func(i)))

        assert np.allclose(r @ r.T, np.eye(2 * M))
        assert np.allclose(r.T @ r, np.eye(2 * M))

    def test_one_dimensional_with_pl(self):
        """Test for a one-dimensional example if the returned unitary behaves as expected
        when used within a PennyLane circuit, i.e., so that the probability of the final control
        wire encodes the function."""
        wires = 3
        M = 2 ** wires
        func = lambda i: np.sin(i) ** 2

        r = func_to_unitary(func, M)

        dev = qml.device("default.qubit", wires=(wires + 1))

        @qml.qnode(dev)
        def apply_r(input_state):
            qml.QubitStateVector(input_state, wires=range(wires))
            qml.QubitUnitary(r, wires=range(wires + 1))
            return qml.probs(wires)

        for i, state in enumerate(np.eye(M)):
            p = apply_r(state)[1]
            assert np.allclose(p, func(i))


def test_V():
    """Test for the _make_V function"""
    dim = 4

    V_expected = - np.eye(dim)
    V_expected[1, 1] = V_expected[3, 3] = 1
    V = _make_V(dim)

    assert np.allclose(V, V_expected)


def test_Z():
    """Test for the _make_Z function"""
    dim = 4

    Z_expected = - np.eye(dim)
    Z_expected[0, 0] = 1
    Z = _make_Z(dim)

    assert np.allclose(Z, Z_expected)


def test_Q():
    """Test for the make_Q function using a fixed example"""

    A = np.array([[ 0.85358423-0.32239299j, -0.12753659+0.38883306j],
       [ 0.39148136-0.11915985j,  0.34064316-0.84646648j]])
    R = np.array([[ 0.45885289+0.03972856j,  0.2798685 -0.05981098j,
         0.64514642-0.51555038j,  0.11015177-0.10877695j],
       [ 0.19407005-0.35483005j,  0.29756077+0.80153453j,
        -0.19147104+0.0507968j ,  0.15553799-0.20493631j],
       [ 0.35083011-0.20807392j, -0.27602911-0.13934692j,
         0.11874165+0.34532609j, -0.45945242-0.62734969j],
       [-0.11379919-0.66706921j, -0.21120956-0.2165113j ,
         0.30133006+0.23367271j,  0.54593491+0.08446372j]])

    Q_expected = np.array([[-0.46513201-1.38777878e-17j, -0.13035515-2.23341802e-01j,
        -0.74047856+7.08652160e-02j, -0.0990036 -3.91977176e-01j],
       [ 0.13035515-2.23341802e-01j,  0.46494302+0.00000000e+00j,
         0.05507901-1.19182067e-01j, -0.80370146-2.31904873e-01j],
       [-0.74047856-7.08652160e-02j, -0.05507901-1.19182067e-01j,
         0.62233412-2.77555756e-17j, -0.0310774 -2.02894077e-01j],
       [ 0.0990036 -3.91977176e-01j, -0.80370146+2.31904873e-01j,
         0.0310774 -2.02894077e-01j, -0.30774091+2.77555756e-17j]])

    Q = make_Q(A, R)

    assert np.allclose(Q, Q_expected)
