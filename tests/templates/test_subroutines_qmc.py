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
from pennylane.templates.subroutines.qmc import probs_to_unitary, func_to_unitary

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
