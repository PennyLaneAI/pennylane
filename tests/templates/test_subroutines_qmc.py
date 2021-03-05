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
from pennylane.templates.subroutines.qmc import probs_to_unitary

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
