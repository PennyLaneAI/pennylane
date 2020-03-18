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
Unit tests for the :mod:`pennylane.collection.covmat` submodule.
"""
import pytest

import pennylane as qml
from pennylane.collections.covmat import symmetric_product, CovarianceMatrix

class TestSymmetricProduct:
    """Test the symmetric product of observables."""

    @pytest.mark.parametrize("obs1,obs2,expected_product", [
        (qml.PauliX(0), qml.PauliX(0), qml.Identity(wires=[0]))
    ])
    def test_symmetric_product(self, obs1, obs2, expected_product):
        """Test that the symmetric product yields the expected observable."""

        result = symmetric_product(obs1, obs2)

        assert result.name == expected_product.name
        assert result.wires == expected_product.wires
        for result_param, expected_param in zip(result.params, expected_product.params):
            assert result_param == expected_param