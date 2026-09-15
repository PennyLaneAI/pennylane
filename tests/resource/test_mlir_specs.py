# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the specs transform"""

# pylint: disable=invalid-sequence-index

import pytest

import pennylane as qp
from pennylane.resource.mlir_specs import resources_from_analysis_pass

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


@pytest.mark.parametrize(
    "level",
    [
        1,
        2,
        [1, 2],
    ],
)
def test_resources_from_analysis_pass_invalid_levels(level):
    """Test that resources_from_analysis_pass raises an error when invalid levels are provided."""

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        qp.RX(0.5, 0)
        qp.Hadamard(wires=0)
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(ValueError, match="Requested specs levels .* not found"):
        resources_from_analysis_pass(
            circuit,
            original_qnode=circuit.original_function,
            level=level,
            level_to_markers={},
            level_to_name={},
        )
