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
"""
Tests for the Fable template.
"""

import numpy as np
import pytest
import pennylane as qml
from pennylane.templates.subroutines.fable import FABLE

input_matrix = [
    [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
    [0.97041007, 0.97041007, 0.99999329, 0.99999329],
    [0.82429855, 0.82429855, 0.98175843, 0.98175843],
    [0.99675093, 0.99675093, 0.83514837, 0.83514837],
]

ancilla = ["ancilla"]
s = int(np.log2(np.array(input_matrix).shape[0]))
wires_i = [f"i{index}" for index in range(s)]
wires_j = [f"j{index}" for index in range(s)]
wire_order = ancilla + wires_i[::-1] + wires_j[::-1]


def test_fable_real():
    np_matrix = np.array(input_matrix)
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        FABLE(np_matrix, tol=0.01)
        return qml.state()

    expected = (
        len(np_matrix)
        * qml.matrix(circuit, wire_order=wire_order)().real[0 : len(np_matrix), 0 : len(np_matrix)]
    )
    assert np.allclose(np_matrix, expected)


@pytest.mark.jax
def test_fable_jax():
    """Test that the Fable operator matrix is correct for jax."""
    import jax.numpy as jnp

    jax_matrix = jnp.array(input_matrix)
    op = FABLE(jax_matrix, 0)

    M = jnp.array(
        len(jax_matrix)
        * qml.matrix(op, wire_order=wire_order).real[0 : len(jax_matrix), 0 : len(jax_matrix)]
    )
    assert np.allclose(M, jax_matrix)
    assert qml.math.get_interface(M) == "jax"
