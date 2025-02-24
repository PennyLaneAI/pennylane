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
Unit tests for the QROMStatePreparation template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np



class TestQROMStatePreparation:

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        state = np.random.rand(2 ** 4)
        state /= np.linalg.norm(state)

        state = state / np.linalg.norm(state)
        wires = qml.registers({"work_wires": 3, "precision_wires": 3, "embedding_wires": 4})

        op = qml.QROMStatePreparation(state_vector=state,
                                      wires=wires["embedding_wires"],
                                      precision_wires=wires["precision_wires"],
                                      work_wires=wires["work_wires"])

        qml.ops.functions.assert_valid(op, skip_differentiation=True)


    @pytest.mark.parametrize(
        ("state", "msg_match"),
        [
            (
                np.array([1., 0, 0]),
                "State vectors must be of length",
            ),
            (
                    np.array([1., 1, 0, 0]),
                    "State vectors have to be of norm 1.0",
            ),
        ],
    )
    def test_QROMStatePrep_error(self, state, msg_match):
        """Test that proper errors are raised for MPSPrep"""
        with pytest.raises(ValueError, match=msg_match):
            qml.QROMStatePreparation(state, wires=[0, 1], precision_wires = [2,3], work_wires = [])

    '''
    @pytest.mark.jax
    def test_jax_jit_qrom_state_prep(self):
        """Check the operation works with jax and jit."""

        import jax
        from jax import numpy as jnp

        state = jnp.array([
            0.0 + 0.0j,
            -0.10705513j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -0.99451217 + 0.0j,
            0.0 + 0.0j,
        ])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.QROMStatePreparation(state, wires=range(4, 7), precision_wires = [ 2,3], work_wires=[0, 1])
            return qml.state()

        output = circuit()[:8]
        output_jit = jax.jit(circuit)()[:8]

        assert jax.numpy.allclose(output, jax.numpy.array(state), rtol=0.01)
        assert jax.numpy.allclose(output_jit, jax.numpy.array(state), rtol=0.01)
    '''