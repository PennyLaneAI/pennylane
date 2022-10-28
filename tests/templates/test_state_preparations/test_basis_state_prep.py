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
Unit tests for the BasisStatePreparation template.
"""
import pytest
import numpy as np
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_wires", [
        ([0], [0], []),
        ([0], [1], []),
        ([1], [0], [0]),
        ([1], [1], [1]),
        ([0, 1], [0, 1], [1]),
        ([1, 0], [1, 4], [1]),
        ([1, 1], [0, 2], [0, 2]),
        ([1, 0], [4, 5], [4]),
        ([0, 0, 1, 0], [1, 2, 3, 4], [3]),
        ([1, 1, 1, 0], [1, 2, 6, 8], [1, 2, 6]),
        ([1, 0, 1, 1], [1, 2, 6, 8], [1, 6, 8]),
    ])
    # fmt: on
    def test_correct_pl_gates(self, basis_state, wires, target_wires):
        """Tests queue for simple cases."""

        op = qml.BasisStatePreparation(basis_state, wires)
        queue = op.expand().operations

        for id, gate in enumerate(queue):
            assert gate.name == "PauliX"
            assert gate.wires.tolist() == [target_wires[id]]

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_state", [
        ([0], [0], [0, 0, 0]),
        ([0], [1], [0, 0, 0]),
        ([1], [0], [1, 0, 0]),
        ([1], [1], [0, 1, 0]),
        ([0, 1], [0, 1], [0, 1, 0]),
        ([1, 1], [0, 2], [1, 0, 1]),
        ([1, 1], [1, 2], [0, 1, 1]),
        ([1, 0], [0, 2], [1, 0, 0]),
        ([1, 1, 0], [0, 1, 2], [1, 1, 0]),
        ([1, 0, 1], [0, 1, 2], [1, 0, 1]),
    ])
    # fmt: on
    def test_state_preparation(self, tol, qubit_device_3_wires, basis_state, wires, target_state):
        """Tests that the template produces the correct expectation values."""

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            qml.BasisStatePreparation(basis_state, wires)

            # Pauli Z gates identify the basis state
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        # Convert from Pauli Z eigenvalues to basis state
        output_state = [0 if x == 1.0 else 1 for x in circuit()]

        assert np.allclose(output_state, target_state, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "basis_state,wires,target_state",
        [
            ([0, 1], [0, 1], [0, 1, 0]),
            ([1, 1, 0], [0, 1, 2], [1, 1, 0]),
        ],
    )
    def test_state_preparation_jax_jit(
        self, tol, qubit_device_3_wires, basis_state, wires, target_state
    ):
        """Tests that the template produces the correct expectation values."""
        import jax

        @qml.qnode(qubit_device_3_wires, interface="jax")
        def circuit(state):
            qml.BasisStatePreparation(state, wires)

            # Pauli Z gates identify the basis state
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        circuit = jax.jit(circuit)

        # Convert from Pauli Z eigenvalues to basis state
        output_state = [0 if x == 1.0 else 1 for x in circuit(basis_state)]

        assert np.allclose(output_state, target_state, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        basis_state = [0, 1, 0]

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.BasisStatePreparation(basis_state, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.BasisStatePreparation(basis_state, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires", [
        ([0], [0, 1]),
        ([0, 1], [0]),
    ])
    # fmt: on
    def test_error_num_qubits(self, basis_state, wires):
        """Tests that the correct error message is raised when the number
        of qubits does not match the number of wires."""

        with pytest.raises(ValueError, match="Basis states must be of (shape|length)"):
            qml.BasisStatePreparation(basis_state, wires)

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires", [
        ([3], [0]),
        ([1, 0, 2], [0, 1, 2]),
    ])
    # fmt: on
    def test_error_basis_state_format(self, basis_state, wires):
        """Tests that the correct error messages is raised when
        the basis state contains numbers different from 0 and 1."""

        with pytest.raises(ValueError, match="Basis states must only (contain|consist)"):
            qml.BasisStatePreparation(basis_state, wires)

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.BasisStatePreparation(basis_state, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Basis states must be one-dimensional"):
            basis_state = np.array([[[0, 1]]])
            circuit(basis_state)

        with pytest.raises(ValueError, match="Basis states must be of length"):
            basis_state = np.array([0, 1, 0])
            circuit(basis_state)

        with pytest.raises(ValueError, match="Basis states must only consist of"):
            basis_state = np.array([0, 2])
            circuit(basis_state)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.BasisStatePreparation(np.array([0, 1]), wires=[0, 1], id="a")
        assert template.id == "a"
