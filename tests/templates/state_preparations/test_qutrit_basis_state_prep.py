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
Unit tests for the QutritBasisStatePreparation template.
"""
import numpy as np

# pylint: disable=too-many-arguments
import pytest

import pennylane as qml


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    basis_state = [2, 1, 0, 2]
    wires = [1, 2, 6, 8]
    op = qml.QutritBasisStatePreparation(basis_state, wires)

    qml.ops.functions.assert_valid(op, skip_differentiation=True)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_wires", [
        ([0], [0], []),
        ([0], [1], []),
        ([1], [0], [0]),
        ([2], [1], [1, 1]),
        ([0, 1], [0, 1], [1]),
        ([2, 0], [1, 4], [1, 1]),
        ([1, 0], [4, 5], [4]),
        ([0, 2], [4, 5], [5, 5]),
        ([1, 2], [0, 2], [0, 2, 2]),
        ([0, 0, 1, 0], [1, 2, 3, 4], [3]),
        ([2, 0, 0, 0], [1, 2, 3, 4], [1, 1]),
        ([1, 1, 1, 0], [1, 2, 6, 8], [1, 2, 6]),
        ([0, 2, 1, 2], [1, 2, 6, 8], [2, 2, 6, 8, 8]),
        ([1, 0, 1, 1], [1, 2, 6, 8], [1, 6, 8]),
        ([2, 1, 0, 2], [1, 2, 6, 8], [1, 1, 2, 8, 8]),
    ])
    # fmt: on
    def test_correct_pl_gates(self, basis_state, wires, target_wires):
        """Tests queue for simple cases."""

        op = qml.QutritBasisStatePreparation(basis_state, wires)
        queue = op.decomposition()

        for id, gate in enumerate(queue):
            assert gate.name == "TShift"
            assert gate.wires.tolist() == [target_wires[id]]

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_state", [
        ([0], [0], [0, 0, 0]),
        ([0], [1], [0, 0, 0]),
        ([1], [0], [1, 0, 0]),
        ([1], [1], [0, 1, 0]),
        ([2], [0], [2, 0, 0]),
        ([2], [2], [0, 0, 2]),
        ([0, 1], [0, 1], [0, 1, 0]),
        ([0, 2], [0, 1], [0, 2, 0]),
        ([1, 1], [0, 2], [1, 0, 1]),
        ([1, 1], [1, 2], [0, 1, 1]),
        ([2, 2], [0, 2], [2, 0, 2]),
        ([2, 2], [1, 2], [0, 2, 2]),
        ([1, 0], [0, 2], [1, 0, 0]),
        ([2, 0], [0, 2], [2, 0, 0]),
        ([1, 1, 0], [0, 1, 2], [1, 1, 0]),
        ([1, 0, 1], [0, 1, 2], [1, 0, 1]),
        ([2, 1, 0], [0, 1, 2], [2, 1, 0]),
        ([1, 0, 2], [0, 1, 2], [1, 0, 2]),
    ])
    # fmt: on
    def test_state_preparation(self, tol, qutrit_device_3_wires, basis_state, wires, target_state):
        """Tests that the template produces the correct expectation values."""

        @qml.qnode(qutrit_device_3_wires)
        def circuit(obs):
            qml.QutritBasisStatePreparation(basis_state, wires)

            return [qml.expval(qml.THermitian(A=obs, wires=i)) for i in range(3)]

        # Convert to basis states
        obs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        output_state = [x - 1 for x in circuit(obs)]

        assert np.allclose(output_state, target_state, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_state_preparation_jax_jit(self):
        """Tests that the template can be JIT compiled."""
        import jax

        dev = qml.device("default.qutrit", wires=1)

        @qml.qnode(dev)
        def circuit(state):
            qml.QutritBasisStatePreparation(state, [0])
            return qml.state()

        circuit = jax.jit(circuit)

        basis_state = qml.math.array([2], like="jax")
        output_state = circuit(basis_state)

        assert qml.math.allclose(output_state, [0, 0, 1])

    @pytest.mark.jax
    def test_state_preparation_with_simpling_jax_jit(self):
        """Tests that the template can be compiled with JIT when returning
        a sampled measurement."""
        import jax

        n = 2

        @jax.jit
        @qml.qnode(qml.device("default.qutrit", wires=n), shots=1)
        def circuit(state):
            qml.QutritBasisStatePreparation(state, wires=range(n))
            return qml.sample(wires=range(n))

        state = jax.numpy.array([1, 1])
        circuit(state)

    @pytest.mark.jax
    @pytest.mark.parametrize("state", [0, 1, 2])
    def test_decomposition_matrix_jax_jit(self, state):
        """Tests that the decomposition matrix is correct when JIT compiled."""
        import jax
        import jax.numpy as jnp

        tshift = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        jit_decomp = jax.jit(qml.QutritBasisStatePreparation.compute_decomposition)

        decomp = jit_decomp(jnp.array([state]), wires=[0])
        matrix = qml.matrix(qml.prod(*decomp[::-1]))
        assert qml.math.allclose(matrix, jnp.linalg.matrix_power(tshift, state))

    @pytest.mark.jax
    @pytest.mark.parametrize("state", [0, 1, 2])
    def test_decomposition_pl_gates_jax_jit(self, state):
        """Tests that the decomposition gates are correct when JIT compiled."""
        import jax
        import jax.numpy as jnp

        jit_decomp = jax.jit(
            qml.QutritBasisStatePreparation.compute_decomposition, static_argnames="wires"
        )

        wire = (0,)
        state = jnp.array([state])
        decomp = jit_decomp(jnp.array([state]), wires=wire)

        op_list = [
            qml.TRY(state * (2 - state) * np.pi, wires=wire, subspace=(0, 1)),
            qml.TRY(state * (1 - state) * np.pi / 2, wires=wire, subspace=(0, 2)),
            qml.TRZ((-2 * state + 3) * state * np.pi, wires=wire, subspace=(0, 2)),
            qml.TRY(state * (2 - state) * np.pi, wires=wire, subspace=(0, 2)),
            qml.TRY(state * (1 - state) * np.pi / 2, wires=wire, subspace=(0, 1)),
            qml.TRZ(-(7 * state - 10) * state * np.pi, wires=wire, subspace=(0, 2)),
        ]

        for op1, op2 in zip(decomp, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "basis_state,wires,target_state",
        [
            ([0, 1], [0, 1], [0, 1, 0]),
            ([1, 1, 0], [0, 1, 2], [1, 1, 0]),
            ([1, 0, 1], [2, 0, 1], [0, 1, 1]),
        ],
    )
    @pytest.mark.xfail(reason="TensorFlow comptability not yet implemented")
    def test_state_preparation_tf_autograph(
        self, tol, qutrit_device_3_wires, basis_state, wires, target_state
    ):
        """Tests that the template produces the correct expectation values."""
        import tensorflow as tf

        @tf.function
        @qml.qnode(qutrit_device_3_wires, interface="tf")
        def circuit(state, obs):
            qml.QutritBasisStatePreparation(state, wires)

            return [qml.expval(qml.THermitian(A=obs, wires=i)) for i in range(3)]

        obs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        output_state = [x - 1 for x in circuit(basis_state, obs)]

        assert np.allclose(output_state, target_state, atol=tol, rtol=0)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        basis_state = [0, 1, 2]

        dev = qml.device("default.qutrit", wires=3)
        dev2 = qml.device("default.qutrit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit(obs):
            qml.QutritBasisStatePreparation(basis_state, wires=range(3))
            return qml.expval(qml.THermitian(A=obs, wires=0))

        @qml.qnode(dev2)
        def circuit2(obs):
            qml.QutritBasisStatePreparation(basis_state, wires=["z", "a", "k"])
            return qml.expval(qml.THermitian(A=obs, wires="z"))

        obs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        circuit(obs)
        circuit2(obs)

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires", [
        ([0], [0, 1]),
        ([2], [0, 1]),
        ([0, 1], [0]),
        ([2, 0], [0]),
    ])
    # fmt: on
    def test_error_num_qutrits(self, basis_state, wires):
        """Tests that the correct error message is raised when the number
        of qutrits does not match the number of wires."""

        with pytest.raises(ValueError, match="Basis states must be of (shape|length)"):
            qml.QutritBasisStatePreparation(basis_state, wires)

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires", [
        ([3], [0]),
        ([2, 0, 4], [0, 1, 2]),
    ])
    # fmt: on
    def test_error_basis_state_format(self, basis_state, wires):
        """Tests that the correct error messages is raised when
        the basis state contains numbers different from 0 ,1 and 2."""

        with pytest.raises(ValueError, match="Basis states must only (contain|consist)"):
            qml.QutritBasisStatePreparation(basis_state, wires)

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""
        dev = qml.device("default.qutrit", wires=2)

        @qml.qnode(dev)
        def circuit(basis_state, obs):
            qml.QutritBasisStatePreparation(basis_state, wires=range(2))
            return qml.expval(qml.THermitian(A=obs, wires=0))

        obs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        with pytest.raises(ValueError, match="Basis states must be one-dimensional"):
            basis_state = np.array([[[0, 2]]])
            circuit(basis_state, obs)

        with pytest.raises(ValueError, match="Basis states must be of length"):
            basis_state = np.array([0, 1, 2])
            circuit(basis_state, obs)

        with pytest.raises(ValueError, match="Basis states must only consist of"):
            basis_state = np.array([0, 3])
            circuit(basis_state, obs)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.QutritBasisStatePreparation(np.array([0, 2]), wires=[0, 1], id="a")
        assert template.id == "a"
