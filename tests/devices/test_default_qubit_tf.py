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
Unit tests and integration tests for the ``default.qubit.tf`` device.
"""
# pylint: disable=too-many-arguments,protected-access,too-many-public-methods
import numpy as np
import pytest
from gate_data import (
    CCZ,
    CNOT,
    CSWAP,
    CZ,
    SWAP,
    ControlledPhaseShift,
    CRot3,
    CRotx,
    CRoty,
    CRotz,
    FermionicSWAP,
    H,
    I,
    MultiRZ1,
    MultiRZ2,
    OrbitalRotation,
    Rot3,
    Rotx,
    Roty,
    Rotz,
    Rphi,
    S,
    T,
    Toffoli,
    X,
    Y,
    Z,
)

import pennylane as qml
from pennylane import DeviceError
from pennylane import numpy as pnp

tf = pytest.importorskip("tensorflow", minversion="2.0")
from pennylane.devices.default_qubit_tf import (  # pylint: disable=wrong-import-position
    DefaultQubitTF,
)

np.random.seed(42)


#####################################################
# Test matrices
#####################################################

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


#####################################################
# Define standard qubit operations
#####################################################

single_qubit = [
    (qml.S, S),
    (qml.T, T),
    (qml.PauliX, X),
    (qml.PauliY, Y),
    (qml.PauliZ, Z),
    (qml.Hadamard, H),
]
single_qubit_param = [
    (qml.PhaseShift, Rphi),
    (qml.RX, Rotx),
    (qml.RY, Roty),
    (qml.RZ, Rotz),
    (qml.MultiRZ, MultiRZ1),
]
two_qubit = [(qml.CZ, CZ), (qml.CNOT, CNOT), (qml.SWAP, SWAP)]
two_qubit_param = [
    (qml.CRX, CRotx),
    (qml.CRY, CRoty),
    (qml.CRZ, CRotz),
    (qml.MultiRZ, MultiRZ2),
    (qml.ControlledPhaseShift, ControlledPhaseShift),
    (qml.FermionicSWAP, FermionicSWAP),
]
three_qubit = [(qml.Toffoli, Toffoli), (qml.CSWAP, CSWAP), (qml.CCZ, CCZ)]
four_qubit_param = [(qml.OrbitalRotation, OrbitalRotation)]

#####################################################
# Fixtures
#####################################################


# pylint: disable=unused-argument
@pytest.fixture(name="init_state")
def init_state_fixture(scope="session"):
    """Generates a random initial state"""

    def _init_state(n):
        """random initial state"""
        np.random.seed(4214152)
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


# pylint: disable=unused-argument
@pytest.fixture(name="broadcasted_init_state")
def broadcasted_init_state_fixture(scope="session"):
    """Generates a random initial state"""

    def _broadcasted_init_state(n, batch_size):
        """random initial state"""
        np.random.seed(4214152)
        state = np.random.random([batch_size, 2**n]) + np.random.random([batch_size, 2**n]) * 1j
        return state / np.linalg.norm(state, axis=1)[:, np.newaxis]

    return _broadcasted_init_state


#####################################################
# Initialization test
#####################################################


@pytest.mark.tf
def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.qubit.tf", wires=1, shots=1, analytic=True)


#####################################################
# Device-level matrix creation tests
#####################################################


@pytest.mark.tf
class TestTFMatrix:
    """Test special case of matrix construction in TensorFlow for
    cases where variables must be casted to complex."""

    @pytest.mark.parametrize(
        "op,params,wires",
        [
            (qml.PhaseShift, [0.1], 0),
            (qml.ControlledPhaseShift, [0.1], [0, 1]),
            (qml.CRX, [0.1], [0, 1]),
            (qml.CRY, [0.1], [0, 1]),
            (qml.CRZ, [0.1], [0, 1]),
            (qml.CRot, [0.1, 0.2, 0.3], [0, 1]),
            (qml.U1, [0.1], 0),
            (qml.U2, [0.1, 0.2], 0),
            (qml.U3, [0.1, 0.2, 0.3], 0),
            (qml.Rot, [0.1, 0.2, 0.3], 0),
        ],
    )
    def test_tf_matrix(self, op, params, wires):
        tf_params = [tf.Variable(x) for x in params]
        expected_mat = op(*params, wires=wires).matrix()
        obtained_mat = op(*tf_params, wires=wires).matrix()
        assert qml.math.get_interface(obtained_mat) == "tensorflow"
        assert qml.math.allclose(qml.math.unwrap(obtained_mat), expected_mat)

    @pytest.mark.parametrize(
        "op,params,wires",
        [
            (qml.PhaseShift, ([0.1, 0.2, 0.5],), 0),
            (qml.ControlledPhaseShift, ([0.1],), [0, 1]),
            (qml.CRX, ([0.1, -0.6, 0.2],), [0, 1]),
            (qml.CRY, ([0.1, -0.4, 6.3],), [0, 1]),
            (qml.CRZ, ([0.1, -0.6, 0.2],), [0, 1]),
            (qml.CRot, ([0.1, 0.2, 0.3], 0.6, [0.2, 1.2, 4.3]), [0, 1]),
            (qml.U1, ([0.1, 0.2, 0.5],), 0),
            (qml.U2, ([0.1, 0.2, 0.5], [0.6, 9.3, 2.1]), 0),
            (qml.U3, ([0.1, 0.2, 0.3], 0.6, [0.2, 1.2, 4.3]), 0),
            (qml.Rot, ([0.1, 0.2, 0.3], 0.6, [0.2, 1.2, 4.3]), 0),
        ],
    )
    def test_broadcasted_tf_matrix(self, op, params, wires):
        params = [np.array(p) for p in params]
        tf_params = [tf.Variable(x) for x in params]
        expected_mat = op(*params, wires=wires).matrix()
        obtained_mat = op(*tf_params, wires=wires).matrix()
        assert qml.math.get_interface(obtained_mat) == "tensorflow"
        assert qml.math.allclose(qml.math.unwrap(obtained_mat), expected_mat)

    @pytest.mark.parametrize(
        "param,pauli,wires",
        [
            (0.1, "I", "a"),
            (0.2, "IX", ["a", "b"]),
            (-0.3, "III", [0, 1, 2]),
            (0.5, "ZXI", [0, 1, 2]),
            # Broadcasted rotations
            ([0.1, 0.6], "I", "a"),
            ([0.2], "IX", ["a", "b"]),
            ([-0.3, 0.0, 0.2], "III", [0, 1, 2]),
            ([0.5, 0.2], "ZXI", [0, 1, 2]),
        ],
    )
    def test_pauli_rot_tf_(self, param, pauli, wires):
        param = np.array(param)
        op = qml.PauliRot(param, pauli, wires=wires)
        expected_mat = op.matrix()
        expected_eigvals = op.eigvals()

        tf_op = qml.PauliRot(tf.Variable(param), pauli, wires=wires)
        obtained_mat = tf_op.matrix()
        obtained_eigvals = tf_op.eigvals()

        assert qml.math.get_interface(obtained_mat) == "tensorflow"
        assert qml.math.get_interface(obtained_eigvals) == "tensorflow"

        assert qml.math.allclose(qml.math.unwrap(obtained_mat), expected_mat)
        assert qml.math.allclose(qml.math.unwrap(obtained_eigvals), expected_eigvals)

    @pytest.mark.parametrize(
        "op,param,wires",
        [
            (qml.PhaseShift, 0.1, [1]),
            (qml.ControlledPhaseShift, 0.1, [1, 2]),
            (qml.CRZ, 0.1, [1, 2]),
            (qml.U1, 0.1, [1]),
            # broadcasted operation matrices
            (qml.PhaseShift, np.array([0.1, 0.6]), [1]),
            (qml.ControlledPhaseShift, np.array([0.1]), [1, 2]),
            (qml.CRZ, np.array([0.1, 0.7, 8.3]), [1, 2]),
            (qml.U1, np.array([0.1, 0.7, 8.3]), [1]),
        ],
    )
    def test_expand_tf_matrix(self, op, param, wires):
        reg_mat = op(param, wires=wires).matrix()

        if len(wires) == 1:
            expected_mat = qml.math.kron(I, qml.math.kron(reg_mat, qml.math.kron(I, I)))
        else:
            expected_mat = qml.math.kron(I, qml.math.kron(reg_mat, I))

        tf_mat = op(tf.Variable(param), wires=wires).matrix()
        obtained_mat = qml.math.expand_matrix(tf_mat, wires, list(range(4)))

        assert qml.math.get_interface(obtained_mat) == "tensorflow"
        assert qml.math.allclose(qml.math.unwrap(obtained_mat), expected_mat)


#####################################################
# Device-level integration tests
#####################################################


@pytest.mark.tf
class TestApply:
    """Test application of PennyLane operations."""

    def test_basis_state(self, tol):
        """Test basis state initialization"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev.state
        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_invalid_basis_state_length(self):
        """Test that an exception is raised if the basis state is the wrong size"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        with pytest.raises(
            ValueError, match=r"BasisState parameter and wires must be of equal length"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2])])

    def test_invalid_basis_state(self):
        """Test that an exception is raised if the basis state is invalid"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 2])

        with pytest.raises(
            ValueError, match=r"BasisState parameter must consist of 0 or 1 integers"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

    def test_state_prep(self, init_state, tol):
        """Test state prep application"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        dev.apply([qml.StatePrep(state, wires=[0])])

        res = dev.state
        expected = state
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_full_subsystem_statevector(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubitTF(wires=["a", "b", "c"])
        state = tf.constant([1, 0, 0, 0, 1, 0, 1, 1], dtype=tf.complex128) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert np.all(tf.reshape(dev._state, [-1]) == state)
        spy.assert_not_called()

    def test_partial_subsystem_statevector(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""
        dev = DefaultQubitTF(wires=["a", "b", "c"])
        state = tf.constant([1, 0, 1, 0], dtype=tf.complex128) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = tf.reshape(tf.reduce_sum(dev._state, axis=(1,)), [-1])

        assert np.all(res == state)
        spy.assert_called()

    def test_invalid_state_prep_size(self):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 1])

        with pytest.raises(ValueError, match=r"State vector must have shape \(2\*\*wires,\)"):
            dev.apply([qml.StatePrep(state, wires=[0, 1])])

    def test_invalid_state_prep_norm(self):
        """Test that an exception is raised if the state
        vector is not normalized"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 12])

        with pytest.raises(ValueError, match=r"Sum of amplitudes-squared does not equal one"):
            dev.apply([qml.StatePrep(state, wires=[0])])

    def test_invalid_state_prep(self):
        """Test that an exception is raised if a state preparation is not the
        first operation in the circuit."""
        dev = DefaultQubitTF(wires=2)
        state = np.array([0, 1])

        with pytest.raises(
            qml.DeviceError,
            match=r"cannot be used after other Operations have already been applied",
        ):
            dev.apply([qml.PauliZ(0), qml.StatePrep(state, wires=[0])])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized single qubit operations"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test parametrized single qubit operations"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation(self, init_state, tol):
        """Test three axis rotation gate"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = Rot3(a, b, c) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation(self, init_state, tol):
        """Test three axis controlled-rotation gate"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = CRot3(a, b, c) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized two qubit operations"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [op(wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, mat, tol):
        """Test application of arbitrary qubit unitaries"""
        N = int(np.log2(len(mat)))
        dev = DefaultQubitTF(wires=N)
        state = init_state(N)

        queue = [qml.StatePrep(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test non-parametrized three qubit operations"""
        dev = DefaultQubitTF(wires=3)
        state = init_state(3)

        queue = [qml.StatePrep(state, wires=[0, 1, 2])]
        queue += [op(wires=[0, 1, 2])]
        dev.apply(queue)

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test two qubit parametrized operations"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [op(theta, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", four_qubit_param)
    def test_four_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test four qubit parametrized operations"""
        dev = DefaultQubitTF(wires=4)
        state = init_state(4)

        queue = [qml.StatePrep(state, wires=[0, 1, 2, 3])]
        queue += [op(theta, wires=[0, 1, 2, 3])]
        dev.apply(queue)

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, atol=tol, rtol=0)

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_apply_ops_not_supported(self, mocker, monkeypatch):
        """Test that when a version of TensorFlow before 2.3.0 is used, the _apply_ops dictionary is
        empty and application of a CNOT gate is performed using _apply_unitary_einsum"""
        with monkeypatch.context() as m:
            m.setattr("pennylane.devices.default_qubit_tf.SUPPORTS_APPLY_OPS", False)
            dev = DefaultQubitTF(wires=3)
            assert dev._apply_ops == {}

            spy = mocker.spy(DefaultQubitTF, "_apply_unitary_einsum")

            queue = [qml.CNOT(wires=[1, 2])]
            dev.apply(queue)

            spy.assert_called_once()

    def test_apply_ops_above_8_wires(self, mocker):
        """Test that when 9 wires are used, the _apply_ops dictionary is empty and application of a
        CNOT gate is performed using _apply_unitary_einsum"""
        dev = DefaultQubitTF(wires=9)
        assert dev._apply_ops == {}

        spy = mocker.spy(DefaultQubitTF, "_apply_unitary_einsum")

        queue = [qml.CNOT(wires=[1, 2])]
        dev.apply(queue)

        spy.assert_called_once()

    @pytest.mark.xfail(
        raises=tf.errors.UnimplementedError,
        reason="Slicing is not supported for more than 8 wires",
        strict=True,
    )
    def test_apply_ops_above_8_wires_using_special(self):
        """Test that special apply methods that involve slicing function correctly when using 9
        wires"""
        dev = DefaultQubitTF(wires=9)
        dev._apply_ops = {"CNOT": dev._apply_cnot}

        queue = [qml.CNOT(wires=[1, 2])]
        dev.apply(queue)

    def test_do_not_split_analytic_tf(self, mocker):
        """Tests that the Hamiltonian is not split for shots=None using the tf device."""
        dev = qml.device("default.qubit.tf", wires=2)
        ham = qml.Hamiltonian(tf.Variable([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit():
            return qml.expval(ham)

        spy = mocker.spy(dev, "expval")

        circuit()
        # evaluated one expval altogether
        assert spy.call_count == 1


@pytest.mark.tf
class TestApplyBroadcasted:
    """Test application of broadcasted PennyLane operations."""

    @pytest.mark.skip("Applying a BasisState does not support broadcasting yet")
    def test_basis_state_broadcasted(self, tol):
        """Test basis state initialization"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev.state
        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.skip("Applying a BasisState does not support broadcasting yet")
    def test_invalid_basis_state_length_broadcasted(self):
        """Test that an exception is raised if the basis state is the wrong size"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 0])

        with pytest.raises(
            ValueError, match=r"BasisState parameter and wires must be of equal length"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2])])

    @pytest.mark.skip("Applying a BasisState does not support broadcasting yet")
    def test_invalid_basis_state_broadcasted(self):
        """Test that an exception is raised if the basis state is invalid"""
        dev = DefaultQubitTF(wires=4)
        state = np.array([0, 0, 1, 2])

        with pytest.raises(
            ValueError, match=r"BasisState parameter must consist of 0 or 1 integers"
        ):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_qubit_state_vector_broadcasted(self, broadcasted_init_state, tol, batch_size):
        """Test broadcasted qubit state vector application"""
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, batch_size=batch_size)

        dev.apply([qml.StatePrep(state, wires=[0])])

        res = dev.state
        expected = state
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_full_subsystem_statevector_broadcasted(self, mocker):
        """Test applying a broadcasted state vector to the full subsystem"""
        dev = DefaultQubitTF(wires=["a", "b", "c"])
        state = (
            tf.constant(
                [[1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 1, 0, 1]],
                dtype=tf.complex128,
            )
            / 2
        )
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert np.all(tf.reshape(dev._state, [3, 8]) == state)
        spy.assert_not_called()

    def test_error_partial_subsystem_statevector_broadcasted(self):
        """Test applying a broadcasted state vector to a subset of wires of the full subsystem"""
        dev = DefaultQubitTF(wires=["a", "b", "c"])
        state = tf.constant(
            [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=tf.complex128
        ) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        with pytest.raises(NotImplementedError, match="Parameter broadcasting is not supported"):
            dev._apply_state_vector(state=state, device_wires=state_wires)

    def test_invalid_qubit_state_vector_size_broadcasted(self):
        """Test that an exception is raised if the broadcasted state
        vector is the wrong size"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])

        with pytest.raises(ValueError, match=r"State vector must have shape \(2\*\*wires,\)"):
            dev.apply([qml.StatePrep(state, wires=[0, 1])])

    def test_invalid_qubit_state_vector_norm_broadcasted(self):
        """Test that an exception is raised if the broadcasted state
        vector is not normalized"""
        dev = DefaultQubitTF(wires=2)
        state = np.array([[1, 0], [0, 12], [1.3, 1]])

        with pytest.raises(ValueError, match=r"Sum of amplitudes-squared does not equal one"):
            dev.apply([qml.StatePrep(state, wires=[0])])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters_broadcasted(self, broadcasted_init_state, op, mat, tol):
        """Test non-parametrized single qubit operations"""
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, 3)

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,kj->ki", mat, state)
        assert isinstance(res, tf.Tensor)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters_broadcasted_state(
        self, broadcasted_init_state, op, func, theta, tol
    ):
        """Test parametrized single qubit operations with broadcasted initial state"""
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, 3)

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,kj->ki", func(theta), state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [[np.pi / 3], [0.5432, -0.232, 0.1]])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters_broadcasted_par(self, init_state, op, func, theta, tol):
        """Test parametrized single qubit operations with broadcasted parameters"""
        theta = np.array(theta)
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        mat = np.array([func(t) for t in theta])
        expected = np.einsum("lij,j->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [[np.pi / 3], [0.5432, -0.232, 0.1]])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters_broadcasted_both(
        self, broadcasted_init_state, op, func, theta, tol
    ):
        """Test parametrized single qubit operations with broadcasted init state and parameters"""
        theta = np.array(theta)
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, batch_size=len(theta))

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [op(theta, wires=0)]
        dev.apply(queue)

        res = dev.state
        mat = np.array([func(t) for t in theta])
        expected = np.einsum("lij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation_broadcasted_state(self, broadcasted_init_state, tol):
        """Test three axis rotation gate with broadcasted state"""
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, 3)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,lj->li", Rot3(a, b, c), state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation_broadcasted_par(self, init_state, tol):
        """Test three axis rotation gate with broadcasted parameters"""
        dev = DefaultQubitTF(wires=1)
        state = init_state(1)

        a = np.array([0.542, 0.96, 0.213])
        b = -0.654
        c = np.array([1.3432, 0.6324, 6.32])

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        mat = np.array([Rot3(_a, b, _c) for _a, _c in zip(a, c)])
        expected = np.einsum("lij,j->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation_broadcasted_both(self, broadcasted_init_state, tol):
        """Test three axis rotation gate with broadcasted state and parameters"""
        dev = DefaultQubitTF(wires=1)
        state = broadcasted_init_state(1, 3)

        a = np.array([0.542, 0.96, 0.213])
        b = np.array([1.3432, 0.6324, 6.32])
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0])]
        queue += [qml.Rot(a, b, c, wires=0)]
        dev.apply(queue)

        res = dev.state
        mat = np.array([Rot3(_a, _b, c) for _a, _b in zip(a, b)])
        expected = np.einsum("lij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation_broadcasted_state(self, broadcasted_init_state, tol):
        """Test controlled three axis rotation gate with broadcasted state"""
        dev = DefaultQubitTF(wires=2)
        state = broadcasted_init_state(2, 3)

        a = 0.542
        b = 1.3432
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,lj->li", CRot3(a, b, c), state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation_broadcasted_par(self, init_state, tol):
        """Test controlled three axis rotation gate with broadcasted parameters"""
        dev = DefaultQubitTF(wires=2)
        state = init_state(2)

        a = np.array([0.542, 0.96, 0.213])
        b = -0.654
        c = np.array([1.3432, 0.6324, 6.32])

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        mat = np.array([CRot3(_a, b, _c) for _a, _c in zip(a, c)])
        expected = np.einsum("lij,j->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_controlled_rotation_broadcasted_both(self, broadcasted_init_state, tol):
        """Test controlled three axis rotation gate with broadcasted state and parameters"""
        dev = DefaultQubitTF(wires=2)
        state = broadcasted_init_state(2, 3)

        a = np.array([0.542, 0.96, 0.213])
        b = np.array([1.3432, 0.6324, 6.32])
        c = -0.654

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [qml.CRot(a, b, c, wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        mat = np.array([CRot3(_a, _b, c) for _a, _b in zip(a, b)])
        expected = np.einsum("lij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op,mat", two_qubit)
    def test_two_qubit_no_parameters_broadcasted(self, broadcasted_init_state, op, mat, tol):
        """Test non-parametrized two qubit operations"""
        dev = DefaultQubitTF(wires=2)
        state = broadcasted_init_state(2, 3)

        queue = [qml.StatePrep(state, wires=[0, 1])]
        queue += [op(wires=[0, 1])]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary_broadcasted_state(self, broadcasted_init_state, mat, tol):
        """Test application of arbitrary qubit unitaries for broadcasted state"""
        N = int(np.log2(len(mat)))
        dev = DefaultQubitTF(wires=N)
        state = broadcasted_init_state(N, 3)

        queue = [qml.StatePrep(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary_broadcasted_par(self, init_state, mat, tol):
        """Test application of broadcasted arbitrary qubit unitaries"""
        mat = np.array([mat, mat, mat])
        N = int(np.log2(mat.shape[-1]))
        dev = DefaultQubitTF(wires=N)
        state = init_state(N)

        queue = [qml.StatePrep(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("lij,j->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary_broadcasted_both(self, broadcasted_init_state, mat, tol):
        """Test application of arbitrary qubit unitaries for broadcasted state and parameters"""
        mat = np.array([mat, mat, mat])
        N = int(np.log2(mat.shape[-1]))
        dev = DefaultQubitTF(wires=N)
        state = broadcasted_init_state(N, 3)

        queue = [qml.StatePrep(state, wires=range(N))]
        queue += [qml.QubitUnitary(mat, wires=range(N))]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("lij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters_broadcasted(self, broadcasted_init_state, op, mat, tol):
        """Test broadcasted non-parametrized three qubit operations"""
        dev = DefaultQubitTF(wires=3)
        state = broadcasted_init_state(3, 2)

        queue = [qml.StatePrep(state, wires=[0, 1, 2])]
        queue += [op(wires=[0, 1, 2])]
        dev.apply(queue)

        res = dev.state
        expected = np.einsum("ij,lj->li", mat, state)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.usefixtures("use_new_opmath")
    def test_direct_eval_hamiltonian_broadcasted_tf(self):
        """Tests that the correct result is returned when attempting to evaluate a Hamiltonian with
        broadcasting and shots=None directly via its sparse representation with TF."""
        dev = qml.device("default.qubit.tf", wires=2)
        ham = qml.ops.LinearCombination(tf.Variable([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit():
            qml.RX(np.zeros(5), 0)  # Broadcast the state by applying a broadcasted identity
            return qml.expval(ham)

        res = circuit()
        assert qml.math.allclose(res, 0.2)

    @pytest.mark.usefixtures("use_legacy_opmath")
    def test_direct_eval_hamiltonian_broadcasted_error_tf_legacy_opmath(self):
        """Tests that an error is raised when attempting to evaluate a Hamiltonian with
        broadcasting and shots=None directly via its sparse representation with TF."""
        dev = qml.device("default.qubit.tf", wires=2)
        ham = qml.Hamiltonian(tf.Variable([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit():
            qml.RX(np.zeros(5), 0)  # Broadcast the state by applying a broadcasted identity
            return qml.expval(ham)

        with pytest.raises(NotImplementedError, match="Hamiltonians for interface!=None"):
            circuit()


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)

scalar_angles = list(zip(THETA, PHI, VARPHI))
broadcasted_angles = [(THETA, PHI, VARPHI), (THETA[0], PHI, VARPHI)]
all_angles = scalar_angles + broadcasted_angles


# pylint: disable=unused-argument
@pytest.mark.tf
@pytest.mark.parametrize("theta, phi, varphi", all_angles)
class TestExpval:
    """Test expectation values"""

    # test data; each tuple is of the form (GATE, OBSERVABLE, EXPECTED)
    single_wire_expval_test_data = [
        (
            qml.RX,
            qml.Identity,
            lambda t, p: np.array(
                [np.ones_like(t) * np.ones_like(p), np.ones_like(t) * np.ones_like(p)]
            ),
        ),
        (
            qml.RX,
            qml.PauliZ,
            lambda t, p: np.array([np.cos(t) * np.ones_like(p), np.cos(t) * np.cos(p)]),
        ),
        (
            qml.RY,
            qml.PauliX,
            lambda t, p: np.array([np.sin(t) * np.sin(p), np.sin(p) * np.ones_like(t)]),
        ),
        (
            qml.RX,
            qml.PauliY,
            lambda t, p: np.array([np.zeros_like(t) * np.zeros_like(p), -np.cos(t) * np.sin(p)]),
        ),
        (
            qml.RY,
            qml.Hadamard,
            lambda t, p: np.array(
                [np.sin(t) * np.sin(p) + np.cos(t), np.cos(t) * np.cos(p) + np.sin(p)]
            )
            / np.sqrt(2),
        ),
    ]

    @pytest.mark.parametrize("gate,obs,expected", single_wire_expval_test_data)
    def test_single_wire_expectation(self, gate, obs, expected, theta, phi, varphi, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = DefaultQubitTF(wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            _ = [gate(theta, wires=0), gate(phi, wires=1), qml.CNOT(wires=[0, 1])]
            _ = [qml.expval(obs(wires=[i])) for i in range(2)]

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        assert np.allclose(res, expected(theta, phi), atol=tol, rtol=0)

    def test_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        dev = DefaultQubitTF(wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            _ = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
            _ = [qml.expval(qml.Hermitian(A, wires=[i])) for i in range(2)]

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multi_mode_hermitian_expectation(self, theta, phi, varphi, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        _A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev = DefaultQubitTF(wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            _ = [qml.RY(theta, wires=0), qml.RY(phi, wires=1), qml.CNOT(wires=[0, 1])]
            _ = [qml.expval(qml.Hermitian(_A, wires=[0, 1]))]

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        dev.reset()

        _A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(_A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        A1 = np.array([[1, 2], [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2
            * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
            * np.sin(varphi)
            + np.cos(theta)
            * (
                18
                + 5 * np.sin(phi)
                + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
                + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
            )
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=2)

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit.tf", wires=3, shots=None)
        Identity = np.array([[1, 0], [0, 1]])
        ham = np.kron(np.kron(Identity, Identity), A)
        obs = qml.Hermitian(ham, wires=[2, 1, 0])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.tf
@pytest.mark.parametrize("theta, phi, varphi", all_angles)
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, varphi, tol):
        """Tests for variance calculation"""
        dev = DefaultQubitTF(wires=1)
        # test correct variance for <Z> of a rotated state

        with qml.queuing.AnnotatedQueue() as q:
            _ = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
            _ = [qml.var(qml.PauliZ(wires=[0]))]

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_hermitian(self, theta, phi, varphi, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = DefaultQubitTF(wires=2)

        # test correct variance for <H> of a rotated state
        ham = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        with qml.queuing.AnnotatedQueue() as q:
            _ = [qml.RX(phi, wires=0), qml.RY(theta, wires=0)]
            _ = [qml.var(qml.Hermitian(ham, wires=[0]))]

        tape = qml.tape.QuantumScript.from_queue(q)
        res = dev.execute(tape)
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.tf", wires=3)

        _A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(_A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)


#####################################################
# QNode-level integration tests
#####################################################


@pytest.mark.tf
class TestQNodeIntegration:
    """Integration tests for default.qubit.tf. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qubit.tf", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "supports_broadcasting": True,
            "passthru_interface": "tf",
            "passthru_devices": {
                "torch": "default.qubit.torch",
                "tf": "default.qubit.tf",
                "autograd": "default.qubit.autograd",
                "jax": "default.qubit.jax",
            },
        }
        assert cap == capabilities

    def test_load_tensornet_tf_device(self):
        """Test that the tensor network plugin loads correctly"""
        dev = qml.device("default.qubit.tf", wires=2)
        assert dev.num_wires == 2
        assert dev.shots is None
        assert dev.short_name == "default.qubit.tf"
        assert dev.capabilities()["passthru_interface"] == "tf"

    def test_qubit_circuit(self, tol):
        """Test that the tensor network plugin provides correct
        result for a simple circuit using the old QNode."""
        p = tf.Variable(0.543)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -tf.math.sin(p)

        assert circuit.gradient_fn == "backprop"
        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_circuit_broadcasted(self, tol):
        """Test that the tensor network plugin provides correct
        result for a simple circuit with broadcasting using the old QNode."""
        p = tf.Variable([0.543, 0.21, 2.41])

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -tf.math.sin(p)

        assert circuit.gradient_fn == "backprop"
        assert np.allclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.qubit.tf", wires=2)

        state = dev.state
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(np.pi / 4, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        amplitude = np.exp(-1j * np.pi / 8) / np.sqrt(2)

        expected = np.array([amplitude, 0, np.conj(amplitude), 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)

    def test_correct_state_broadcasted(self, tol):
        """Test that the device state is correct after applying a
        broadcasted quantum function on the device"""

        dev = qml.device("default.qubit.tf", wires=2)

        state = dev.state
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(tf.constant([np.pi / 4, np.pi / 2]), wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        phase = np.exp(-1j * np.pi / 8)

        expected = np.array(
            [
                [phase / np.sqrt(2), 0, np.conj(phase) / np.sqrt(2), 0],
                [phase**2 / np.sqrt(2), 0, np.conj(phase) ** 2 / np.sqrt(2), 0],
            ]
        )
        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_one_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the one-qubit single parameter rotations by passing
        a TF data structure as a parameter"""
        dev = qml.device("default.qubit.tf", wires=1)
        state = init_state(1)

        @qml.qnode(dev, interface="tf")
        def circuit(params):
            qml.StatePrep(state, wires=[0])
            op(params[0], wires=[0])
            return qml.expval(qml.PauliZ(0))

        # Pass a TF Variable to the qfunc
        params = tf.Variable(np.array([theta]))
        circuit(params)
        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res.numpy(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, 4.213])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the two-qubit single parameter rotations by passing
        a TF data structure as a parameter"""
        dev = qml.device("default.qubit.tf", wires=2)
        state = init_state(2)

        @qml.qnode(dev, interface="tf")
        def circuit(params):
            qml.StatePrep(state, wires=[0, 1])
            op(params[0], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # Pass a TF Variable to the qfunc
        params = tf.Variable(np.array([theta]))
        circuit(params)
        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res.numpy(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, 4.213])
    @pytest.mark.parametrize("op,func", four_qubit_param)
    def test_four_qubit_param_gates(self, theta, op, func, init_state, tol):
        """Test the integration of the four-qubit single parameter rotations by passing
        a TF data structure as a parameter"""
        dev = qml.device("default.qubit.tf", wires=4)
        state = init_state(4)

        @qml.qnode(dev, interface="tf")
        def circuit(params):
            qml.StatePrep(state, wires=[0, 1, 2, 3])
            op(params[0], wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        # Pass a TF Variable to the qfunc
        params = tf.Variable(np.array([theta]))
        circuit(params)
        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res.numpy(), expected, atol=tol, rtol=0)

    def test_controlled_rotation_integration(self, init_state, tol):
        """Test the integration of the two-qubit controlled rotation by passing
        a TF data structure as a parameter"""
        dev = qml.device("default.qubit.tf", wires=2)
        a = 1.7
        b = 1.3432
        c = -0.654
        state = init_state(2)

        @qml.qnode(dev, interface="tf")
        def circuit(params):
            qml.StatePrep(state, wires=[0, 1])
            qml.CRot(params[0], params[1], params[2], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # Pass a TF Variable to the qfunc
        params = tf.Variable(np.array([a, b, c]))
        circuit(params)
        res = dev.state
        expected = CRot3(a, b, c) @ state
        assert np.allclose(res.numpy(), expected, atol=tol, rtol=0)


@pytest.mark.tf
class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    def test_jacobian_variable_multiply(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = tf.Variable(0.43316321)
        y = tf.Variable(0.2162158)
        z = tf.Variable(0.75110998)

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit([x, y, z])

        expected = tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2) - tf.math.sin(
            3 * x
        ) * tf.math.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tf.stack(tape.jacobian(res, [x, y, z]), axis=0)

        expected = np.array(
            [
                -3
                * (
                    tf.math.sin(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.sin(z / 2)
                ),
                -tf.math.cos(3 * x) * tf.math.sin(y) * tf.math.cos(z / 2),
                -0.5
                * (
                    tf.math.sin(3 * x) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.sin(z / 2)
                ),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_variable_multiply_broadcasted(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of broadcasted parameters multiplied by scalars"""
        x = tf.Variable([0.43316321, 92.1, -0.5129])
        y = tf.Variable([0.2162158, 0.241, -0.51])
        z = tf.Variable([0.75110998, 0.12512, 9.12])

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit([x, y, z])

        expected = tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2) - tf.math.sin(
            3 * x
        ) * tf.math.sin(z / 2)
        assert qml.math.shape(res) == (3,)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, [x, y, z])
        res = qml.math.stack([qml.math.diag(j.numpy()) for j in jac])

        expected = np.array(
            [
                -3
                * (
                    tf.math.sin(3 * x) * tf.math.cos(y) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.sin(z / 2)
                ),
                -tf.math.cos(3 * x) * tf.math.sin(y) * tf.math.cos(z / 2),
                -0.5
                * (
                    tf.math.sin(3 * x) * tf.math.cos(z / 2)
                    + tf.math.cos(3 * x) * tf.math.cos(y) * tf.math.sin(z / 2)
                ),
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = tf.Variable([x, y, z])
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated_broadcasted(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result in the case of repeated broadcasted parameters"""
        x = tf.Variable([0.433, 92.1, -0.512])
        y = tf.Variable([0.218, 0.241, -0.51])
        z = tf.Variable([0.71, 0.152, 9.12])
        p = tf.Variable([x, y, z])
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0 * x]
        )
        assert all(np.allclose(res[i, :, i], expected[:, i], atol=tol, rtol=0) for i in range(3))

    def test_backprop_jacobian_agrees_parameter_shift(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.tf device
        gives the correct result with respect to the parameter-shift method"""
        p = pnp.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])
        p_tf = tf.Variable(p, trainable=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.qubit.legacy", wires=3)
        dev2 = qml.device("default.qubit.legacy", wires=3)

        def cost(x):
            return qml.math.stack(circuit(x))

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="tf")
        circuit2 = qml.QNode(cost, dev2, diff_method="parameter-shift")

        with tf.GradientTape() as tape:
            res = tf.experimental.numpy.hstack(circuit1(p_tf))

        assert np.allclose(res, circuit2(p), atol=tol, rtol=0)

        assert circuit1.gradient_fn == "backprop"
        assert circuit2.gradient_fn is qml.gradients.param_shift

        res = tape.jacobian(res, p_tf)
        assert np.allclose(res, qml.jacobian(circuit2)(p), atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [[0], ["abc"]])
    def test_state_differentiability(self, wires, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            return qml.state()

        a = tf.Variable(0.54)

        with tf.GradientTape() as tape:
            res = tf.abs(circuit(a)) ** 2
            res = res[1] - res[0]

        grad = tape.gradient(res, a)
        expected = tf.sin(a)
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_state_differentiability_broadcasted(self, tol):
        """Test that the broadcasted device state can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable([0.54, 0.32, 1.2])

        with tf.GradientTape() as tape:
            circuit(a)
            res = tf.abs(dev.state) ** 2
            res = res[:, 1] - res[:, 0]

        jac = tape.jacobian(res, a)
        expected = tf.sin(a)
        assert np.allclose(qml.math.diag(jac.numpy()), expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = tf.Variable(0.54)
        b = tf.Variable(0.12)

        with tf.GradientTape() as tape:
            # get the probability of wire 1
            prob_wire_1 = circuit(a, b)
            # compute Prob(|1>_1) - Prob(|0>_1)
            res = prob_wire_1[1] - prob_wire_1[0]  # pylint:disable=unsubscriptable-object

        expected = -tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.gradient(res, [a, b])
        expected = [tf.sin(a) * tf.cos(b), tf.cos(a) * tf.sin(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability_broadcasted(self, tol):
        """Test that the broadcasted device probability can be differentiated"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = tf.Variable([0.54, 0.32, 1.2])
        b = tf.Variable(0.12)

        with tf.GradientTape() as tape:
            # get the probability of wire 1
            prob_wire_1 = circuit(a, b)
            # compute Prob(|1>_1) - Prob(|0>_1)
            res = prob_wire_1[:, 1] - prob_wire_1[:, 0]  # pylint:disable=unsubscriptable-object

        expected = -tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, [a, b])
        expected = [tf.sin(a) * tf.cos(b), tf.cos(a) * tf.sin(b)]
        assert np.allclose(qml.math.diag(jac[0].numpy()), expected[0])
        assert np.allclose(jac[1], expected[1])

    def test_backprop_gradient(self, tol):
        """Tests that the gradient of the qnode is correct"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = -0.234
        b = 0.654

        a_tf = tf.Variable(a, dtype=tf.float64)
        b_tf = tf.Variable(b, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch([a_tf, b_tf])
            res = circuit(a_tf, b_tf)

        # the analytic result of evaluating circuit(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)

        # the analytic result of evaluating grad(circuit(a, b))
        expected_grad = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )

        # pylint:disable=no-member
        assert np.allclose(res.numpy(), expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, [a_tf, b_tf])
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    def test_backprop_gradient_broadcasted(self, tol):
        """Tests that the gradient of the broadcasted qnode is correct"""
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = np.array(0.12)
        b = np.array([0.54, 0.32, 1.2])

        a_tf = tf.Variable(a, dtype=tf.float64)
        b_tf = tf.Variable(b, dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch([a_tf, b_tf])
            res = circuit(a_tf, b_tf)

        # the analytic result of evaluating circuit(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)

        # the analytic result of evaluating grad(circuit(a, b))
        expected_jac = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )

        # pylint:disable=no-member
        assert np.allclose(res.numpy(), expected_cost, atol=tol, rtol=0)

        jac = tape.jacobian(res, [a_tf, b_tf])
        assert np.allclose(jac[0], expected_jac[0], atol=tol, rtol=0)
        assert np.allclose(qml.math.diag(jac[1].numpy()), expected_jac[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("x, shift", [(0.0, 0.0), (0.5, -0.5)])
    def test_hessian_at_zero(self, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.qubit.tf", wires=1)

        shift = tf.constant(shift)
        x = tf.Variable(x)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t1:
                value = circuit(x)
            grad = t1.gradient(value, x)
            jac = t1.jacobian(value, x)
        hess_grad = t2.gradient(grad, x)
        hess_jac = t2.jacobian(jac, x)

        assert qml.math.isclose(grad, 0.0)
        assert qml.math.isclose(hess_grad, -1.0)
        assert qml.math.isclose(hess_jac, -1.0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_tf_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TensorFlow interface, using a variety of differentiation methods."""
        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.StatePrep(1j * np.array([1, -1]) / np.sqrt(2), wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        # Check that the correct QNode type is being used.
        if diff_method == "backprop":
            assert circuit.gradient_fn == "backprop"
        elif diff_method == "parameter-shift":
            assert circuit.gradient_fn is qml.gradients.param_shift
        elif diff_method == "finite-diff":
            assert circuit.gradient_fn is qml.gradients.finite_diff

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = tf.Variable([theta, phi, lam], dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(params)
            res = cost(params)

        # check that the result is correct
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert np.allclose(res.numpy(), expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, params)

        # check that the gradient is correct
        expected_grad = (
            np.array(
                [
                    np.sin(theta) * np.cos(lam) * np.cos(phi),
                    np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                    np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
                ]
            )
            * 2
            * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
        )
        assert np.allclose(res.numpy(), expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd", "torch"])
    def test_error_backprop_wrong_interface(self, interface, tol):
        """Tests that an error is raised if diff_method='backprop' but not using
        the TF interface"""
        dev = qml.device("default.qubit.tf", wires=1)

        def circuit(x, w=None):
            qml.RZ(x, wires=w)
            return qml.expval(qml.PauliX(w))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="default.qubit.tf only supports diff_method='backprop' when using the tf interface",
        ):
            qml.qnode(dev, diff_method="backprop", interface=interface)(circuit)

    def test_hermitian_backprop(self, tol):
        """Test that backprop with qml.Hermitian works correctly"""
        dev = qml.device("default.qubit.tf", wires=2)

        K = tf.linalg.diag([1, 2, 3, 4])

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(op):
            qml.PauliX(0)
            qml.PauliX(1)
            return qml.expval(op)

        res = circuit(qml.Hermitian(K, wires=range(2)))
        assert isinstance(res, tf.Tensor)
        assert res == 4.0


@pytest.mark.tf
class TestSamples:
    """Tests for sampling outputs"""

    def test_sample_observables(self):
        """Test that the device allows for sampling from observables."""
        shots = 100
        dev = qml.device("default.qubit.tf", wires=2, shots=shots)

        @qml.qnode(dev, diff_method="best", interface="tf")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(0))

        a = tf.Variable(0.54, dtype=tf.float64)
        res = circuit(a)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (shots,)  # pylint:disable=comparison-with-callable
        assert set(res.numpy()) == {-1, 1}  # pylint:disable=no-member

    def test_estimating_marginal_probability(self, tol):
        """Test that the probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=[0])

        res = circuit()

        assert isinstance(res, tf.Tensor)

        expected = np.array([0, 1])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_full_probability(self, tol):
        """Test that the probability of all wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            return qml.probs(wires=[0, 1])

        res = circuit()

        assert isinstance(res, tf.Tensor)

        expected = np.array([0, 0, 0, 1])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_estimating_expectation_values(self, tol):
        """Test that estimating expectation values using a finite number
        of shots produces a numeric tensor"""
        dev = qml.device("default.qubit.tf", wires=3, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = tf.Variable(0.543, dtype=tf.float64)
        b = tf.Variable(0.43, dtype=tf.float64)

        res = circuit(a, b)
        assert isinstance(res, tuple)

        # We don't check the expected value due to stochasticity, but
        # leave it here for completeness.
        # expected = [tf.cos(a), tf.cos(a) * tf.cos(b)]
        # assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.tf
class TestSamplesBroadcasted:
    """Tests for broadcasted sampling outputs"""

    def test_sample_observables_broadcasted(self):
        """Test that the device allows for broadcasted sampling from observables."""
        shots = 100
        dev = qml.device("default.qubit.tf", wires=2, shots=shots)

        @qml.qnode(dev, diff_method="best", interface="tf")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(0))

        a = tf.Variable([0.54, -0.32, 0.19], dtype=tf.float64)
        res = circuit(a)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (3, shots)  # pylint:disable=comparison-with-callable
        assert set(res.numpy().flat) == {-1, 1}  # pylint:disable=no-member

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_estimating_marginal_probability_broadcasted(self, batch_size, tol):
        """Test that the broadcasted probability of a subset of wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit():
            qml.RX(tf.zeros(batch_size), 0)
            qml.PauliX(0)
            return qml.probs(wires=[0])

        res = circuit()

        assert isinstance(res, tf.Tensor)
        assert qml.math.shape(res) == (batch_size, 2)

        expected = np.array([[0, 1]] * batch_size)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_estimating_full_probability_broadcasted(self, batch_size, tol):
        """Test that the broadcasted probability of all wires is accurately estimated."""
        dev = qml.device("default.qubit.tf", wires=2, shots=1000)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit():
            qml.RX(tf.zeros(batch_size), 0)
            qml.PauliX(0)
            qml.PauliX(1)
            return qml.probs(wires=[0, 1])

        res = circuit()

        assert isinstance(res, tf.Tensor)
        assert qml.math.shape(res) == (batch_size, 4)

        expected = np.array([[0, 0, 0, 1]] * batch_size)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.skip("Parameter broadcasting is not supported for multiple return values yet")
    @pytest.mark.parametrize("a", [[0.54, -0.32, 0.19], [0.52]])
    def test_estimating_expectation_values_broadcasted(self, a, tol):
        """Test that estimating broadcasted expectation values using a finite number
        of shots produces a numeric tensor"""
        batch_size = len(a)
        dev = qml.device("default.qubit.tf", wires=3, shots=None)

        @qml.qnode(dev, diff_method=None, interface="tf")
        def circuit(a, b):
            qml.RX(a, wires=[0])
            qml.RX(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = tf.Variable(a, dtype=tf.float64)
        b = tf.Variable(0.43, dtype=tf.float64)

        res = circuit(a, b)
        assert isinstance(res, tf.Tensor)
        assert qml.math.shape(res) == (batch_size, 2)


@pytest.mark.tf
def test_asarray_ragged_dtype_conversion(monkeypatch):
    """Test that the _asarray internal method handles ragged arrays well when
    the dtype argument was provided."""
    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    dev = qml.device("default.qubit.tf", wires=2)

    def mock_func(arr, dtype):
        raise InvalidArgumentError(
            None, None, "SomeMessage"
        )  # args passed are non-significant for test case

    monkeypatch.setattr(tf, "convert_to_tensor", mock_func)
    res = dev._asarray(np.array([1]), tf.float32)
    assert res.dtype == tf.float32


@pytest.mark.tf
class TestGetBatchSize:
    """Tests for the updated helper method ``_get_batch_size`` of ``DefaultQubitTF``."""

    @pytest.mark.parametrize("shape", [(4, 4), (1, 8), (4,)])
    def test_batch_size_None(self, shape):
        """Test that a ``batch_size=None`` is reported correctly."""
        dev = qml.device("default.qubit.tf", wires=2)
        tensor0 = np.ones(shape, dtype=complex)
        assert dev._get_batch_size(tensor0, shape, qml.math.prod(shape)) is None

    @pytest.mark.parametrize("shape", [(4, 4), (1, 8), (4,)])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_batch_size_int(self, shape, batch_size):
        """Test that an integral ``batch_size`` is reported correctly."""
        dev = qml.device("default.qubit.tf", wires=2)
        full_shape = (batch_size,) + shape
        tensor0 = np.ones(full_shape, dtype=complex)
        assert dev._get_batch_size(tensor0, shape, qml.math.prod(shape)) == batch_size

    def test_invalid_tensor(self):
        """Test that an error is raised if a tensor is provided that does not
        have a proper shape/ndim."""
        dev = qml.device("default.qubit.tf", wires=2)
        with pytest.raises(ValueError, match="Can't convert non-rectangular Python"):
            dev._get_batch_size([qml.math.ones((2, 3)), qml.math.ones((2, 2))], (2, 2, 2), 8)

    @pytest.mark.parametrize("jit_compile", [True, False])
    def test_no_error_abstract_tensor(self, jit_compile):
        """Test that no error is raised if an abstract tensor is provided"""
        dev = qml.device("default.qubit.tf", wires=2)
        signature = (tf.TensorSpec(shape=None, dtype=tf.float32),)

        @tf.function(jit_compile=jit_compile, input_signature=signature)
        def get_batch_size(tensor):
            return dev._get_batch_size(tensor, (2,), 2)

        assert get_batch_size(tf.Variable(0.2)) is None
