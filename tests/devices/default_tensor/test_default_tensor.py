# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the DefaultTensor class.
"""


import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.default_tensor import _operations, stopping_condition
from pennylane.exceptions import DeviceError, WireError
from pennylane.math.decomposition import givens_decomposition
from pennylane.typing import TensorLike

quimb = pytest.importorskip("quimb")

pytestmark = pytest.mark.external

# gates for which device support is tested
operations_list = {
    "Identity": qml.Identity(wires=[0]),
    "Identity()": qml.Identity(),
    "BlockEncode": qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
    "CNOT": qml.CNOT(wires=[0, 1]),
    "CRX": qml.CRX(1.234, wires=[0, 1]),
    "CRY": qml.CRY(1.234, wires=[0, 1]),
    "CRZ": qml.CRZ(1.234, wires=[0, 1]),
    "CRot": qml.CRot(1.234, 0, 0, wires=[0, 1]),
    "CSWAP": qml.CSWAP(wires=[0, 1, 2]),
    "CZ": qml.CZ(wires=[0, 1]),
    "CCZ": qml.CCZ(wires=[0, 1, 2]),
    "CY": qml.CY(wires=[0, 1]),
    "CH": qml.CH(wires=[0, 1]),
    "DiagonalQubitUnitary": qml.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "MultiRZ": qml.MultiRZ(1.234, wires=[0, 1]),
    "MultiRZ(1)": qml.MultiRZ(1.234, wires=[0]),
    "PauliX": qml.X(0),
    "PauliY": qml.Y(0),
    "PauliZ": qml.Z(0),
    "X": qml.X([0]),
    "Y": qml.Y([0]),
    "Z": qml.Z([0]),
    "PhaseShift": qml.PhaseShift(1.234, wires=[0]),
    "PCPhase": qml.PCPhase(1.234, 1, wires=[0, 1]),
    "ControlledPhaseShift": qml.ControlledPhaseShift(1.234, wires=[0, 1]),
    "CPhaseShift00": qml.CPhaseShift00(1.234, wires=[0, 1]),
    "CPhaseShift01": qml.CPhaseShift01(1.234, wires=[0, 1]),
    "CPhaseShift10": qml.CPhaseShift10(1.234, wires=[0, 1]),
    "QubitUnitary": qml.QubitUnitary(np.eye(2), wires=[0]),
    "SpecialUnitary": qml.SpecialUnitary(np.array([0.2, -0.1, 2.3]), wires=1),
    "ControlledQubitUnitary": qml.ControlledQubitUnitary(np.eye(2), wires=[1, 0]),
    "MultiControlledX": qml.MultiControlledX(wires=[1, 2, 0]),
    "IntegerComparator": qml.IntegerComparator(1, geq=True, wires=[0, 1, 2]),
    "RX": qml.RX(1.234, wires=[0]),
    "RY": qml.RY(1.234, wires=[0]),
    "RZ": qml.RZ(1.234, wires=[0]),
    "Rot": qml.Rot(1.234, 0, 0, wires=[0]),
    "S": qml.S(wires=[0]),
    "Adjoint(S)": qml.adjoint(qml.S(wires=[0])),
    "SWAP": qml.SWAP(wires=[0, 1]),
    "ISWAP": qml.ISWAP(wires=[0, 1]),
    "PSWAP": qml.PSWAP(1.234, wires=[0, 1]),
    "ECR": qml.ECR(wires=[0, 1]),
    "Adjoint(ISWAP)": qml.adjoint(qml.ISWAP(wires=[0, 1])),
    "T": qml.T(wires=[0]),
    "Adjoint(T)": qml.adjoint(qml.T(wires=[0])),
    "SX": qml.SX(wires=[0]),
    "Adjoint(SX)": qml.adjoint(qml.SX(wires=[0])),
    "Toffoli": qml.Toffoli(wires=[0, 1, 2]),
    "QFT": qml.templates.QFT(wires=[0, 1, 2]),
    "IsingXX": qml.IsingXX(1.234, wires=[0, 1]),
    "IsingYY": qml.IsingYY(1.234, wires=[0, 1]),
    "IsingZZ": qml.IsingZZ(1.234, wires=[0, 1]),
    "IsingXY": qml.IsingXY(1.234, wires=[0, 1]),
    "SingleExcitation": qml.SingleExcitation(1.234, wires=[0, 1]),
    "SingleExcitationPlus": qml.SingleExcitationPlus(1.234, wires=[0, 1]),
    "SingleExcitationMinus": qml.SingleExcitationMinus(1.234, wires=[0, 1]),
    "DoubleExcitation": qml.DoubleExcitation(1.234, wires=[0, 1, 2, 3]),
    "QubitCarry": qml.QubitCarry(wires=[0, 1, 2, 3]),
    "QubitSum": qml.QubitSum(wires=[0, 1, 2]),
    "PauliRot": qml.PauliRot(1.234, "XXYY", wires=[0, 1, 2, 3]),
    "PauliRot(1)": qml.PauliRot(1.234, "X", wires=[0]),
    "U1": qml.U1(1.234, wires=0),
    "U2": qml.U2(1.234, 0.2, wires=0),
    "U3": qml.U3(1.234, 0.2, 0.3, wires=0),
    "SISWAP": qml.SISWAP(wires=[0, 1]),
    "Adjoint(SISWAP)": qml.adjoint(qml.SISWAP(wires=[0, 1])),
    "OrbitalRotation": qml.OrbitalRotation(1.234, wires=[0, 1, 2, 3]),
    "FermionicSWAP": qml.FermionicSWAP(1.234, wires=[0, 1]),
    "GlobalPhase": qml.GlobalPhase(1.23423, wires=[0, 1]),
    "GlobalPhase()": qml.GlobalPhase(1.23423),
}

all_ops = operations_list.keys()

# observables for which device support is tested
observables_list = {
    "Identity": qml.Identity(wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "Hermitian": qml.Hermitian(np.eye(2), wires=[0]),
    "PauliX": qml.PauliX(0),
    "PauliY": qml.PauliY(0),
    "PauliZ": qml.PauliZ(0),
    "X": qml.X(0),
    "Y": qml.Y(0),
    "Z": qml.Z(0),
    "Projector": [
        qml.Projector(np.array([1]), wires=[0]),
        qml.Projector(np.array([0, 1]), wires=[0]),
    ],
    "SparseHamiltonian": qml.SparseHamiltonian(csr_matrix(np.eye(8)), wires=[0, 1, 2]),
    "Hamiltonian": qml.Hamiltonian([1, 1], [qml.Z(0), qml.X(0)]),
    "LinearCombination": qml.ops.LinearCombination([1, 1], [qml.Z(0), qml.X(0)]),
}

all_obs = observables_list.keys()


def test_name():
    """Test the name of DefaultTensor."""
    assert qml.device("default.tensor").name == "default.tensor"


def test_wires():
    """Test that a device can be created with wires."""
    assert qml.device("default.tensor").wires is None
    assert qml.device("default.tensor", wires=2).wires == qml.wires.Wires([0, 1])
    assert qml.device("default.tensor", wires=[0, 2]).wires == qml.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        qml.device("default.tensor").wires = [0, 1]


def test_wires_runtime():
    """Test that this device can execute a tape with wires determined at runtime if they are not provided."""
    dev = qml.device("default.tensor")
    ops = [qml.Identity(0), qml.Identity((0, 1)), qml.RX(2, 0), qml.RY(1, 5), qml.RX(2, 1)]
    measurements = [qml.expval(qml.PauliZ(15))]
    tape = qml.tape.QuantumScript(ops, measurements)
    assert dev.execute(tape) == 1.0


def test_wires_runtime_error():
    """Test that this device raises an error if the wires are provided by user and there is a mismatch."""
    dev = qml.device("default.tensor", wires=1)
    ops = [qml.Identity(0), qml.Identity((0, 1)), qml.RX(2, 0), qml.RY(1, 5), qml.RX(2, 1)]
    measurements = [qml.expval(qml.PauliZ(15))]
    tape = qml.tape.QuantumScript(ops, measurements)

    with pytest.raises(WireError):
        dev.execute(tape)


@pytest.mark.parametrize("max_bond_dim", [None, 10])
@pytest.mark.parametrize("cutoff", [1e-16, 1e-12])
def test_kwargs_mps(max_bond_dim, cutoff):
    """Test the class initialization with different arguments and returned properties for the MPS method."""

    max_bond_dim = 10
    cutoff = 1e-16
    method = "mps"

    dev = qml.device("default.tensor", method=method, max_bond_dim=max_bond_dim, cutoff=cutoff)

    config = dev.setup_execution_config()
    assert config.device_options["method"] == method
    assert config.device_options["max_bond_dim"] == max_bond_dim
    assert config.device_options["cutoff"] == cutoff
    assert config.device_options["contract"] == "auto-mps"


def test_kwargs_tn():
    """Test the class initialization with different arguments and returned properties for the TN method."""

    method = "tn"
    dev = qml.device("default.tensor", method=method)

    config = dev.setup_execution_config()
    assert config.device_options["method"] == method
    assert config.device_options["contract"] == "auto-split-gate"


def test_invalid_kwarg():
    """Test an invalid keyword argument."""
    with pytest.raises(
        TypeError,
        match="Unexpected argument: fake_arg during initialization of the default.tensor device.",
    ):
        qml.device("default.tensor", fake_arg=None)


def test_invalid_contract():
    """Test an invalid combination of method and contract."""

    with pytest.raises(
        ValueError, match="Unsupported gate contraction option: 'auto-split-gate' for 'mps' method."
    ):
        qml.device("default.tensor", method="mps", contract="auto-split-gate")

    with pytest.raises(
        ValueError, match="Unsupported gate contraction option: 'auto-mps' for 'tn' method."
    ):
        qml.device("default.tensor", method="tn", contract="auto-mps")


@pytest.mark.parametrize("method", ["mps", "tn"])
def test_method(method):
    """Test the device method."""
    assert qml.device("default.tensor", method=method).method == method


def test_invalid_method():
    """Test an invalid method."""
    method = "invalid_method"
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        qml.device("default.tensor", method=method)


@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
def test_data_type(c_dtype):
    """Test the data type."""
    assert qml.device("default.tensor", c_dtype=c_dtype).c_dtype == c_dtype


def test_ivalid_data_type():
    """Test that data type can only be np.complex64 or np.complex128."""
    with pytest.raises(TypeError):
        qml.device("default.tensor", c_dtype=float)


@pytest.mark.parametrize("method", ["mps", "tn"])
def test_draw(method):
    """Test the draw method."""

    dev = qml.device("default.tensor", wires=10, method=method)
    fig = dev.draw(color="auto", title="Test", return_fig=True)
    assert fig is not None


def test_warning_useless_kwargs():
    """Test that a warning is raised if the user provides a combination of arguments that are not used."""

    with pytest.warns():
        qml.device("default.tensor", method="tn", max_bond_dim=10)
        qml.device("default.tensor", method="tn", cutoff=1e-16)


def test_kahypar_warning_not_raised(recwarn):
    """Test that a warning is not raised if the user does not have kahypar installed when initializing the
    default.tensor device"""
    try:
        import kahypar  # pylint: disable=import-outside-toplevel, unused-import

        pytest.skip(reason="Test is for when kahypar is not installed")
    except ImportError:
        _ = qml.device("default.tensor", wires=1)
        assert len(recwarn) == 0


def test_passing_shots_None():
    """Test that passing shots=None on initialization works without error."""
    dev = qml.device("default.tensor", shots=None)
    assert dev.shots == qml.measurements.Shots(None)


def test_passing_finite_shots_error():
    """Test that an error is raised if finite shots are passed on initialization."""

    with pytest.raises(DeviceError, match=r"only supports analytic simulations"):
        qml.device("default.tensor", shots=10)


@pytest.mark.parametrize("method", ["mps", "tn"])
class TestSupportedGatesAndObservables:
    """Test that the DefaultTensor device supports all gates and observables that it claims to support."""

    # Note: we could potentially test each 'contract' option for both methods, but this would significantly
    # increase the number of tests. Furthermore, the 'contract' option is tested in the quimb library itself.

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, operation, method):
        """Test that the device can implement all its supported gates."""

        dev = qml.device("default.tensor", wires=4, method=method)

        tape = qml.tape.QuantumScript(
            [operations_list[operation]],
            [qml.expval(qml.Identity(wires=0))],
        )

        result = dev.execute(circuits=tape)
        assert np.allclose(result, 1.0)

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_yield_correct_state(self, operation, method):
        """Test that the device can implement all its supported gates."""
        nwires = 4
        dq = qml.device("default.qubit", wires=nwires)
        dev = qml.device("default.tensor", wires=nwires, method=method)

        state = np.random.rand(2**nwires) + 1j * np.random.rand(2**nwires)
        state /= np.linalg.norm(state)
        wires = qml.wires.Wires(range(nwires))
        tape = qml.tape.QuantumScript(
            [qml.StatePrep(state, wires=wires), operations_list[operation]],
            [qml.state()],
        )
        result = dev.execute(circuits=[tape])[0]
        ref = dq.execute(circuits=[tape])[0]
        assert np.allclose(result, ref)

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, observable, method):
        """Test that the device can implement all its supported observables."""

        dev = qml.device("default.tensor", wires=3, method=method)

        if observable == "Projector":
            for o in observables_list[observable]:
                tape = qml.tape.QuantumScript(
                    [qml.PauliX(0)],
                    [qml.expval(o)],
                )
                result = dev.execute(circuits=tape)
                assert isinstance(result, (float, np.ndarray))

        else:
            tape = qml.tape.QuantumScript(
                [qml.PauliX(0)],
                [qml.expval(observables_list[observable])],
            )
            result = dev.execute(circuits=tape)
            assert isinstance(result, (float, np.ndarray))

    def test_not_implemented_meas(self, method):
        """Tests that support only exists for `qml.expval` and `qml.var` so far."""

        op = [qml.Identity(0)]
        measurements = [qml.probs(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(op, measurements)

        dev = qml.device("default.tensor", wires=tape.wires, method=method)

        with pytest.raises(NotImplementedError):
            dev.execute(tape)


class TestSupportsDerivatives:
    """Test that DefaultTensor states what kind of derivatives it supports."""

    def test_support_derivatives(self):
        """Test that the device does not support derivatives yet."""
        dev = qml.device("default.tensor")
        assert not dev.supports_derivatives()

    def test_compute_derivatives(self):
        """Test that an error is raised if the `compute_derivatives` method is called."""
        dev = qml.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_derivatives(circuits=None)

    def test_execute_and_compute_derivatives(self):
        """Test that an error is raised if `execute_and_compute_derivative` method is called."""
        dev = qml.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.execute_and_compute_derivatives(circuits=None)

    def test_supports_vjp(self):
        """Test that the device does not support VJP yet."""
        dev = qml.device("default.tensor")
        assert not dev.supports_vjp()

    def test_compute_vjp(self):
        """Test that an error is raised if `compute_vjp` method is called."""
        dev = qml.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_vjp(circuits=None, cotangents=None)

    def test_execute_and_compute_vjp(self):
        """Test that an error is raised if `execute_and_compute_vjp` method is called."""
        dev = qml.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
        ):
            dev.execute_and_compute_vjp(circuits=None, cotangents=None)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("method", ["mps", "tn"])
@pytest.mark.jax
class TestJaxSupport:
    """Test the JAX support for the DefaultTensor device."""

    def test_jax(self, method):
        """Test the device with JAX."""

        jax = pytest.importorskip("jax")
        dev = qml.device("default.tensor", wires=1, method=method)
        ref_dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.Z(0))

        weights = jax.numpy.array([0.2, 0.5, 0.1])
        qnode = qml.QNode(circuit, dev, interface="jax")
        ref_qnode = qml.QNode(circuit, ref_dev, interface="jax")

        assert np.allclose(qnode(weights), ref_qnode(weights))

    def test_jax_jit(self, method):
        """Test the device with JAX's JIT compiler."""

        jax = pytest.importorskip("jax")
        dev = qml.device("default.tensor", wires=1, method=method)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        assert np.allclose(circuit(), 0.0)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("method", ["mps", "tn"])
@pytest.mark.parametrize(
    "operation, expected_output, par",
    [
        (qml.BasisState, [0, 0, 1 + 0j, 0], [1 + 0j, 0]),
        (qml.BasisState, [0, 0, 0, 1 + 0j], [1 + 0j, 1 + 0j]),
        (qml.BasisState, [0, 0, 1, 0], [1, 0]),
        (qml.BasisState, [0, 0, 0, 1], [1, 1]),
        (qml.StatePrep, [0, 0, 1 + 0j, 0], [0, 0, 1 + 0j, 0]),
        (qml.StatePrep, [0, 0, 0, 1 + 0j], [0, 0, 0, 1 + 0j]),
        (qml.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qml.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
        (
            qml.StatePrep,
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
        (
            qml.StatePrep,
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
    ],
)
def test_apply_operation_state_preparation(operation, expected_output, par, method):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""

    par = np.array(par)
    dev = qml.device("default.tensor", method=method, wires=2)

    @qml.qnode(dev)
    def circuit():
        operation(par, wires=[0, 1])
        return qml.state()

    state = circuit()
    assert np.allclose(state, np.array(expected_output), rtol=0)


# At this stage, this test is especially relevant for the MPS method, but we test both methods for consistency.
@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("num_orbitals", [2, 4])
@pytest.mark.parametrize("method", ["mps", "tn"])
def test_wire_order_dense_vector(method, num_orbitals):
    """Test that the wire order is preserved if the initial state is created from a dense vector."""

    dev = qml.device("default.tensor", wires=int(2 * num_orbitals + 1), method=method)
    qubits = dev.wires.tolist()

    wave_fun = np.random.random(2 ** (2 * num_orbitals)) + 1j * np.random.random(
        2 ** (2 * num_orbitals)
    )
    wave_fun = wave_fun / np.linalg.norm(wave_fun)

    X0 = np.random.random((num_orbitals, num_orbitals))
    U0 = expm((X0 + X0.T) / 2.0)

    def basis_rotation_ops(unitary_matrix, wires):
        _, givens_list = givens_decomposition(unitary_matrix)

        for grot_mat, indices in givens_list:
            theta = np.arccos(np.real(grot_mat[1, 1]))
            qml.SingleExcitation(2 * theta, wires=[int(wires[indices[0]]), int(wires[indices[1]])])

    control_wires = 1

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(wave_fun, wires=qubits[1:])
        qml.Hadamard(qubits[0])
        basis_rotation_ops(U0, [int(2 * i + 1 + control_wires) for i in range(num_orbitals)])
        return qml.state()

    state = circuit()
    assert isinstance(state, TensorLike)
    assert len(state) == 2 ** (2 * num_orbitals + 1)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
class TestMCMs:
    """Test that default.tensor can handle mid circuit measurements."""

    @pytest.mark.parametrize("mcm_method", ("one-shot", "tree-traversal"))
    def test_error_on_unsupported_mcm_method(self, mcm_method):
        """Test that an error is raised on unsupported mcm methods."""

        mcm_config = qml.devices.MCMConfig(mcm_method=mcm_method)
        config = qml.devices.ExecutionConfig(mcm_config=mcm_config)
        with pytest.raises(DeviceError, match=r"only supports the deferred measurement principle."):
            qml.device("default.tensor").preprocess(config)

    def test_simple_mcm_present(self):
        """Test that the device can execute a circuit with a mid circuit measurement."""

        dev = qml.device("default.tensor")

        @qml.qnode(dev)
        def circuit():
            qml.measure(0)
            return qml.expval(qml.Z(0))

        res = circuit()
        assert qml.math.allclose(res, 1)

    def test_mcm_conditional(self):
        """Test that the device execute a circuit with an MCM and a conditional."""

        dev = qml.device("default.tensor")

        @qml.qnode(dev)
        def circuit(x):
            m0 = qml.measure(0)
            qml.cond(~m0, qml.RX)(x, 0)
            return qml.expval(qml.Z(0))

        res = circuit(0.5)
        assert qml.math.allclose(res, np.cos(0.5))


class TestPreprocessingTransforms:
    """Tests for the preprocessing transform pipeline."""

    def test_preprocess_transforms_structure(self):
        """Test that the preprocessing transforms are set up correctly."""
        dev = qml.device("default.tensor", wires=3)
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Check that we have the expected transforms
        transform_names = [
            transform_container.transform.__name__ for transform_container in program
        ]
        expected_transforms = [
            "validate_measurements",
            "validate_observables",
            "validate_device_wires",
            "defer_measurements",
            "decompose",
            "broadcast_expand",
        ]

        for expected_transform in expected_transforms:
            assert expected_transform in transform_names

    def test_decompose_transform_has_device_wires_and_target_gates(self):
        """Test that the decompose transform is configured with device_wires and target_gates."""
        dev = qml.device("default.tensor", wires=[0, 1, 2])
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Find the decompose transform
        decompose_transform = None
        for transform_container in program:
            if transform_container.transform.__name__ == "decompose":
                decompose_transform = transform_container
                break

        assert decompose_transform is not None

        # Check that device_wires and target_gates are passed correctly
        assert "device_wires" in decompose_transform.kwargs
        assert "target_gates" in decompose_transform.kwargs
        assert decompose_transform.kwargs["device_wires"] == dev.wires
        assert decompose_transform.kwargs["target_gates"] == _operations

    def test_decompose_with_stopping_condition(self):
        """Test that decompose transform uses the correct stopping condition."""
        dev = qml.device("default.tensor", wires=3)
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Find the decompose transform
        decompose_transform = None
        for transform_container in program:
            if transform_container.transform.__name__ == "decompose":
                decompose_transform = transform_container
                break

        assert decompose_transform is not None
        assert "stopping_condition" in decompose_transform.kwargs
        assert decompose_transform.kwargs["stopping_condition"] == stopping_condition

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.integration
    def test_integration_with_qnode(self):
        """Test integration with QNode to ensure the device works end-to-end."""
        dev = qml.device("default.tensor", wires=3)

        @qml.qnode(dev)
        def circuit():
            # Use an operation that needs decomposition
            qml.QFT(wires=[0, 1])
            return qml.expval(qml.Z(0))

        # This should work without errors
        result = circuit()
        assert isinstance(result, (float, np.floating))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.integration
    def test_integration_with_multiple_decomposition_layers(self):
        """Test that operations requiring multiple layers of decomposition work."""
        dev = qml.device("default.tensor", wires=4)

        @qml.qnode(dev)
        def circuit():
            # Operations that may require multiple decomposition steps
            qml.QFT(wires=[0, 1, 2])
            qml.GroverOperator(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        # This should work without errors
        result = circuit()
        assert isinstance(result, (float, np.floating))


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestGraphModeExclusiveFeatures:  # pylint: disable=too-few-public-methods
    """Tests that only work when graph mode is enabled."""

    @pytest.mark.parametrize(("wires", "expected_program_len"), [(1, 2), (2, 2), (4, 1), (None, 1)])
    def test_work_wire_constraint_respected(self, wires, expected_program_len):
        """Test that decompositions requiring more work wires than available are discarded."""

        # Create a mock operation with different decomposition options
        class MyOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 1

        # Fallback decomposition (no work wires needed)
        @qml.register_resources({qml.Hadamard: 2})
        def decomp_fallback(wires):
            qml.Hadamard(wires)
            qml.Hadamard(wires)

        # Work wire decomposition (needs more wires than available)
        @qml.register_resources({qml.PauliX: 1}, work_wires={"burnable": 3})
        def decomp_with_work_wire(wires):
            qml.PauliX(wires)

        qml.add_decomps(MyOp, decomp_fallback, decomp_with_work_wire)

        tape = qml.tape.QuantumScript([MyOp(0)], [qml.expval(qml.Z(0))])
        dev = qml.device("default.tensor", wires=wires)
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        assert len(out_tape.operations) == expected_program_len
