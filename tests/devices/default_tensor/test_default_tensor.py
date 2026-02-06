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

import pennylane as qp
from pennylane.devices import ExecutionConfig
from pennylane.devices.default_tensor import _operations, stopping_condition
from pennylane.exceptions import DeviceError, WireError
from pennylane.math.decomposition import givens_decomposition
from pennylane.typing import TensorLike

quimb = pytest.importorskip("quimb")

pytestmark = pytest.mark.external

# gates for which device support is tested
operations_list = {
    "Identity": qp.Identity(wires=[0]),
    "Identity()": qp.Identity(),
    "BlockEncode": qp.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
    "CNOT": qp.CNOT(wires=[0, 1]),
    "CRX": qp.CRX(1.234, wires=[0, 1]),
    "CRY": qp.CRY(1.234, wires=[0, 1]),
    "CRZ": qp.CRZ(1.234, wires=[0, 1]),
    "CRot": qp.CRot(1.234, 0, 0, wires=[0, 1]),
    "CSWAP": qp.CSWAP(wires=[0, 1, 2]),
    "CZ": qp.CZ(wires=[0, 1]),
    "CCZ": qp.CCZ(wires=[0, 1, 2]),
    "CY": qp.CY(wires=[0, 1]),
    "CH": qp.CH(wires=[0, 1]),
    "DiagonalQubitUnitary": qp.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qp.Hadamard(wires=[0]),
    "MultiRZ": qp.MultiRZ(1.234, wires=[0, 1]),
    "MultiRZ(1)": qp.MultiRZ(1.234, wires=[0]),
    "PauliX": qp.X(0),
    "PauliY": qp.Y(0),
    "PauliZ": qp.Z(0),
    "X": qp.X([0]),
    "Y": qp.Y([0]),
    "Z": qp.Z([0]),
    "PhaseShift": qp.PhaseShift(1.234, wires=[0]),
    "PCPhase": qp.PCPhase(1.234, 1, wires=[0, 1]),
    "ControlledPhaseShift": qp.ControlledPhaseShift(1.234, wires=[0, 1]),
    "CPhaseShift00": qp.CPhaseShift00(1.234, wires=[0, 1]),
    "CPhaseShift01": qp.CPhaseShift01(1.234, wires=[0, 1]),
    "CPhaseShift10": qp.CPhaseShift10(1.234, wires=[0, 1]),
    "QubitUnitary": qp.QubitUnitary(np.eye(2), wires=[0]),
    "SpecialUnitary": qp.SpecialUnitary(np.array([0.2, -0.1, 2.3]), wires=1),
    "ControlledQubitUnitary": qp.ControlledQubitUnitary(np.eye(2), wires=[1, 0]),
    "MultiControlledX": qp.MultiControlledX(wires=[1, 2, 0]),
    "IntegerComparator": qp.IntegerComparator(1, geq=True, wires=[0, 1, 2]),
    "RX": qp.RX(1.234, wires=[0]),
    "RY": qp.RY(1.234, wires=[0]),
    "RZ": qp.RZ(1.234, wires=[0]),
    "Rot": qp.Rot(1.234, 0, 0, wires=[0]),
    "S": qp.S(wires=[0]),
    "Adjoint(S)": qp.adjoint(qp.S(wires=[0])),
    "SWAP": qp.SWAP(wires=[0, 1]),
    "ISWAP": qp.ISWAP(wires=[0, 1]),
    "PSWAP": qp.PSWAP(1.234, wires=[0, 1]),
    "ECR": qp.ECR(wires=[0, 1]),
    "Adjoint(ISWAP)": qp.adjoint(qp.ISWAP(wires=[0, 1])),
    "T": qp.T(wires=[0]),
    "Adjoint(T)": qp.adjoint(qp.T(wires=[0])),
    "SX": qp.SX(wires=[0]),
    "Adjoint(SX)": qp.adjoint(qp.SX(wires=[0])),
    "Toffoli": qp.Toffoli(wires=[0, 1, 2]),
    "QFT": qp.templates.QFT(wires=[0, 1, 2]),
    "IsingXX": qp.IsingXX(1.234, wires=[0, 1]),
    "IsingYY": qp.IsingYY(1.234, wires=[0, 1]),
    "IsingZZ": qp.IsingZZ(1.234, wires=[0, 1]),
    "IsingXY": qp.IsingXY(1.234, wires=[0, 1]),
    "SingleExcitation": qp.SingleExcitation(1.234, wires=[0, 1]),
    "SingleExcitationPlus": qp.SingleExcitationPlus(1.234, wires=[0, 1]),
    "SingleExcitationMinus": qp.SingleExcitationMinus(1.234, wires=[0, 1]),
    "DoubleExcitation": qp.DoubleExcitation(1.234, wires=[0, 1, 2, 3]),
    "QubitCarry": qp.QubitCarry(wires=[0, 1, 2, 3]),
    "QubitSum": qp.QubitSum(wires=[0, 1, 2]),
    "PauliRot": qp.PauliRot(1.234, "XXYY", wires=[0, 1, 2, 3]),
    "PauliRot(1)": qp.PauliRot(1.234, "X", wires=[0]),
    "U1": qp.U1(1.234, wires=0),
    "U2": qp.U2(1.234, 0.2, wires=0),
    "U3": qp.U3(1.234, 0.2, 0.3, wires=0),
    "SISWAP": qp.SISWAP(wires=[0, 1]),
    "Adjoint(SISWAP)": qp.adjoint(qp.SISWAP(wires=[0, 1])),
    "OrbitalRotation": qp.OrbitalRotation(1.234, wires=[0, 1, 2, 3]),
    "FermionicSWAP": qp.FermionicSWAP(1.234, wires=[0, 1]),
    "GlobalPhase": qp.GlobalPhase(1.23423, wires=[0, 1]),
    "GlobalPhase()": qp.GlobalPhase(1.23423),
}

all_ops = operations_list.keys()

# observables for which device support is tested
observables_list = {
    "Identity": qp.Identity(wires=[0]),
    "Hadamard": qp.Hadamard(wires=[0]),
    "Hermitian": qp.Hermitian(np.eye(2), wires=[0]),
    "PauliX": qp.PauliX(0),
    "PauliY": qp.PauliY(0),
    "PauliZ": qp.PauliZ(0),
    "X": qp.X(0),
    "Y": qp.Y(0),
    "Z": qp.Z(0),
    "Projector": [
        qp.Projector(np.array([1]), wires=[0]),
        qp.Projector(np.array([0, 1]), wires=[0]),
    ],
    "SparseHamiltonian": qp.SparseHamiltonian(csr_matrix(np.eye(8)), wires=[0, 1, 2]),
    "Hamiltonian": qp.Hamiltonian([1, 1], [qp.Z(0), qp.X(0)]),
    "LinearCombination": qp.ops.LinearCombination([1, 1], [qp.Z(0), qp.X(0)]),
}

all_obs = observables_list.keys()


def test_name():
    """Test the name of DefaultTensor."""
    assert qp.device("default.tensor").name == "default.tensor"


def test_wires():
    """Test that a device can be created with wires."""
    assert qp.device("default.tensor").wires is None
    assert qp.device("default.tensor", wires=2).wires == qp.wires.Wires([0, 1])
    assert qp.device("default.tensor", wires=[0, 2]).wires == qp.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        qp.device("default.tensor").wires = [0, 1]


def test_wires_runtime():
    """Test that this device can execute a tape with wires determined at runtime if they are not provided."""
    dev = qp.device("default.tensor")
    ops = [qp.Identity(0), qp.Identity((0, 1)), qp.RX(2, 0), qp.RY(1, 5), qp.RX(2, 1)]
    measurements = [qp.expval(qp.PauliZ(15))]
    tape = qp.tape.QuantumScript(ops, measurements)
    assert dev.execute(tape) == 1.0


def test_wires_runtime_error():
    """Test that this device raises an error if the wires are provided by user and there is a mismatch."""
    dev = qp.device("default.tensor", wires=1)
    ops = [qp.Identity(0), qp.Identity((0, 1)), qp.RX(2, 0), qp.RY(1, 5), qp.RX(2, 1)]
    measurements = [qp.expval(qp.PauliZ(15))]
    tape = qp.tape.QuantumScript(ops, measurements)

    with pytest.raises(WireError):
        dev.execute(tape)


@pytest.mark.parametrize("max_bond_dim", [None, 10])
@pytest.mark.parametrize("cutoff", [1e-16, 1e-12])
def test_kwargs_mps(max_bond_dim, cutoff):
    """Test the class initialization with different arguments and returned properties for the MPS method."""

    max_bond_dim = 10
    cutoff = 1e-16
    method = "mps"

    dev = qp.device("default.tensor", method=method, max_bond_dim=max_bond_dim, cutoff=cutoff)

    config = dev.setup_execution_config()
    assert config.device_options["method"] == method
    assert config.device_options["max_bond_dim"] == max_bond_dim
    assert config.device_options["cutoff"] == cutoff
    assert config.device_options["contract"] == "auto-mps"


def test_kwargs_tn():
    """Test the class initialization with different arguments and returned properties for the TN method."""

    method = "tn"
    dev = qp.device("default.tensor", method=method)

    config = dev.setup_execution_config()
    assert config.device_options["method"] == method
    assert config.device_options["contract"] == "auto-split-gate"


def test_invalid_kwarg():
    """Test an invalid keyword argument."""
    with pytest.raises(
        TypeError,
        match="Unexpected argument: fake_arg during initialization of the default.tensor device.",
    ):
        qp.device("default.tensor", fake_arg=None)


def test_invalid_contract():
    """Test an invalid combination of method and contract."""

    with pytest.raises(
        ValueError, match="Unsupported gate contraction option: 'auto-split-gate' for 'mps' method."
    ):
        qp.device("default.tensor", method="mps", contract="auto-split-gate")

    with pytest.raises(
        ValueError, match="Unsupported gate contraction option: 'auto-mps' for 'tn' method."
    ):
        qp.device("default.tensor", method="tn", contract="auto-mps")


@pytest.mark.parametrize("method", ["mps", "tn"])
def test_method(method):
    """Test the device method."""
    assert qp.device("default.tensor", method=method).method == method


def test_invalid_method():
    """Test an invalid method."""
    method = "invalid_method"
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        qp.device("default.tensor", method=method)


@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
def test_data_type(c_dtype):
    """Test the data type."""
    assert qp.device("default.tensor", c_dtype=c_dtype).c_dtype == c_dtype


def test_ivalid_data_type():
    """Test that data type can only be np.complex64 or np.complex128."""
    with pytest.raises(TypeError):
        qp.device("default.tensor", c_dtype=float)


@pytest.mark.parametrize("method", ["mps", "tn"])
def test_draw(method):
    """Test the draw method."""

    dev = qp.device("default.tensor", wires=10, method=method)
    fig = dev.draw(color="auto", title="Test", return_fig=True)
    assert fig is not None


def test_warning_useless_kwargs():
    """Test that a warning is raised if the user provides a combination of arguments that are not used."""

    with pytest.warns():
        qp.device("default.tensor", method="tn", max_bond_dim=10)
        qp.device("default.tensor", method="tn", cutoff=1e-16)


def test_kahypar_warning_not_raised(recwarn):
    """Test that a warning is not raised if the user does not have kahypar installed when initializing the
    default.tensor device"""
    try:
        import kahypar  # pylint: disable=import-outside-toplevel, unused-import

        pytest.skip(reason="Test is for when kahypar is not installed")
    except ImportError:
        _ = qp.device("default.tensor", wires=1)
        assert len(recwarn) == 0


def test_passing_shots_None():
    """Test that passing shots=None on initialization works without error."""
    dev = qp.device("default.tensor", shots=None)
    assert dev.shots == qp.measurements.Shots(None)


def test_passing_finite_shots_error():
    """Test that an error is raised if finite shots are passed on initialization."""

    with pytest.raises(DeviceError, match=r"only supports analytic simulations"):
        qp.device("default.tensor", shots=10)


@pytest.mark.parametrize("method", ["mps", "tn"])
class TestSupportedGatesAndObservables:
    """Test that the DefaultTensor device supports all gates and observables that it claims to support."""

    # Note: we could potentially test each 'contract' option for both methods, but this would significantly
    # increase the number of tests. Furthermore, the 'contract' option is tested in the quimb library itself.

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, operation, method):
        """Test that the device can implement all its supported gates."""

        dev = qp.device("default.tensor", wires=4, method=method)

        tape = qp.tape.QuantumScript(
            [operations_list[operation]],
            [qp.expval(qp.Identity(wires=0))],
        )

        result = dev.execute(circuits=tape)
        assert np.allclose(result, 1.0)

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_yield_correct_state(self, operation, method):
        """Test that the device can implement all its supported gates."""
        nwires = 4
        dq = qp.device("default.qubit", wires=nwires)
        dev = qp.device("default.tensor", wires=nwires, method=method)

        state = np.random.rand(2**nwires) + 1j * np.random.rand(2**nwires)
        state /= np.linalg.norm(state)
        wires = qp.wires.Wires(range(nwires))
        tape = qp.tape.QuantumScript(
            [qp.StatePrep(state, wires=wires), operations_list[operation]],
            [qp.state()],
        )
        result = dev.execute(circuits=[tape])[0]
        ref = dq.execute(circuits=[tape])[0]
        assert np.allclose(result, ref)

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, observable, method):
        """Test that the device can implement all its supported observables."""

        dev = qp.device("default.tensor", wires=3, method=method)

        if observable == "Projector":
            for o in observables_list[observable]:
                tape = qp.tape.QuantumScript(
                    [qp.PauliX(0)],
                    [qp.expval(o)],
                )
                result = dev.execute(circuits=tape)
                assert isinstance(result, (float, np.ndarray))

        else:
            tape = qp.tape.QuantumScript(
                [qp.PauliX(0)],
                [qp.expval(observables_list[observable])],
            )
            result = dev.execute(circuits=tape)
            assert isinstance(result, (float, np.ndarray))

    def test_not_implemented_meas(self, method):
        """Tests that support only exists for `qp.expval` and `qp.var` so far."""

        op = [qp.Identity(0)]
        measurements = [qp.probs(qp.PauliZ(0))]
        tape = qp.tape.QuantumScript(op, measurements)

        dev = qp.device("default.tensor", wires=tape.wires, method=method)

        with pytest.raises(NotImplementedError):
            dev.execute(tape)


class TestSupportsDerivatives:
    """Test that DefaultTensor states what kind of derivatives it supports."""

    def test_support_derivatives(self):
        """Test that the device does not support derivatives yet."""
        dev = qp.device("default.tensor")
        assert not dev.supports_derivatives()

    def test_compute_derivatives(self):
        """Test that an error is raised if the `compute_derivatives` method is called."""
        dev = qp.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_derivatives(circuits=None)

    def test_execute_and_compute_derivatives(self):
        """Test that an error is raised if `execute_and_compute_derivative` method is called."""
        dev = qp.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.execute_and_compute_derivatives(circuits=None)

    def test_supports_vjp(self):
        """Test that the device does not support VJP yet."""
        dev = qp.device("default.tensor")
        assert not dev.supports_vjp()

    def test_compute_vjp(self):
        """Test that an error is raised if `compute_vjp` method is called."""
        dev = qp.device("default.tensor")
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_vjp(circuits=None, cotangents=None)

    def test_execute_and_compute_vjp(self):
        """Test that an error is raised if `execute_and_compute_vjp` method is called."""
        dev = qp.device("default.tensor")
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
        dev = qp.device("default.tensor", wires=1, method=method)
        ref_dev = qp.device("default.qubit", wires=1)

        def circuit(x):
            qp.RX(x[1], wires=0)
            qp.Rot(x[0], x[1], x[2], wires=0)
            return qp.expval(qp.Z(0))

        weights = jax.numpy.array([0.2, 0.5, 0.1])
        qnode = qp.QNode(circuit, dev, interface="jax")
        ref_qnode = qp.QNode(circuit, ref_dev, interface="jax")

        assert np.allclose(qnode(weights), ref_qnode(weights))

    def test_jax_jit(self, method):
        """Test the device with JAX's JIT compiler."""

        jax = pytest.importorskip("jax")
        dev = qp.device("default.tensor", wires=1, method=method)

        @jax.jit
        @qp.qnode(dev, interface="jax")
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        assert np.allclose(circuit(), 0.0)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("method", ["mps", "tn"])
@pytest.mark.parametrize(
    "operation, expected_output, par",
    [
        (qp.BasisState, [0, 0, 1 + 0j, 0], [1 + 0j, 0]),
        (qp.BasisState, [0, 0, 0, 1 + 0j], [1 + 0j, 1 + 0j]),
        (qp.BasisState, [0, 0, 1, 0], [1, 0]),
        (qp.BasisState, [0, 0, 0, 1], [1, 1]),
        (qp.StatePrep, [0, 0, 1 + 0j, 0], [0, 0, 1 + 0j, 0]),
        (qp.StatePrep, [0, 0, 0, 1 + 0j], [0, 0, 0, 1 + 0j]),
        (qp.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qp.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
        (
            qp.StatePrep,
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
        (
            qp.StatePrep,
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
    ],
)
def test_apply_operation_state_preparation(operation, expected_output, par, method):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""

    par = np.array(par)
    dev = qp.device("default.tensor", method=method, wires=2)

    @qp.qnode(dev)
    def circuit():
        operation(par, wires=[0, 1])
        return qp.state()

    state = circuit()
    assert np.allclose(state, np.array(expected_output), rtol=0)


# At this stage, this test is especially relevant for the MPS method, but we test both methods for consistency.
@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
@pytest.mark.parametrize("num_orbitals", [2, 4])
@pytest.mark.parametrize("method", ["mps", "tn"])
def test_wire_order_dense_vector(method, num_orbitals):
    """Test that the wire order is preserved if the initial state is created from a dense vector."""

    dev = qp.device("default.tensor", wires=int(2 * num_orbitals + 1), method=method)
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
            qp.SingleExcitation(2 * theta, wires=[int(wires[indices[0]]), int(wires[indices[1]])])

    control_wires = 1

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(wave_fun, wires=qubits[1:])
        qp.Hadamard(qubits[0])
        basis_rotation_ops(U0, [int(2 * i + 1 + control_wires) for i in range(num_orbitals)])
        return qp.state()

    state = circuit()
    assert isinstance(state, TensorLike)
    assert len(state) == 2 ** (2 * num_orbitals + 1)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
class TestMCMs:
    """Test that default.tensor can handle mid circuit measurements."""

    @pytest.mark.parametrize("mcm_method", ("one-shot", "tree-traversal"))
    def test_error_on_unsupported_mcm_method(self, mcm_method):
        """Test that an error is raised on unsupported mcm methods."""

        mcm_config = qp.devices.MCMConfig(mcm_method=mcm_method)
        config = qp.devices.ExecutionConfig(mcm_config=mcm_config)
        with pytest.raises(DeviceError, match=r"only supports the deferred measurement principle."):
            qp.device("default.tensor").preprocess(config)

    def test_simple_mcm_present(self):
        """Test that the device can execute a circuit with a mid circuit measurement."""

        dev = qp.device("default.tensor")

        @qp.qnode(dev)
        def circuit():
            qp.measure(0)
            return qp.expval(qp.Z(0))

        res = circuit()
        assert qp.math.allclose(res, 1)

    def test_mcm_conditional(self):
        """Test that the device execute a circuit with an MCM and a conditional."""

        dev = qp.device("default.tensor")

        @qp.qnode(dev)
        def circuit(x):
            m0 = qp.measure(0)
            qp.cond(~m0, qp.RX)(x, 0)
            return qp.expval(qp.Z(0))

        res = circuit(0.5)
        assert qp.math.allclose(res, np.cos(0.5))


class TestPreprocessingTransforms:
    """Tests for the preprocessing transform pipeline."""

    def test_preprocess_transforms_structure(self):
        """Test that the preprocessing transforms are set up correctly."""
        dev = qp.device("default.tensor", wires=3)
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Check that we have the expected transforms
        transform_names = [
            transform_container.tape_transform.__name__ for transform_container in program
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
        dev = qp.device("default.tensor", wires=[0, 1, 2])
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Find the decompose transform
        decompose_transform = None
        for transform_container in program:
            if transform_container.tape_transform.__name__ == "decompose":
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
        dev = qp.device("default.tensor", wires=3)
        config = ExecutionConfig()

        program, _ = dev.preprocess(config)

        # Find the decompose transform
        decompose_transform = None
        for transform_container in program:
            if transform_container.tape_transform.__name__ == "decompose":
                decompose_transform = transform_container
                break

        assert decompose_transform is not None
        assert "stopping_condition" in decompose_transform.kwargs
        assert decompose_transform.kwargs["stopping_condition"] == stopping_condition

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.integration
    def test_integration_with_qnode(self):
        """Test integration with QNode to ensure the device works end-to-end."""
        dev = qp.device("default.tensor", wires=3)

        @qp.qnode(dev)
        def circuit():
            # Use an operation that needs decomposition
            qp.QFT(wires=[0, 1])
            return qp.expval(qp.Z(0))

        # This should work without errors
        result = circuit()
        assert isinstance(result, (float, np.floating))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.integration
    def test_integration_with_multiple_decomposition_layers(self):
        """Test that operations requiring multiple layers of decomposition work."""
        dev = qp.device("default.tensor", wires=4)

        @qp.qnode(dev)
        def circuit():
            # Operations that may require multiple decomposition steps
            qp.QFT(wires=[0, 1, 2])
            qp.GroverOperator(wires=[0, 1, 2, 3])
            return qp.expval(qp.Z(0))

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
        class MyOp(qp.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 1

        # Fallback decomposition (no work wires needed)
        @qp.register_resources({qp.Hadamard: 2})
        def decomp_fallback(wires):
            qp.Hadamard(wires)
            qp.Hadamard(wires)

        # Work wire decomposition (needs more wires than available)
        @qp.register_resources({qp.PauliX: 1}, work_wires={"burnable": 3})
        def decomp_with_work_wire(wires):
            qp.PauliX(wires)

        qp.add_decomps(MyOp, decomp_fallback, decomp_with_work_wire)

        tape = qp.tape.QuantumScript([MyOp(0)], [qp.expval(qp.Z(0))])
        dev = qp.device("default.tensor", wires=wires)
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        assert len(out_tape.operations) == expected_program_len
