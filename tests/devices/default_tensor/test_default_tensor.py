# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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


import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pennylane as qml

quimb = pytest.importorskip("quimb")

pytestmark = pytest.mark.external

# gates for which device support is tested
operations_list = {
    "Identity": qml.Identity(wires=[0]),
    "BlockEncode": qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
    "CNOT": qml.CNOT(wires=[0, 1]),
    "CRX": qml.CRX(0, wires=[0, 1]),
    "CRY": qml.CRY(0, wires=[0, 1]),
    "CRZ": qml.CRZ(0, wires=[0, 1]),
    "CRot": qml.CRot(0, 0, 0, wires=[0, 1]),
    "CSWAP": qml.CSWAP(wires=[0, 1, 2]),
    "CZ": qml.CZ(wires=[0, 1]),
    "CCZ": qml.CCZ(wires=[0, 1, 2]),
    "CY": qml.CY(wires=[0, 1]),
    "CH": qml.CH(wires=[0, 1]),
    "DiagonalQubitUnitary": qml.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "MultiRZ": qml.MultiRZ(0, wires=[0]),
    "PauliX": qml.X(0),
    "PauliY": qml.Y(0),
    "PauliZ": qml.Z(0),
    "X": qml.X([0]),
    "Y": qml.Y([0]),
    "Z": qml.Z([0]),
    "PhaseShift": qml.PhaseShift(0, wires=[0]),
    "PCPhase": qml.PCPhase(0, 1, wires=[0, 1]),
    "ControlledPhaseShift": qml.ControlledPhaseShift(0, wires=[0, 1]),
    "CPhaseShift00": qml.CPhaseShift00(0, wires=[0, 1]),
    "CPhaseShift01": qml.CPhaseShift01(0, wires=[0, 1]),
    "CPhaseShift10": qml.CPhaseShift10(0, wires=[0, 1]),
    "QubitUnitary": qml.QubitUnitary(np.eye(2), wires=[0]),
    "SpecialUnitary": qml.SpecialUnitary(np.array([0.2, -0.1, 2.3]), wires=1),
    "ControlledQubitUnitary": qml.ControlledQubitUnitary(np.eye(2), control_wires=[1], wires=[0]),
    "MultiControlledX": qml.MultiControlledX(wires=[1, 2, 0]),
    "IntegerComparator": qml.IntegerComparator(1, geq=True, wires=[0, 1, 2]),
    "RX": qml.RX(0, wires=[0]),
    "RY": qml.RY(0, wires=[0]),
    "RZ": qml.RZ(0, wires=[0]),
    "Rot": qml.Rot(0, 0, 0, wires=[0]),
    "S": qml.S(wires=[0]),
    "Adjoint(S)": qml.adjoint(qml.S(wires=[0])),
    "SWAP": qml.SWAP(wires=[0, 1]),
    "ISWAP": qml.ISWAP(wires=[0, 1]),
    "PSWAP": qml.PSWAP(0, wires=[0, 1]),
    "ECR": qml.ECR(wires=[0, 1]),
    "Adjoint(ISWAP)": qml.adjoint(qml.ISWAP(wires=[0, 1])),
    "T": qml.T(wires=[0]),
    "Adjoint(T)": qml.adjoint(qml.T(wires=[0])),
    "SX": qml.SX(wires=[0]),
    "Adjoint(SX)": qml.adjoint(qml.SX(wires=[0])),
    "Toffoli": qml.Toffoli(wires=[0, 1, 2]),
    "QFT": qml.templates.QFT(wires=[0, 1, 2]),
    "IsingXX": qml.IsingXX(0, wires=[0, 1]),
    "IsingYY": qml.IsingYY(0, wires=[0, 1]),
    "IsingZZ": qml.IsingZZ(0, wires=[0, 1]),
    "IsingXY": qml.IsingXY(0, wires=[0, 1]),
    "SingleExcitation": qml.SingleExcitation(0, wires=[0, 1]),
    "SingleExcitationPlus": qml.SingleExcitationPlus(0, wires=[0, 1]),
    "SingleExcitationMinus": qml.SingleExcitationMinus(0, wires=[0, 1]),
    "DoubleExcitation": qml.DoubleExcitation(0, wires=[0, 1, 2, 3]),
    "QubitCarry": qml.QubitCarry(wires=[0, 1, 2, 3]),
    "QubitSum": qml.QubitSum(wires=[0, 1, 2]),
    "PauliRot": qml.PauliRot(0, "XXYY", wires=[0, 1, 2, 3]),
    "U1": qml.U1(0, wires=0),
    "U2": qml.U2(0, 0, wires=0),
    "U3": qml.U3(0, 0, 0, wires=0),
    "SISWAP": qml.SISWAP(wires=[0, 1]),
    "Adjoint(SISWAP)": qml.adjoint(qml.SISWAP(wires=[0, 1])),
    "OrbitalRotation": qml.OrbitalRotation(0, wires=[0, 1, 2, 3]),
    "FermionicSWAP": qml.FermionicSWAP(0, wires=[0, 1]),
    "GlobalPhase": qml.GlobalPhase(0.123, wires=[0, 1]),
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
    assert qml.device("default.tensor", wires=0).name == "default.tensor"


def test_wires():
    """Test that a device can be created with wires."""
    assert qml.device("default.tensor", wires=0).wires is not None
    assert qml.device("default.tensor", wires=2).wires == qml.wires.Wires([0, 1])
    assert qml.device("default.tensor", wires=[0, 2]).wires == qml.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        qml.device("default.tensor", wires=0).wires = [0, 1]


def test_wires_error():
    """Test that an error is raised if the wires are not provided."""
    with pytest.raises(TypeError):
        qml.device("default.tensor")


def test_wires_execution_error():
    """Test that this device cannot execute a tape if its wires do not match the wires on the device."""
    dev = qml.device("default.tensor", wires=3)
    ops = [
        qml.Identity(0),
        qml.Identity((0, 1)),
        qml.RX(2, 0),
        qml.RY(1, 5),
        qml.RX(2, 1),
    ]
    measurements = [qml.expval(qml.PauliZ(15))]
    tape = qml.tape.QuantumScript(ops, measurements)

    with pytest.raises(AttributeError):
        dev.execute(tape)


@pytest.mark.parametrize("max_bond_dim", [None, 10])
@pytest.mark.parametrize("cutoff", [1e-16, 1e-12])
@pytest.mark.parametrize("contract", ["auto-mps", "nonlocal"])
def test_kwargs(max_bond_dim, cutoff, contract):
    """Test the class initialization with different arguments and returned properties."""

    kwargs = {"max_bond_dim": max_bond_dim, "cutoff": cutoff, "contract": contract}

    dev = qml.device("default.tensor", wires=0, **kwargs)

    _, config = dev.preprocess()
    assert config.device_options["method"] == "mps"
    assert config.device_options["max_bond_dim"] == max_bond_dim
    assert config.device_options["cutoff"] == cutoff
    assert config.device_options["contract"] == contract


def test_invalid_kwarg():
    """Test an invalid keyword argument."""
    with pytest.raises(
        TypeError,
        match="Unexpected argument: fake_arg during initialization of the default.tensor device.",
    ):
        qml.device("default.tensor", wires=0, fake_arg=None)


def test_method():
    """Test the device method."""
    assert qml.device("default.tensor", wires=0).method == "mps"


def test_invalid_method():
    """Test an invalid method."""
    method = "invalid_method"
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        qml.device("default.tensor", wires=0, method=method)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_data_type(dtype):
    """Test the data type."""
    assert qml.device("default.tensor", wires=0, dtype=dtype).dtype == dtype


def test_ivalid_data_type():
    """Test that data type can only be np.complex64 or np.complex128."""
    with pytest.raises(TypeError):
        qml.device("default.tensor", wires=0, dtype=float)


def test_invalid_shots():
    """Test that an error is raised if finite number of shots are requestd."""
    with pytest.raises(ValueError, match="default.tensor does not support finite shots."):
        qml.device("default.tensor", wires=0, shots=5)

    with pytest.raises(AttributeError):
        qml.device("default.tensor", wires=0).shots = 10


class TestSupportedGatesAndObservables:
    """Test that the DefaultTensor device supports all gates and observables that it claims to support."""

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, operation):
        """Test that the device can implement all its supported gates."""

        dev = qml.device("default.tensor", wires=4, method="mps")

        tape = qml.tape.QuantumScript(
            [operations_list[operation]],
            [qml.expval(qml.Identity(wires=0))],
        )

        result = dev.execute(circuits=tape)
        assert np.allclose(result, 1.0)

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, observable):
        """Test that the device can implement all its supported observables."""

        dev = qml.device("default.tensor", wires=3, method="mps")

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

    def test_not_implemented_meas(self):
        """Tests that support only exists for `qml.expval` and `qml.var` so far."""

        op = [qml.Identity(0)]
        measurements = [qml.probs(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(op, measurements)

        dev = qml.device("default.tensor", wires=tape.wires)

        with pytest.raises(NotImplementedError):
            dev.execute(tape)


class TestSupportsDerivatives:
    """Test that DefaultTensor states what kind of derivatives it supports."""

    def test_support_derivatives(self):
        """Test that the device does not support derivatives yet."""
        dev = qml.device("default.tensor", wires=0)
        assert not dev.supports_derivatives()

    def test_compute_derivatives(self):
        """Test that an error is raised if the `compute_derivatives` method is called."""
        dev = qml.device("default.tensor", wires=0)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_derivatives(circuits=None)

    def test_execute_and_compute_derivatives(self):
        """Test that an error is raised if `execute_and_compute_derivative` method is called."""
        dev = qml.device("default.tensor", wires=0)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the default.tensor device.",
        ):
            dev.execute_and_compute_derivatives(circuits=None)

    def test_supports_vjp(self):
        """Test that the device does not support VJP yet."""
        dev = qml.device("default.tensor", wires=0)
        assert not dev.supports_vjp()

    def test_compute_vjp(self):
        """Test that an error is raised if `compute_vjp` method is called."""
        dev = qml.device("default.tensor", wires=0)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
        ):
            dev.compute_vjp(circuits=None, cotangents=None)

    def test_execute_and_compute_vjp(self):
        """Test that an error is raised if `execute_and_compute_vjp` method is called."""
        dev = qml.device("default.tensor", wires=0)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
        ):
            dev.execute_and_compute_vjp(circuits=None, cotangents=None)

    @pytest.mark.jax
    def test_jax(self):
        """Test the device with JAX."""

        jax = pytest.importorskip("jax")
        dev = qml.device("default.tensor", wires=1)
        ref_dev = qml.device("default.qubit.jax", wires=1)

        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.Z(0))

        weights = jax.numpy.array([0.2, 0.5, 0.1])
        print(isinstance(dev, qml.Device))
        qnode = qml.QNode(circuit, dev, interface="jax")
        ref_qnode = qml.QNode(circuit, ref_dev, interface="jax")

        assert np.allclose(qnode(weights), ref_qnode(weights))

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test the device with JAX's JIT compiler."""

        jax = pytest.importorskip("jax")
        dev = qml.device("default.tensor", wires=1)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        assert np.allclose(circuit(), 0.0)
