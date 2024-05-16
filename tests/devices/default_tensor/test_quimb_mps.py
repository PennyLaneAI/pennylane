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
Unit tests for the ``quimb`` interface.
"""

import itertools
import math

import numpy as np
import pennylane as qml
import pytest
import quimb.tensor as qtn
from conftest import LightningDevice  # tested device
from pennylane import QNode
from pennylane.devices import DefaultQubit
from pennylane.wires import Wires
from scipy.sparse import csr_matrix

from pennylane_lightning.lightning_tensor import LightningTensor

if not LightningDevice._new_API:
    pytest.skip("Exclusive tests for new API. Skipping.", allow_module_level=True)

if LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("Device doesn't have C++ support yet.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)

# gates for which interface support is tested
ops = {
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

all_ops = ops.keys()

# observables for which interface support is tested
obs = {
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

all_obs = obs.keys()


@pytest.mark.parametrize("backend", ["quimb"])
@pytest.mark.parametrize("method", ["mps"])
class TestQuimbMPS:
    """Tests for the MPS method."""

    @pytest.mark.parametrize("num_wires", [None, 4])
    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("max_bond_dim", [None, 10])
    @pytest.mark.parametrize("cutoff", [1e-16, 1e-12])
    @pytest.mark.parametrize("contract", ["auto-mps", "nonlocal"])
    def test_device_init(self, num_wires, c_dtype, backend, method, max_bond_dim, cutoff, contract):
        """Test the class initialization with different arguments and returned properties."""

        kwargs = {"max_bond_dim": max_bond_dim, "cutoff": cutoff, "contract": contract}

        wires = Wires(range(num_wires)) if num_wires else None
        dev = LightningTensor(
            wires=wires, backend=backend, method=method, c_dtype=c_dtype, **kwargs
        )
        assert isinstance(dev._interface.state, qtn.MatrixProductState)
        assert isinstance(dev._interface.state_to_array(), np.ndarray)

        _, config = dev.preprocess()
        assert config.device_options["c_dtype"] == c_dtype
        assert config.device_options["backend"] == backend
        assert config.device_options["method"] == method
        assert config.device_options["max_bond_dim"] == max_bond_dim
        assert config.device_options["cutoff"] == cutoff
        assert config.device_options["contract"] == contract

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, operation, backend, method):
        """Test that the interface can implement all its supported gates."""

        dev = LightningTensor(
            wires=Wires(range(4)), backend=backend, method=method, c_dtype=np.complex64
        )

        tape = qml.tape.QuantumScript(
            [ops[operation]],
            [qml.expval(qml.Identity(wires=0))],
        )

        result = dev.execute(circuits=tape)
        assert np.allclose(result, 1.0)

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, observable, backend, method):
        """Test that the interface can implement all its supported observables."""
        dev = LightningTensor(
            wires=Wires(range(3)), backend=backend, method=method, c_dtype=np.complex64
        )

        if observable == "Projector":
            for o in obs[observable]:
                tape = qml.tape.QuantumScript(
                    [qml.PauliX(0)],
                    [qml.expval(o)],
                )
                result = dev.execute(circuits=tape)
                assert isinstance(result, (float, np.ndarray))

        else:

            tape = qml.tape.QuantumScript(
                [qml.PauliX(0)],
                [qml.expval(obs[observable])],
            )
            result = dev.execute(circuits=tape)
            assert isinstance(result, (float, np.ndarray))

    def test_not_implemented_meas(self, backend, method):
        """Tests that support only exists for `qml.expval` and `qml.var` so far."""

        ops = [qml.Identity(0)]
        measurements = [qml.probs(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        dev = LightningTensor(
            wires=tape.wires, backend=backend, method=method, c_dtype=np.complex64
        )

        with pytest.raises(NotImplementedError):
            dev.execute(tape)

    def test_not_implemented_shots(self, backend, method):
        """Tests that this interface does not support measurements with finite shots."""

        ops = [qml.Identity(0)]
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)
        tape._shots = 5

        dev = LightningTensor(
            wires=tape.wires, backend=backend, method=method, c_dtype=np.complex64
        )

        with pytest.raises(NotImplementedError):
            dev.execute(tape)

    def test_interface_jax(self, backend, method):
        """Test the interface with JAX."""

        jax = pytest.importorskip("jax")
        dev = LightningTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)
        ref_dev = qml.device("default.qubit.jax", wires=1)

        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.Z(0))

        weights = jax.numpy.array([0.2, 0.5, 0.1])
        qnode = QNode(circuit, dev, interface="jax")
        ref_qnode = QNode(circuit, ref_dev, interface="jax")

        assert np.allclose(qnode(weights), ref_qnode(weights))

    def test_interface_jax_jit(self, backend, method):
        """Test the interface with JAX's JIT compiler."""

        jax = pytest.importorskip("jax")
        dev = LightningTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        assert np.allclose(circuit(), 0.0)

    def test_(self, backend, method):
        """..."""

        # jax = pytest.importorskip("jax")
        dev = LightningTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)

        def circuit():
            qml.RX(0.0, wires=0)

        with pytest.raises(qml.QuantumFunctionError):
            QNode(circuit, dev, interface="jax", diff_method="adjoint")
