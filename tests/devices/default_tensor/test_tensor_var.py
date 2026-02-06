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
Tests for the variance calculation on the DefaultTensor device.
"""
import numpy as np
import pytest

import pennylane as qp
from pennylane.devices import DefaultQubit

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)

quimb = pytest.importorskip("quimb")

pytestmark = pytest.mark.external

# pylint: disable=too-many-arguments, redefined-outer-name


@pytest.fixture(
    params=[
        (np.complex64, "mps"),
        (np.complex64, "tn"),
        (np.complex128, "mps"),
        (np.complex128, "tn"),
    ]
)
def dev(request):
    """Device fixture."""
    c_dtype, method = request.param
    return qp.device("default.tensor", wires=3, method=method, c_dtype=c_dtype)


def calculate_reference(tape):
    """Calculate the reference value of the tape using DefaultQubit."""
    ref_dev = DefaultQubit(max_workers=1)
    program = ref_dev.preprocess_transforms()
    tapes, transf_fn = program([tape])
    results = ref_dev.execute(tapes)
    return transf_fn(results)


def execute(device, tape):
    """Execute the tape on the device and return the result."""
    results = device.execute(tape)
    return results


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestVar:
    """Tests for the variance"""

    def test_Identity(self, theta, phi, dev):
        """Tests applying identities."""

        ops = [
            qp.Identity(0),
            qp.Identity((0, 1)),
            qp.RX(theta, 0),
            qp.Identity((1, 2)),
            qp.RX(phi, 1),
        ]
        measurements = [qp.var(qp.PauliZ(0))]
        tape = qp.tape.QuantumScript(ops, measurements)

        result = dev.execute(tape)
        expected = 1 - np.cos(theta) ** 2
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_identity_variance(self, theta, phi, dev):
        """Tests identity variances."""

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(qp.Identity(wires=[0])), qp.var(qp.Identity(wires=[1]))],
        )
        result = execute(dev, tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    def test_multi_wire_identity_variance(self, theta, phi, dev):
        """Tests multi-wire identity."""

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, wires=[0]), qp.RX(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(qp.Identity(wires=[0, 1]))],
        )
        result = execute(dev, tape)
        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(0.0, result, atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [([0, 1]), (["a", 1]), (["b", "a"]), ([-1, 2.5])])
    @pytest.mark.parametrize("method", ["mps", "tn"])
    def test_custom_wires(self, theta, phi, wires, method):
        """Tests custom wires."""
        device = qp.device("default.tensor", wires=wires, method=method)

        tape = qp.tape.QuantumScript(
            [
                qp.RX(theta, wires=wires[0]),
                qp.RX(phi, wires=wires[1]),
                qp.CNOT(wires=wires),
            ],
            [qp.var(qp.PauliZ(wires=wires[0])), qp.var(qp.PauliZ(wires=wires[1]))],
        )

        calculated_val = execute(device, tape)
        reference_val = np.array(
            [1 - np.cos(theta) ** 2, 1 - np.cos(theta) ** 2 * np.cos(phi) ** 2]
        )

        tol = 1e-5 if device.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "Obs, Op, expected_fn",
        [
            (
                [qp.PauliX(wires=[0]), qp.PauliX(wires=[1])],
                qp.RY,
                lambda theta, phi: np.array(
                    [1 - np.sin(theta) ** 2 * np.sin(phi) ** 2, 1 - np.sin(phi) ** 2]
                ),
            ),
            (
                [qp.PauliY(wires=[0]), qp.PauliY(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array([1, 1 - np.cos(theta) ** 2 * np.sin(phi) ** 2]),
            ),
            (
                [qp.PauliZ(wires=[0]), qp.PauliZ(wires=[1])],
                qp.RX,
                lambda theta, phi: np.array(
                    [1 - np.cos(theta) ** 2, 1 - np.cos(theta) ** 2 * np.cos(phi) ** 2]
                ),
            ),
            (
                [qp.Hadamard(wires=[0]), qp.Hadamard(wires=[1])],
                qp.RY,
                lambda theta, phi: np.array(
                    [
                        1 - (np.sin(theta) * np.sin(phi) + np.cos(theta)) ** 2 / 2,
                        1 - (np.cos(theta) * np.cos(phi) + np.sin(phi)) ** 2 / 2,
                    ]
                ),
            ),
        ],
    )
    def test_single_wire_observables_variance(self, Obs, Op, expected_fn, theta, phi, dev):
        """Test that variance values for single wire observables are correct"""

        tape = qp.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qp.CNOT(wires=[0, 1])],
            [qp.var(Obs[0]), qp.var(Obs[1])],
        )
        result = execute(dev, tape)
        expected = expected_fn(theta, phi)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(result, expected, atol=tol, rtol=0)

    def test_hermitian_variance(self, theta, phi, dev):
        """Tests an Hermitian operator."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)
            qp.RX(theta + phi, wires=2)

            for idx in range(3):
                qp.var(qp.Hermitian([[1, 0], [0, -1]], wires=[idx]))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_hamiltonian_variance(self, theta, phi, dev):
        """Tests a Hamiltonian."""

        ham = qp.Hamiltonian(
            [1.0, 0.3, 0.3, 0.4],
            [
                qp.PauliX(0) @ qp.PauliX(1),
                qp.PauliZ(0),
                qp.PauliZ(1),
                qp.PauliX(0) @ qp.PauliY(1),
            ],
        )

        with qp.tape.QuantumTape() as tape1:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)
            qp.RX(theta + phi, wires=2)

            qp.var(ham)

        tape2 = qp.tape.QuantumScript(tape1.operations, [qp.var(qp.dot(ham.coeffs, ham.ops))])

        calculated_val = execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)

    def test_sparse_hamiltonian_variance(self, theta, phi, dev):
        """Tests a Hamiltonian."""

        ham = qp.SparseHamiltonian(
            qp.Hamiltonian(
                [1.0, 0.3, 0.3, 0.4],
                [
                    qp.PauliX(0) @ qp.PauliX(1),
                    qp.PauliZ(0),
                    qp.PauliZ(1),
                    qp.PauliX(0) @ qp.PauliY(1),
                ],
            ).sparse_matrix(),
            wires=[0, 1],
        )

        with qp.tape.QuantumTape() as tape1:
            qp.RX(theta, wires=0)
            qp.RX(phi, wires=1)

            qp.var(ham)

        tape2 = qp.tape.QuantumScript(
            tape1.operations, [qp.var(qp.Hermitian(ham.matrix(), wires=[0, 1]))]
        )

        calculated_val = execute(dev, tape1)
        reference_val = calculate_reference(tape2)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(calculated_val, reference_val, atol=tol, rtol=0)


@pytest.mark.parametrize("phi", PHI)
class TestOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    @pytest.mark.parametrize(
        "obs",
        [
            qp.s_prod(0.5, qp.PauliZ(0)),
            qp.prod(qp.PauliZ(0), qp.PauliX(1)),
            qp.sum(qp.PauliZ(0), qp.PauliX(1)),
        ],
    )
    def test_op_math(self, phi, dev, obs):
        """Tests the `SProd`, `Prod`, and `Sum` classes."""

        tape = qp.tape.QuantumScript(
            [
                qp.RX(phi, wires=[0]),
                qp.Hadamard(wires=[1]),
                qp.PauliZ(wires=[1]),
                qp.RX(-1.1 * phi, wires=[1]),
            ],
            [qp.var(obs)],
        )

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_integration(self, phi, dev):
        """Test a Combination of `Sum`, `SProd`, and `Prod`."""

        obs = qp.sum(
            qp.s_prod(2.3, qp.PauliZ(0)),
            -0.5 * qp.prod(qp.PauliY(0), qp.PauliZ(1)),
        )

        tape = qp.tape.QuantumScript(
            [qp.RX(phi, wires=[0]), qp.RX(-1.1 * phi, wires=[0])],
            [qp.var(obs)],
        )

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Test tensor variances"""

    def test_PauliX_PauliY(self, theta, phi, varphi, dev):
        """Tests a tensor product involving PauliX and PauliY."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliX(0) @ qp.PauliY(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_identity(self, theta, phi, varphi, dev):
        """Tests a tensor product involving PauliZ and Identity."""

        with qp.tape.QuantumTape() as tape:
            qp.Identity(wires=[0])
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliZ(0) @ qp.Identity(1) @ qp.PauliZ(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)

    def test_PauliZ_hadamard_PauliY(self, theta, phi, varphi, dev):
        """Tests a tensor product involving PauliY, PauliZ and Hadamard."""

        with qp.tape.QuantumTape() as tape:
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.var(qp.PauliZ(0) @ qp.Hadamard(1) @ qp.PauliY(2))

        calculated_val = execute(dev, tape)
        reference_val = calculate_reference(tape)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7

        assert np.allclose(calculated_val, reference_val, tol)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
@pytest.mark.parametrize("method", ["mps", "tn"])
def test_multi_qubit_gates(theta, phi, method):
    """Tests a simple circuit with multi-qubit gates."""

    ops = [
        qp.PauliX(wires=[0]),
        qp.RX(theta, wires=[0]),
        qp.CSWAP(wires=[7, 0, 5]),
        qp.RX(phi, wires=[1]),
        qp.CNOT(wires=[3, 4]),
        qp.DoubleExcitation(phi, wires=[1, 2, 3, 4]),
        qp.CZ(wires=[4, 5]),
        qp.Hadamard(wires=[4]),
        qp.CCZ(wires=[0, 1, 2]),
        qp.CSWAP(wires=[2, 3, 4]),
        qp.QFT(wires=[0, 1, 2]),
        qp.CNOT(wires=[2, 4]),
        qp.Toffoli(wires=[0, 1, 2]),
        qp.DoubleExcitation(phi, wires=[0, 1, 3, 4]),
    ]

    meas = [
        qp.var(qp.PauliY(wires=[2])),
        qp.var(qp.Hamiltonian([1, 5, 6], [qp.Z(6), qp.X(0), qp.Hadamard(4)])),
        qp.var(
            qp.Hamiltonian(
                [4, 5, 7],
                [
                    qp.Z(6) @ qp.Y(4),
                    qp.X(0),
                    qp.Hadamard(7),
                ],
            )
        ),
    ]

    tape = qp.tape.QuantumScript(ops=ops, measurements=meas)

    reference_val = calculate_reference(tape)
    device = qp.device("default.tensor", method=method)
    calculated_val = device.execute(tape)

    assert np.allclose(calculated_val, reference_val)
