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
"""Tests that the different measurement types work correctly on a device."""
# pylint: disable=no-self-use,pointless-statement, no-member
import pytest
from flaky import flaky
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.skip_unsupported

# ==========================================================
# Some useful global variables

# observables for which device support is tested
obs = {
    "Identity": qml.Identity(wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "Hermitian": qml.Hermitian(np.eye(2), wires=[0]),
    "PauliX": qml.PauliX(wires=[0]),
    "PauliY": qml.PauliY(wires=[0]),
    "PauliZ": qml.PauliZ(wires=[0]),
    "Projector": qml.Projector(np.array([1]), wires=[0]),
    "SparseHamiltonian": qml.SparseHamiltonian(csr_matrix(np.eye(8)), wires=[0, 1, 2]),
    "Hamiltonian": qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(0)]),
}

all_obs = obs.keys()

# All qubit observables should be available to test in the device test suite
all_available_obs = qml.ops._qubit__obs__.copy()  # pylint: disable=protected-access
# Note that the identity is not technically a qubit observable
all_available_obs |= {"Identity"}

if not set(all_obs) == all_available_obs:
    raise ValueError(
        "A qubit observable has been added that is not being tested in the "
        "device test suite. Please add to the obs dictionary in "
        "pennylane/devices/tests/test_measurements.py"
    )

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

obs_lst = [
    qml.PauliX(wires=0) @ qml.PauliY(wires=1),
    qml.PauliX(wires=1) @ qml.PauliY(wires=0),
    qml.PauliX(wires=1) @ qml.PauliZ(wires=2),
    qml.PauliX(wires=2) @ qml.PauliZ(wires=1),
    qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.PauliZ(wires=2),
    qml.PauliZ(wires=0) @ qml.PauliX(wires=1) @ qml.PauliY(wires=2),
]

obs_permuted_lst = [
    qml.PauliY(wires=1) @ qml.PauliX(wires=0),
    qml.PauliY(wires=0) @ qml.PauliX(wires=1),
    qml.PauliZ(wires=2) @ qml.PauliX(wires=1),
    qml.PauliZ(wires=1) @ qml.PauliX(wires=2),
    qml.PauliZ(wires=2) @ qml.Identity(wires=0) @ qml.Identity(wires=1),
    qml.PauliX(wires=1) @ qml.PauliY(wires=2) @ qml.PauliZ(wires=0),
]

label_maps = [[0, 1, 2], ["a", "b", "c"], ["beta", "alpha", "gamma"], [3, "beta", "a"]]


def sub_routine(label_map):
    """Quantum function to initalize state in tests"""
    qml.Hadamard(wires=label_map[0])
    qml.RX(0.12, wires=label_map[1])
    qml.RY(3.45, wires=label_map[2])


class TestSupportedObservables:
    """Test that the device can implement all observables that it supports."""

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, device_kwargs, observable):
        """Test that the device can implement all its supported observables."""
        device_kwargs["wires"] = 3
        dev = qml.device(**device_kwargs)

        if device_kwargs.get("shots", None) is not None and observable == "SparseHamiltonian":
            pytest.skip("SparseHamiltonian only supported in analytic mode")

        assert hasattr(dev, "observables")
        if observable in dev.observables:

            kwargs = {"diff_method": "parameter-shift"} if observable == "SparseHamiltonian" else {}

            @qml.qnode(dev, **kwargs)
            def circuit():
                if dev.supports_operation(qml.PauliX):  # ionq can't have empty circuits
                    qml.PauliX(0)
                return qml.expval(obs[observable])

            assert isinstance(circuit(), (float, np.ndarray))

    def test_tensor_observables_can_be_implemented(self, device_kwargs):
        """Test that the device can implement a simple tensor observable.
        This test is skipped for devices that do not support tensor observables."""
        device_kwargs["wires"] = 2
        dev = qml.device(**device_kwargs)
        supports_tensor = (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )
        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

        @qml.qnode(dev)
        def circuit():
            if dev.supports_operation(qml.PauliX):  # ionq can't have empty circuits
                qml.PauliX(0)
            return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))

        assert isinstance(circuit(), (float, np.ndarray))


# pylint: disable=too-few-public-methods
@flaky(max_runs=10)
class TestHamiltonianSupport:
    """Separate test to ensure that the device can differentiate Hamiltonian observables."""

    def test_hamiltonian_diff(self, device_kwargs, tol):
        """Tests a simple VQE gradient using parameter-shift rules."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        coeffs = np.array([-0.05, 0.17])
        param = np.array(1.7, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)],))

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        def circuit1(param):
            """First Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliX(0))

        def circuit2(param):
            """Second Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliZ(0))

        half1 = qml.QNode(circuit1, dev, diff_method="parameter-shift")
        half2 = qml.QNode(circuit2, dev, diff_method="parameter-shift")

        def combine(coeffs, param):
            return coeffs[0] * half1(param) + coeffs[1] * half2(param)

        grad_fn_expected = qml.grad(combine)
        grad_expected = grad_fn_expected(coeffs, param)

        assert np.allclose(grad[0], grad_expected[0], atol=tol(dev.shots))
        assert np.allclose(grad[1], grad_expected[1], atol=tol(dev.shots))


@flaky(max_runs=10)
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, tol):
        """Test that identity expectation value (i.e. the trace) is 1."""
        n_wires = 2
        dev = device(n_wires)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(wires=0)), qml.expval(qml.Identity(wires=1))

        res = circuit()
        assert np.allclose(res, np.array([1, 1]), atol=tol(dev.shots))

    def test_pauliz_expectation(self, device, tol):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        res = circuit()
        assert np.allclose(
            res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), atol=tol(dev.shots)
        )

    def test_paulix_expectation(self, device, tol):
        """Test that PauliX expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))

        res = circuit()
        expected = np.array([np.sin(theta) * np.sin(phi), np.sin(phi)])
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_pauliy_expectation(self, device, tol):
        """Test that PauliY expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=0)), qml.expval(qml.PauliY(wires=1))

        res = circuit()
        expected = np.array([0.0, -np.cos(theta) * np.sin(phi)])
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_hadamard_expectation(self, device, tol):
        """Test that Hadamard expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hadamard(wires=0)), qml.expval(qml.Hadamard(wires=1))

        res = circuit()
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_hermitian_expectation(self, device, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A, wires=0)), qml.expval(qml.Hermitian(A, wires=1))

        res = circuit()

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_projector_expectation(self, device, tol):
        """Test that arbitrary Projector expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.732
        phi = 0.523

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector(basis_state, wires=[0, 1]))

        res = circuit([0, 0])
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([0, 1])
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 0])
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 1])
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_multi_mode_hermitian_expectation(self, device, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        theta = 0.432
        phi = 0.123
        A_ = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A_, wires=[0, 1]))

        res = circuit()

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(wires=0) @ qml.PauliY(wires=2))

        res = circuit()

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=0) @ qml.Hadamard(wires=1) @ qml.PauliY(wires=2))

        res = circuit()

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "base_obs, permuted_obs", list(zip(obs_lst, obs_permuted_lst)),
    )
    def test_wire_order_in_tensor_prod_observables(
        self, device, base_obs, permuted_obs, tol, skip_if
    ):
        """Test that when given a tensor observable the expectation value is the same regardless of the order of terms
        in the tensor observable, provided the wires each term acts on remain constant.

        eg:
        ob1 = qml.PauliZ(wires=0) @ qml.PauliY(wires=1)
        ob2 = qml.PauliY(wires=1) @ qml.PauliZ(wires=0)

        @qml.qnode(dev)
        def circ(obs):
            return qml.expval(obs)

        circ(ob1) == circ(ob2)
        """
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        @qml.qnode(dev)
        def circ(ob):
            sub_routine(label_map=range(3))
            return qml.expval(ob)

        assert np.allclose(circ(base_obs), circ(permuted_obs), atol=tol(dev.shots), rtol=0)

    @pytest.mark.parametrize("label_map", label_maps)
    def test_wire_label_in_tensor_prod_observables(self, device, label_map, tol, skip_if):
        """Test that when given a tensor observable the expectation value is the same regardless of how the
        wires are labelled, as long as they match the device order.

        For example:

        dev1 = qml.device("default.qubit", wires=[0, 1, 2])
        dev2 = qml.device("default.qubit", wires=['c', 'b', 'a']

        def circ(wire_labels):
            return qml.expval(qml.PauliZ(wires=wire_labels[0]) @ qml.PauliX(wires=wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
        dev = device(wires=3)
        dev_custom_labels = device(wires=label_map)
        skip_if(dev, {"supports_tensor_observables": False})

        def circ(wire_labels):
            sub_routine(wire_labels)
            return qml.expval(
                qml.PauliX(wire_labels[0]) @ qml.PauliY(wire_labels[1]) @ qml.PauliZ(wire_labels[2])
            )

        circ_base_label = qml.QNode(circ, device=dev)
        circ_custom_label = qml.QNode(circ, device=dev_custom_labels)

        assert np.allclose(
            circ_base_label(wire_labels=range(3)),
            circ_custom_label(wire_labels=label_map),
            atol=tol(dev.shots),
            rtol=0,
        )

    def test_hermitian(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543
        A_ = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=0) @ qml.Hermitian(A_, wires=[1, 2]))

        res = circuit()

        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_projector(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.732
        phi = 0.523
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Projector(basis_state, wires=[1, 2]))

        res = circuit([0, 0])
        expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([0, 1])
        expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 0])
        expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 1])
        expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_sparse_hamiltonian_expval(self, device, tol):
        """Test that expectation values of sparse Hamiltonians are properly calculated."""
        n_wires = 4
        dev = device(n_wires)

        if "SparseHamiltonian" not in dev.observables:
            pytest.skip("Skipped because device does not support the SparseHamiltonian observable.")
        if dev.shots is not None:
            pytest.skip("SparseHamiltonian only supported in analytic mode")

        h_row = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        h_col = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        h_data = np.array(
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1], dtype=np.complex128
        )
        h = csr_matrix((h_data, (h_row, h_col)), shape=(16, 16))  # XXYY

        @qml.qnode(dev, diff_method="parameter-shift")
        def result():
            qml.PauliX(0)
            qml.PauliX(2)
            qml.SingleExcitation(0.1, wires=[0, 1])
            qml.SingleExcitation(0.2, wires=[2, 3])
            qml.SingleExcitation(0.3, wires=[1, 2])
            return qml.expval(qml.SparseHamiltonian(h, wires=[0, 1, 2, 3]))

        res = result()
        exp_res = 0.019833838076209875
        assert np.allclose(res, exp_res, atol=tol(False))


@flaky(max_runs=10)
class TestSample:
    """Tests for the sample return type."""

    def test_sample_values(self, device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.5708, wires=[0])
            return qml.sample(qml.PauliZ(wires=0))

        res = circuit()

        # res should only contain 1 and -1
        assert np.allclose(res ** 2, 1, atol=tol(False))

    def test_sample_values_hermitian(self, device, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        A_ = np.array([[1, 2j], [-2j, 0]])
        theta = 0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            return qml.sample(qml.Hermitian(A_, wires=0))

        res = circuit().flatten()

        # res should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A_)
        assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))
        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(
            np.mean(res), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, atol=tol(False)
        )
        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(
            np.var(res), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, atol=tol(False)
        )

    def test_sample_values_projector(self, device, tol):
        """Tests if the samples of a Projector observable returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            return qml.sample(qml.Projector(basis_state, wires=0))

        res = circuit([0]).flatten()

        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))

        assert np.allclose(np.mean(res), np.cos(theta / 2) ** 2, atol=tol(False))

        assert np.allclose(
            np.var(res), np.cos(theta / 2) ** 2 - (np.cos(theta / 2) ** 2) ** 2, atol=tol(False)
        )

        res = circuit([1]).flatten()

        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))

        assert np.allclose(np.mean(res), np.sin(theta / 2) ** 2, atol=tol(False))

        assert np.allclose(
            np.var(res), np.sin(theta / 2) ** 2 - (np.sin(theta / 2) ** 2) ** 2, atol=tol(False)
        )

    def test_sample_values_hermitian_multi_qubit(self, device, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        theta = 0.543
        A_ = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Hermitian(A_, wires=[0, 1]))

        res = circuit().flatten()

        # res should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A_)
        assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))

        # make sure the mean matches the analytic mean
        expected = (
            88 * np.sin(theta)
            + 24 * np.sin(2 * theta)
            - 40 * np.sin(3 * theta)
            + 5 * np.cos(theta)
            - 6 * np.cos(2 * theta)
            + 27 * np.cos(3 * theta)
            + 6
        ) / 32
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))

    def test_sample_values_projector_multi_qubit(self, device, tol):
        """Tests if the samples of a multi-qubit Projector observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Projector(basis_state, wires=[0, 1]))

        res = circuit([0, 0]).flatten()
        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        expected = (np.cos(theta / 2) * np.cos(theta)) ** 2
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))

        res = circuit([0, 1]).flatten()
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        expected = (np.cos(theta / 2) * np.sin(theta)) ** 2
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))

        res = circuit([1, 0]).flatten()
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        expected = (np.sin(theta / 2) * np.sin(theta)) ** 2
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))

        res = circuit([1, 1]).flatten()
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        expected = (np.sin(theta / 2) * np.cos(theta)) ** 2
        assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorSample:
    """Test tensor sample values."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]))

        res = circuit()

        # res should only contain 1 and -1
        assert np.allclose(res ** 2, 1, atol=tol(False))

        mean = np.mean(res)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol(False))

        var = np.var(res)
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol(False))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(
                qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2])
            )

        res = circuit()

        # s1 should only contain 1 and -1
        assert np.allclose(res ** 2, 1, atol=tol(False))

        mean = np.mean(res)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol(False))

        var = np.var(res)
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol(False))

    def test_hermitian(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        A_ = 0.1 * np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.PauliZ(wires=[0]) @ qml.Hermitian(A_, wires=[1, 2]))

        res = circuit()

        # res should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A_))
        assert np.allclose(sorted(np.unique(res)), sorted(eigvals), atol=tol(False))

        mean = np.mean(res)
        expected = (
            0.1
            * 0.5
            * (
                -6 * np.cos(theta) * (np.cos(varphi) + 1)
                - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
                + 3 * np.cos(varphi) * np.sin(phi)
                + np.sin(phi)
            )
        )
        assert np.allclose(mean, expected, atol=tol(False))

        var = np.var(res)
        expected = (
            0.01
            * (
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
            )
            / 16
        )
        assert np.allclose(var, expected, atol=tol(False))

    def test_projector(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if dev.shots is None:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 1.432
        phi = 1.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.PauliZ(wires=[0]) @ qml.Projector(basis_state, wires=[1, 2]))

        res = circuit([0, 0])
        # res should only contain the eigenvalues of the projector matrix tensor product Z, i.e. {-1, 0, 1}
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        mean = np.mean(res)
        expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(var, expected, atol=tol(False))

        res = circuit([0, 1])
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        mean = np.mean(res)
        expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(var, expected, atol=tol(False))

        res = circuit([1, 0])
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        mean = np.mean(res)
        expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(var, expected, atol=tol(False))

        res = circuit([1, 1])
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        mean = np.mean(res)
        expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(var, expected, atol=tol(False))


@flaky(max_runs=10)
class TestVar:
    """Tests for the variance return type"""

    def test_var(self, device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        phi = 0.543
        theta = 0.6543

        @qml.qnode(dev)
        def circuit():
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            return qml.var(qml.PauliZ(wires=0))

        res = circuit()

        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_var_hermitian(self, device, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        phi = 0.543
        theta = 0.6543
        # test correct variance for <H> of a rotated state
        H = 0.1 * np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        @qml.qnode(dev)
        def circuit():
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[0])
            return qml.var(qml.Hermitian(H, wires=0))

        res = circuit()

        expected = (
            0.01
            * 0.5
            * (
                2 * np.sin(2 * theta) * np.cos(phi) ** 2
                + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
                + 35 * np.cos(2 * phi)
                + 39
            )
        )

        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_var_projector(self, device, tol):
        """Tests if the samples of a Projector observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        phi = 0.543
        theta = 0.654

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(basis_state, wires=[0, 1]))

        res = circuit([0, 0])
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([0, 1])
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 0])
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 1])
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorVar:
    """Test tensor variance measurements."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]))

        res = circuit()

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2]))

        res = circuit()

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(res, expected, atol=tol(dev.shots))

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "base_obs, permuted_obs", list(zip(obs_lst, obs_permuted_lst)),
    )
    def test_wire_order_in_tensor_prod_observables(
        self, device, base_obs, permuted_obs, tol, skip_if
    ):
        """Test that when given a tensor observable the variance is the same regardless of the order of terms
        in the tensor observable, provided the wires each term acts on remain constant.

        eg:
        ob1 = qml.PauliZ(wires=0) @ qml.PauliY(wires=1)
        ob2 = qml.PauliY(wires=1) @ qml.PauliZ(wires=0)

        @qml.qnode(dev)
        def circ(obs):
            return qml.var(obs)

        circ(ob1) == circ(ob2)
        """
        n_wires = 3
        dev = device(n_wires)
        skip_if(dev, {"supports_tensor_observables": False})

        @qml.qnode(dev)
        def circ(ob):
            sub_routine(label_map=range(3))
            return qml.var(ob)

        assert np.allclose(circ(base_obs), circ(permuted_obs), atol=tol(dev.shots), rtol=0)

    @pytest.mark.parametrize("label_map", label_maps)
    def test_wire_label_in_tensor_prod_observables(self, device, label_map, tol, skip_if):
        """Test that when given a tensor observable the variance is the same regardless of how the
        wires are labelled, as long as they match the device order.

        eg:
        dev1 = qml.device("default.qubit", wires=[0, 1, 2])
        dev2 = qml.device("default.qubit", wires=['c', 'b', 'a']

        def circ(wire_labels):
            return qml.var(qml.PauliZ(wires=wire_labels[0]) @ qml.PauliX(wires=wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
        dev = device(wires=3)
        dev_custom_labels = device(wires=label_map)
        skip_if(dev, {"supports_tensor_observables": False})

        def circ(wire_labels):
            sub_routine(wire_labels)
            return qml.var(
                qml.PauliX(wire_labels[0]) @ qml.PauliY(wire_labels[1]) @ qml.PauliZ(wire_labels[2])
            )

        circ_base_label = qml.QNode(circ, device=dev)
        circ_custom_label = qml.QNode(circ, device=dev_custom_labels)

        assert np.allclose(
            circ_base_label(wire_labels=range(3)),
            circ_custom_label(wire_labels=label_map),
            atol=tol(dev.shots),
            rtol=0,
        )

    def test_hermitian(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if "Hermitian" not in dev.observables:
            pytest.skip("Skipped because device does not support the Hermitian observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        A_ = 0.1 * np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliZ(wires=[0]) @ qml.Hermitian(A_, wires=[1, 2]))

        res = circuit()

        expected = (
            0.01
            * (
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
            )
            / 16
        )

        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_projector(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.var(qml.PauliZ(wires=[0]) @ qml.Projector(basis_state, wires=[1, 2]))

        res = circuit([0, 0])
        expected = (
            (
                (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            - (
                (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([0, 1])
        expected = (
            (
                (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            - (
                (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 0])
        expected = (
            (
                (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            - (
                (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(res, expected, atol=tol(dev.shots))

        res = circuit([1, 1])
        expected = (
            (
                (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            - (
                (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(res, expected, atol=tol(dev.shots))
