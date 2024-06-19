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
from pennylane.measurements import (
    ClassicalShadowMP,
    MeasurementTransform,
    SampleMeasurement,
    SampleMP,
    StateMeasurement,
    StateMP,
)
from pennylane.wires import Wires

pytestmark = pytest.mark.skip_unsupported

# ==========================================================
# Some useful global variables

# observables for which device support is tested
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
    "Prod": qml.prod(qml.X(0), qml.Z(1)),
    "SProd": qml.s_prod(0.1, qml.Z(0)),
    "Sum": qml.sum(qml.s_prod(0.1, qml.Z(0)), qml.prod(qml.X(0), qml.Z(1))),
    "LinearCombination": qml.ops.LinearCombination([1, 1], [qml.Z(0), qml.X(0)]),
}

all_obs = obs.keys()

# All qubit observables should be available to test in the device test suite
all_available_obs = qml.ops._qubit__obs__.copy().union(  # pylint: disable=protected-access
    {"Prod", "SProd", "Sum"}
)
# Note that the identity is not technically a qubit observable
all_available_obs |= {"Identity"}

if not set(all_obs) == all_available_obs | {"LinearCombination"}:
    raise ValueError(
        "A qubit observable has been added that is not being tested in the "
        "device test suite. Please add to the obs dictionary in "
        "pennylane/devices/tests/test_measurements.py"
    )

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

obs_lst = [
    qml.X(0) @ qml.Y(1),
    qml.X(1) @ qml.Y(0),
    qml.X(1) @ qml.Z(2),
    qml.X(2) @ qml.Z(1),
    qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.Z(2),
    qml.Z(0) @ qml.X(1) @ qml.Y(2),
]

obs_permuted_lst = [
    qml.Y(1) @ qml.X(0),
    qml.Y(0) @ qml.X(1),
    qml.Z(2) @ qml.X(1),
    qml.Z(1) @ qml.X(2),
    qml.Z(2) @ qml.Identity(wires=0) @ qml.Identity(wires=1),
    qml.X(1) @ qml.Y(2) @ qml.Z(0),
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

        if dev.shots and observable == "SparseHamiltonian":
            pytest.skip("SparseHamiltonian only supported in analytic mode")

        if isinstance(dev, qml.Device):
            assert hasattr(dev, "observables")
            if observable not in dev.observables:
                pytest.skip("observable not supported")

        kwargs = {"diff_method": "parameter-shift"} if observable == "SparseHamiltonian" else {}

        @qml.qnode(dev, **kwargs)
        def circuit(obs_circ):
            qml.PauliX(0)
            return qml.expval(obs_circ)

        if observable == "Projector":
            for o in obs[observable]:
                assert isinstance(circuit(o), (float, np.ndarray))
        else:
            assert isinstance(circuit(obs[observable]), (float, np.ndarray))

    def test_tensor_observables_can_be_implemented(self, device_kwargs):
        """Test that the device can implement a simple tensor observable.
        This test is skipped for devices that do not support tensor observables."""
        device_kwargs["wires"] = 2
        dev = qml.device(**device_kwargs)
        supports_tensor = isinstance(dev, qml.devices.Device) or (
            "supports_tensor_observables" in dev.capabilities()
            and dev.capabilities()["supports_tensor_observables"]
        )
        if not supports_tensor:
            pytest.skip("Device does not support tensor observables.")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))

        assert isinstance(circuit(), (float, np.ndarray))


# pylint: disable=too-few-public-methods
@flaky(max_runs=10)
class TestHamiltonianSupport:
    """Separate test to ensure that the device can differentiate Hamiltonian observables."""

    @pytest.mark.parametrize("ham_constructor", [qml.ops.Hamiltonian, qml.ops.LinearCombination])
    @pytest.mark.filterwarnings("ignore::pennylane.PennyLaneDeprecationWarning")
    def test_hamiltonian_diff(self, ham_constructor, device_kwargs, tol):
        """Tests a simple VQE gradient using parameter-shift rules."""

        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        coeffs = np.array([-0.05, 0.17])
        param = np.array(1.7, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(coeffs, param):
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(
                ham_constructor(
                    coeffs,
                    [qml.X(0), qml.Z(0)],
                )
            )

        grad_fn = qml.grad(circuit)
        grad = grad_fn(coeffs, param)

        def circuit1(param):
            """First Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.X(0))

        def circuit2(param):
            """Second Pauli subcircuit"""
            qml.RX(param, wires=0)
            qml.RY(param, wires=0)
            return qml.expval(qml.Z(0))

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
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res = circuit()
        expected = np.array([np.cos(theta), np.cos(theta) * np.cos(phi)])
        assert np.allclose(res, expected, atol=tol(dev.shots))

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
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

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
            return qml.expval(qml.Y(0)), qml.expval(qml.Y(1))

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

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
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

        if isinstance(dev, qml.Device) and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.732
        phi = 0.523

        @qml.qnode(dev)
        def circuit(state):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector(state, wires=[0, 1]))

        basis_state, state_vector = [0, 0], [1, 0, 0, 0]
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [0, 1], [0, 1, 0, 0]
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [1, 0], [0, 0, 1, 0]
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [1, 1], [0, 0, 0, 1]
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

    def test_multi_mode_hermitian_expectation(self, device, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
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

    @pytest.mark.parametrize(
        "o",
        [
            qml.prod(qml.X(0), qml.Z(1)),
            qml.s_prod(0.1, qml.Z(0)),
            qml.sum(qml.s_prod(0.1, qml.Z(0)), qml.prod(qml.X(0), qml.Z(1))),
        ],
    )
    def test_op_arithmetic_matches_default_qubit(self, o, device, tol):
        """Test that devices (which support the observable) match default.qubit results."""
        dev = device(2)
        if isinstance(dev, qml.Device) and o.name not in dev.observables:
            pytest.skip(f"Skipped because device does not support the {o.name} observable.")

        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.expval(o)

        res_dq = qml.QNode(circuit, qml.device("default.qubit"))()
        res = qml.QNode(circuit, dev)()
        assert qml.math.shape(res) == ()
        assert np.isclose(res, res_dq, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
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
            return qml.expval(qml.X(0) @ qml.Y(2))

        res = circuit()

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
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
            return qml.expval(qml.Z(0) @ qml.Hadamard(wires=1) @ qml.Y(2))

        res = circuit()

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "base_obs, permuted_obs",
        list(zip(obs_lst, obs_permuted_lst)),
    )
    def test_wire_order_in_tensor_prod_observables(
        self, device, base_obs, permuted_obs, tol, skip_if
    ):
        """Test that when given a tensor observable the expectation value is the same regardless of the order of terms
        in the tensor observable, provided the wires each term acts on remain constant.

        eg:
        ob1 = qml.Z(0) @ qml.Y(1)
        ob2 = qml.Y(1) @ qml.Z(0)

        @qml.qnode(dev)
        def circ(obs):
            return qml.expval(obs)

        circ(ob1) == circ(ob2)
        """
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
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
            return qml.expval(qml.Z(wire_labels[0]) @ qml.X(wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
        dev = device(wires=3)
        dev_custom_labels = device(wires=label_map)
        if isinstance(dev, qml.Device):
            skip_if(dev, {"supports_tensor_observables": False})

        def circ(wire_labels):
            sub_routine(wire_labels)
            return qml.expval(qml.X(wire_labels[0]) @ qml.Y(wire_labels[1]) @ qml.Z(wire_labels[2]))

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

        if isinstance(dev, qml.Device):
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
            return qml.expval(qml.Z(0) @ qml.Hermitian(A_, wires=[1, 2]))

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

        if isinstance(dev, qml.Device):
            if "Projector" not in dev.observables:
                pytest.skip("Skipped because device does not support the Projector observable.")

            skip_if(dev, {"supports_tensor_observables": False})

        theta = 0.732
        phi = 0.523
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.Z(0) @ qml.Projector(state, wires=[1, 2]))

        basis_state, state_vector = [0, 0], [1, 0, 0, 0]
        expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [0, 1], [0, 1, 0, 0]
        expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [1, 0], [0, 0, 1, 0]
        expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

        basis_state, state_vector = [1, 1], [0, 0, 0, 1]
        expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

    def test_sparse_hamiltonian_expval(self, device, tol):
        """Test that expectation values of sparse Hamiltonians are properly calculated."""
        n_wires = 4
        dev = device(n_wires)

        if isinstance(dev, qml.Device):
            if "SparseHamiltonian" not in dev.observables:
                pytest.skip(
                    "Skipped because device does not support the SparseHamiltonian observable."
                )
        if dev.shots:
            pytest.skip("SparseHamiltonian only supported in analytic mode")

        h_row = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        h_col = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        h_data = np.array(
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1], dtype=np.complex128
        )
        h = csr_matrix((h_data, (h_row, h_col)), shape=(16, 16))  # XXYY

        @qml.qnode(dev, diff_method="parameter-shift")
        def result():
            qml.X(0)
            qml.X(2)
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

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.5708, wires=[0])
            return qml.sample(qml.Z(0))

        res = circuit()

        # res should only contain 1 and -1
        assert np.allclose(res**2, 1, atol=tol(False))

    def test_sample_values_hermitian(self, device, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 1
        dev = device(n_wires)

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
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

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device) and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            return qml.sample(qml.Projector(state, wires=0))

        expected = np.cos(theta / 2) ** 2
        res_basis = circuit([0]).flatten()
        res_state = circuit([1, 0]).flatten()
        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected - (expected) ** 2, atol=tol(False))
        assert np.allclose(np.var(res_state), expected - (expected) ** 2, atol=tol(False))

        expected = np.sin(theta / 2) ** 2
        res_basis = circuit([1]).flatten()
        res_state = circuit([0, 1]).flatten()
        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected - (expected) ** 2, atol=tol(False))
        assert np.allclose(np.var(res_state), expected - (expected) ** 2, atol=tol(False))

        expected = 0.5
        res = circuit(np.array([1, 1]) / np.sqrt(2)).flatten()
        assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res), expected, atol=tol(False))
        assert np.allclose(np.var(res), expected - (expected) ** 2, atol=tol(False))

    def test_sample_values_hermitian_multi_qubit(self, device, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
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

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device) and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        theta = 0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            qml.RY(2 * theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Projector(state, wires=[0, 1]))

        expected = (np.cos(theta / 2) * np.cos(theta)) ** 2
        res_basis = circuit([0, 0]).flatten()
        res_state = circuit([1, 0, 0, 0]).flatten()
        # res should only contain 0 or 1, the eigenvalues of the projector
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))

        expected = (np.cos(theta / 2) * np.sin(theta)) ** 2
        res_basis = circuit([0, 1]).flatten()
        res_state = circuit([0, 1, 0, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))

        expected = (np.sin(theta / 2) * np.sin(theta)) ** 2
        res_basis = circuit([1, 0]).flatten()
        res_state = circuit([0, 0, 1, 0]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))

        expected = (np.sin(theta / 2) * np.cos(theta)) ** 2
        res_basis = circuit([1, 1]).flatten()
        res_state = circuit([0, 0, 0, 1]).flatten()
        assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
        assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
        assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorSample:
    """Test tensor sample values."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device):
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
            return qml.sample(qml.X(0) @ qml.Y(2))

        res = circuit()

        # res should only contain 1 and -1
        assert np.allclose(res**2, 1, atol=tol(False))

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

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device):
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
            return qml.sample(qml.Z(0) @ qml.Hadamard(wires=[1]) @ qml.Y(2))

        res = circuit()

        # s1 should only contain 1 and -1
        assert np.allclose(res**2, 1, atol=tol(False))

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

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device):
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
            return qml.sample(qml.Z(0) @ qml.Hermitian(A_, wires=[1, 2]))

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

    def test_projector(self, device, tol, skip_if):  # pylint: disable=too-many-statements
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)

        if not dev.shots:
            pytest.skip("Device is in analytic mode, cannot test sampling.")

        if isinstance(dev, qml.Device):
            if "Projector" not in dev.observables:
                pytest.skip("Skipped because device does not support the Projector observable.")

            skip_if(dev, {"supports_tensor_observables": False})

        theta = 1.432
        phi = 1.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.Z(0) @ qml.Projector(state, wires=[1, 2]))

        res_basis = circuit([0, 0]).flatten()
        res_state = circuit([1, 0, 0, 0]).flatten()
        expected_mean = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        expected_var = (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        # res should only contain the eigenvalues of the projector matrix tensor product Z, i.e. {-1, 0, 1}
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))

        res_basis = circuit([0, 1]).flatten()
        res_state = circuit([0, 1, 0, 0]).flatten()
        expected_mean = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        expected_var = (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))

        res_basis = circuit([1, 0]).flatten()
        res_state = circuit([0, 0, 1, 0]).flatten()
        expected_mean = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        expected_var = (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))

        res_basis = circuit([1, 1]).flatten()
        res_state = circuit([0, 0, 0, 1]).flatten()
        expected_mean = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        expected_var = (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            - (
                (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
                - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
            )
            ** 2
        )
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))

        res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
        expected_mean = 0.5 * (
            (np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
            + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
            - (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
            - (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
        )
        expected_var = (
            0.5
            * (
                (np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
            )
            - expected_mean**2
        )
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res), expected_var, atol=tol(False))


@flaky(max_runs=10)
class TestSumExpval:
    """Test expectation values of Sum observables."""

    def test_sum_containing_identity_on_no_wires(self, device, tol):
        """Test that the device can handle Identity on no wires."""
        dev = device(1)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.expval(qml.sum(qml.Z(0) + 3 * qml.I()))

        res = circuit()
        assert qml.math.allclose(res, 2.0, atol=tol(dev.shots))


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
            return qml.var(qml.Z(0))

        res = circuit()

        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_var_hermitian(self, device, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        n_wires = 2
        dev = device(n_wires)

        if isinstance(dev, qml.Device) and "Hermitian" not in dev.observables:
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

        if isinstance(dev, qml.Device) and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        phi = 0.543
        theta = 0.654

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(phi, wires=[0])
            qml.RY(theta, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(state, wires=[0, 1]))

        res_basis = circuit([0, 0])
        res_state = circuit([1, 0, 0, 0])
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([0, 1])
        res_state = circuit([0, 1, 0, 0])
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([1, 0])
        res_state = circuit([0, 0, 1, 0])
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([1, 1])
        res_state = circuit([0, 0, 0, 1])
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
        expected_mean = 0.5 * (
            (np.cos(theta / 2) * np.cos(phi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2)) ** 2
        )
        expected_var = expected_mean - expected_mean**2
        assert np.allclose(res, expected_var, atol=tol(dev.shots))


@flaky(max_runs=10)
class TestTensorVar:
    """Test tensor variance measurements."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
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
            return qml.var(qml.X(0) @ qml.Y(2))

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
        if isinstance(dev, qml.Device):
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
            return qml.var(qml.Z(0) @ qml.Hadamard(wires=[1]) @ qml.Y(2))

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
        "base_obs, permuted_obs",
        list(zip(obs_lst, obs_permuted_lst)),
    )
    def test_wire_order_in_tensor_prod_observables(
        self, device, base_obs, permuted_obs, tol, skip_if
    ):
        """Test that when given a tensor observable the variance is the same regardless of the order of terms
        in the tensor observable, provided the wires each term acts on remain constant.

        eg:
        ob1 = qml.Z(0) @ qml.Y(1)
        ob2 = qml.Y(1) @ qml.Z(0)

        @qml.qnode(dev)
        def circ(obs):
            return qml.var(obs)

        circ(ob1) == circ(ob2)
        """
        n_wires = 3
        dev = device(n_wires)
        if isinstance(dev, qml.Device):
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
            return qml.var(qml.Z(wire_labels[0]) @ qml.X(wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
        dev = device(wires=3)
        dev_custom_labels = device(wires=label_map)
        if isinstance(dev, qml.Device):
            skip_if(dev, {"supports_tensor_observables": False})

        def circ(wire_labels):
            sub_routine(wire_labels)
            return qml.var(qml.X(wire_labels[0]) @ qml.Y(wire_labels[1]) @ qml.Z(wire_labels[2]))

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

        if isinstance(dev, qml.Device):
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
            return qml.var(qml.Z(0) @ qml.Hermitian(A_, wires=[1, 2]))

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

        if isinstance(dev, qml.Device):
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
            return qml.var(qml.Z(0) @ qml.Projector(basis_state, wires=[1, 2]))

        res_basis = circuit([0, 0])
        res_state = circuit([1, 0, 0, 0])
        expected = (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([0, 1])
        res_state = circuit([0, 1, 0, 0])
        expected = (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([1, 0])
        res_state = circuit([0, 0, 1, 0])
        expected = (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res_basis = circuit([1, 1])
        res_state = circuit([0, 0, 0, 1])
        expected = (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res_basis, expected, atol=tol(dev.shots))
        assert np.allclose(res_state, expected, atol=tol(dev.shots))

        res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
        expected_mean = 0.5 * (
            (np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
            + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
            - (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
            - (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
        )
        expected_var = (
            0.5
            * (
                (np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2
                + (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2
            )
            - expected_mean**2
        )
        assert np.allclose(res, expected_var, atol=tol(False))


def _skip_test_for_braket(dev):
    """Skip the specific test because the Braket plugin does not yet support custom measurement processes."""
    if "braket" in getattr(dev, "short_name", dev.name):
        pytest.skip(f"Custom measurement test skipped for {dev.short_name}.")


class TestSampleMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_sample_measurement(self, device):
        """Test the execution of a custom sampled measurement."""

        dev = device(2)
        _skip_test_for_braket(dev)

        if not dev.shots:
            pytest.skip("Shots must be specified in the device to compute a sampled measurement.")

        class MyMeasurement(SampleMeasurement):
            """Dummy sampled measurement."""

            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                return 1

            def process_counts(self, counts: dict, wire_order: Wires):
                return 1

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        res = circuit()
        assert qml.math.allequal(res, [1, 1])

    def test_sample_measurement_without_shots(self, device):
        """Test that executing a sampled measurement with ``shots=None`` raises an error."""
        dev = device(2)

        if dev.shots:
            pytest.skip("If shots!=None no error is raised.")

        class MyMeasurement(SampleMeasurement):
            """Dummy sampled measurement."""

            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                return 1

            def process_counts(self, counts: dict, wire_order: Wires):
                return 1

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        with pytest.raises((ValueError, qml.DeviceError)):
            circuit()

    def test_method_overriden_by_device(self, device):
        """Test that the device can override a measurement process."""
        dev = device(2)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test specific for old device interface.")
        _skip_test_for_braket(dev)

        if dev.shots is None:
            pytest.skip(
                "The number of shots has to be explicitly set on the device when using "
                "sample-based measurements."
            )

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.sample(wires=0), qml.sample(wires=1)

        circuit.device.measurement_map[SampleMP] = "test_method"
        circuit.device.test_method = lambda obs, shot_range=None, bin_size=None: 2

        assert qml.math.allequal(circuit(), [2, 2])


class TestStateMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_state_measurement(self, device):
        """Test the execution of a custom state measurement."""
        dev = device(2)
        _skip_test_for_braket(dev)

        if dev.shots:
            pytest.skip("Some plugins don't update state information when shots is not None.")

        class MyMeasurement(StateMeasurement):
            """Dummy state measurement."""

            def process_state(self, state, wire_order):
                return 1

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement()

        assert circuit() == 1

    def test_sample_measurement_with_shots(self, device):
        """Test that executing a state measurement with shots raises a warning."""
        dev = device(2)
        _skip_test_for_braket(dev)

        if not dev.shots:
            pytest.skip("If shots=None no warning is raised.")

        class MyMeasurement(StateMeasurement):
            """Dummy state measurement."""

            def process_state(self, state, wire_order):
                return 1

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement()

        if isinstance(dev, qml.Device):
            with pytest.warns(
                UserWarning,
                match="Requested measurement MyMeasurement with finite shots",
            ):
                circuit()
        else:
            with pytest.raises(qml.DeviceError):
                circuit()

    def test_method_overriden_by_device(self, device):
        """Test that the device can override a measurement process."""
        dev = device(2)

        _skip_test_for_braket(dev)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is specific to old device interface")

        @qml.qnode(dev, interface="autograd", diff_method=None)
        def circuit():
            qml.X(0)
            return qml.state()

        circuit.device.measurement_map[StateMP] = "test_method"
        circuit.device.test_method = lambda obs, shot_range=None, bin_size=None: 2

        assert circuit() == 2


class TestCustomMeasurement:
    """Tests for the CustomMeasurement class."""

    def test_custom_measurement(self, device):
        """Test the execution of a custom measurement."""
        dev = device(2)
        _skip_test_for_braket(dev)

        class MyMeasurement(MeasurementTransform):
            """Dummy measurement transform."""

            def process(self, tape, device):
                return 1

        if isinstance(dev, qml.devices.Device):
            tape = qml.tape.QuantumScript([], [MyMeasurement()])
            try:
                dev.preprocess()[0]((tape,))
            except qml.DeviceError:
                pytest.xfail("Device does not support custom measurement transforms.")

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return MyMeasurement()

        assert circuit() == 1

    def test_method_overriden_by_device(self, device):
        """Test that the device can override a measurement process."""
        dev = device(2)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test specific to old device interface.")
        _skip_test_for_braket(dev)

        if dev.shots is None:
            pytest.skip(
                "The number of shots has to be explicitly set on the device when using "
                "sample-based measurements."
            )

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.classical_shadow(wires=0)

        circuit.device.measurement_map[ClassicalShadowMP] = "test_method"
        circuit.device.test_method = lambda tape: 2

        assert circuit() == 2
