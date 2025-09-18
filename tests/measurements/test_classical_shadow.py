# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the classical shadows measurement processes"""

import copy
import itertools

import autograd.numpy
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning
from pennylane.measurements import ClassicalShadowMP
from pennylane.measurements.classical_shadow import ShadowExpvalMP

# pylint: disable=dangerous-default-value, too-many-arguments


def get_circuit(wires, shots, seed_recipes, interface="autograd", device="default.qubit"):
    """
    Return a QNode that prepares the state (|00...0> + |11...1>) / sqrt(2)
        and performs the classical shadow measurement
    """
    dev = qml.device(device, wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)

        for target in range(1, wires):
            qml.CNOT(wires=[0, target])

        return qml.classical_shadow(wires=range(wires), seed=seed_recipes)

    return circuit


def get_x_basis_circuit(wires, shots, interface="autograd"):
    """
    Return a QNode that prepares the |++..+> state and performs a classical shadow measurement
    """
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        for wire in range(wires):
            qml.Hadamard(wire)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def get_y_basis_circuit(wires, shots, interface="autograd"):
    """
    Return a QNode that prepares the |+i>|+i>...|+i> state and performs a classical shadow measurement
    """
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        for wire in range(wires):
            qml.Hadamard(wire)
            qml.RZ(np.pi / 2, wire)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def get_z_basis_circuit(wires, shots, interface="autograd"):
    """
    Return a QNode that prepares the |00..0> state and performs a classical shadow measurement
    """
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        return qml.classical_shadow(wires=range(wires))

    return circuit


wires_list = [1, 3]


class TestProcessState:
    """Unit tests for process_state_with_shots for the classical_shadow
    and shadow_expval measurements"""

    def test_shape_and_dtype(self):
        """Test that the shape and dtype of the measurement is correct"""
        mp = qml.classical_shadow(wires=[0, 1])
        res = mp.process_state_with_shots(np.ones((2, 2)) / 2, qml.wires.Wires([0, 1]), shots=100)

        assert res.shape == (2, 100, 2)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert np.all(np.logical_or(res[0] == 0, res[0] == 1))

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert np.all(np.logical_or(np.logical_or(res[1] == 0, res[1] == 1), res[1] == 2))

    def test_wire_order(self):
        """Test that the wire order is respected"""
        state = np.array([[1, 1], [0, 0]]) / np.sqrt(2)

        mp = qml.classical_shadow(wires=[0, 1])
        res = mp.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=1000)

        assert res.shape == (2, 1000, 2)
        assert res.dtype == np.int8

        # test that the first qubit samples are all 0s when the recipe is Z
        assert np.all(res[0][res[1, ..., 0] == 2][:, 0] == 0)

        # test that the second qubit samples contain 1s when the recipe is Z
        assert np.any(res[0][res[1, ..., 1] == 2][:, 1] == 1)

        res = mp.process_state_with_shots(state, qml.wires.Wires([1, 0]), shots=1000)

        assert res.shape == (2, 1000, 2)
        assert res.dtype == np.int8

        # now test that the first qubit samples contain 1s when the recipe is Z
        assert np.any(res[0][res[1, ..., 0] == 2][:, 0] == 1)

        # now test that the second qubit samples are all 0s when the recipe is Z
        assert np.all(res[0][res[1, ..., 1] == 2][:, 1] == 0)

    def test_subset_wires(self):
        """Test that the measurement is correct when only a subset of wires is measured"""
        mp = qml.classical_shadow(wires=[0, 1])

        # GHZ state
        state = np.zeros((2, 2, 2))
        state[np.array([0, 1]), np.array([0, 1]), np.array([0, 1])] = 1 / np.sqrt(2)

        res = mp.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=100)

        assert res.shape == (2, 100, 2)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert np.all(np.logical_or(res[0] == 0, res[0] == 1))

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert np.all(np.logical_or(np.logical_or(res[1] == 0, res[1] == 1), res[1] == 2))

    def test_same_rng(self):
        """Test results when the rng is the same"""
        state = np.ones((2, 2)) / 2

        mp1 = qml.classical_shadow(wires=[0, 1], seed=123)
        mp2 = qml.classical_shadow(wires=[0, 1], seed=123)

        res1 = mp1.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=100)
        res2 = mp2.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=100)

        # test recipes are the same but bits are different
        assert np.all(res1[1] == res2[1])
        assert np.any(res1[0] != res2[0])

        res1 = mp1.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=100, rng=456)
        res2 = mp2.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=100, rng=456)

        # now test everything is the same
        assert np.all(res1[1] == res2[1])
        assert np.all(res1[0] == res2[0])

    def test_expval_shape_and_val(self):
        """Test that shadow expval measurements work as expected"""
        mp = qml.shadow_expval(qml.PauliX(0) @ qml.PauliX(1), seed=200)
        res = mp.process_state_with_shots(
            np.ones((2, 2)) / 2, qml.wires.Wires([0, 1]), shots=1000, rng=100
        )

        assert res.shape == ()
        assert np.allclose(res, 1.0, atol=0.05)

    def test_expval_wire_order(self):
        """Test that shadow expval respects the wire order"""
        state = np.array([[1, 1], [0, 0]]) / np.sqrt(2)

        mp = qml.shadow_expval(qml.PauliZ(0), seed=200)
        res = mp.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=3000, rng=100)

        assert res.shape == ()
        assert np.allclose(res, 1.0, atol=0.05)

        res = mp.process_state_with_shots(state, qml.wires.Wires([1, 0]), shots=3000, rng=100)

        assert res.shape == ()
        assert np.allclose(res, 0.0, atol=0.05)

    def test_expval_same_rng(self):
        """Test expval results when the rng is the same"""
        state = np.ones((2, 2)) / 2

        mp1 = qml.shadow_expval(qml.PauliZ(0) @ qml.PauliZ(1), seed=123)
        mp2 = qml.shadow_expval(qml.PauliZ(0) @ qml.PauliZ(1), seed=123)

        res1 = mp1.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=1000, rng=100)
        res2 = mp2.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=1000, rng=200)

        # test results are different
        assert res1 != res2

        res1 = mp1.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=1000, rng=456)
        res2 = mp2.process_state_with_shots(state, qml.wires.Wires([0, 1]), shots=1000, rng=456)

        # now test that results are the same
        assert res1 == res2


class TestProcessDensityMatrix:
    """Unit tests for process_density_matrix_with_shots for the classical_shadow
    and shadow_expval measurements"""

    def test_shape_and_dtype(self):
        """Test that the shape and dtype of the measurement is correct"""
        mp = qml.classical_shadow(wires=[0, 1])
        state = np.ones((2, 2)) / 2
        dm = np.outer(state, state).reshape((2,) * 4)
        res = mp.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=100)

        assert res.shape == (2, 100, 2)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert set(res[0].ravel()).issubset({0, 1})

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert set(res[1].ravel()).issubset({0, 1, 2})

    def test_wire_order(self, seed):
        """Test that the wire order is respected"""
        state = np.array([[1, 1], [0, 0]]) / np.sqrt(2)
        dm = np.outer(state, state).reshape((2,) * 4)

        mp = qml.classical_shadow(wires=[0, 1])
        res = mp.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=1000, rng=seed
        )

        assert res.shape == (2, 1000, 2)
        assert res.dtype == np.int8

        # test that the first qubit samples are all 0s when the recipe is Z
        assert np.all(res[0][res[1, ..., 0] == 2][:, 0] == 0)

        # test that the second qubit samples contain 1s when the recipe is Z
        assert np.any(res[0][res[1, ..., 1] == 2][:, 1] == 1)

        res = mp.process_density_matrix_with_shots(
            dm, qml.wires.Wires([1, 0]), shots=1000, rng=seed
        )

        assert res.shape == (2, 1000, 2)
        assert res.dtype == np.int8

        # now test that the first qubit samples contain 1s when the recipe is Z
        assert np.any(res[0][res[1, ..., 0] == 2][:, 0] == 1)

        # now test that the second qubit samples are all 0s when the recipe is Z
        assert np.all(res[0][res[1, ..., 1] == 2][:, 1] == 0)

    @pytest.mark.parametrize("num_wires", [3, 4, 5])
    @pytest.mark.parametrize("shots", [100])
    def test_wire_order_generalized(self, num_wires, shots, seed):
        """Test that the wire order is respected for any number of wires and all permutations"""
        # Prepare a GHZ-like state for num_wires
        state = np.zeros((2,) * num_wires)
        idx0 = tuple([0] * num_wires)
        idx1 = tuple([1] * num_wires)
        state[idx0] = 1 / np.sqrt(2)
        state[idx1] = 1 / np.sqrt(2)
        dm = np.outer(state.ravel(), state.ravel()).reshape((2,) * (2 * num_wires))

        wires = list(range(num_wires))
        mp = qml.classical_shadow(wires=wires)

        for perm in itertools.permutations(wires):
            res = mp.process_density_matrix_with_shots(
                dm, qml.wires.Wires(list(perm)), shots=shots, rng=seed
            )
            assert res.shape == (2, shots, num_wires)
            # bits are always 0 or 1
            assert set(res[0].ravel()).issubset({0, 1})
            # recipes are always 0, 1, or 2
            assert set(res[1].ravel()).issubset({0, 1, 2})

            # For GHZ state, when all qubits are measured in Z basis,
            # they should all have the same value (all 0s or all 1s)
            z_shots = np.where(np.all(res[1] == 2, axis=1))[0]
            if len(z_shots) > 0:  # Only test if we have any all-Z shots
                for shot_idx in z_shots:
                    # All qubits should have the same measurement outcome
                    first_bit = res[0][shot_idx, 0]
                    assert np.all(res[0][shot_idx] == first_bit)

    def test_subset_wires(self):
        """Test that the measurement is correct when only a subset of wires is measured"""
        mp = qml.classical_shadow(wires=[0, 1])

        # GHZ state
        state = np.zeros((2, 2, 2))
        state[np.array([0, 1]), np.array([0, 1]), np.array([0, 1])] = 1 / np.sqrt(2)

        dm = np.outer(state, state).reshape((2,) * 6)

        res = mp.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=100)

        assert res.shape == (2, 100, 2)
        assert res.dtype == np.int8

        # test that the bits are either 0 and 1
        assert set(res[0].ravel()).issubset({0, 1})

        # test that the recipes are either 0, 1, or 2 (X, Y, or Z)
        assert set(res[1].ravel()).issubset({0, 1, 2})

    def test_same_rng(self):
        """Test results when the rng is the same"""
        state = np.ones((2, 2)) / 2

        dm = np.outer(state, state).reshape((2,) * 4)

        mp1 = qml.classical_shadow(wires=[0, 1], seed=123)
        mp2 = qml.classical_shadow(wires=[0, 1], seed=123)

        res1 = mp1.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=100)
        res2 = mp2.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=100)

        # test recipes are the same but bits are different
        assert np.all(res1[1] == res2[1])
        assert np.any(res1[0] != res2[0])

        res1 = mp1.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=100, rng=456
        )
        res2 = mp2.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=100, rng=456
        )

        # now test everything is the same
        assert np.all(res1[1] == res2[1])
        assert np.all(res1[0] == res2[0])

    def test_expval_shape_and_val(self):
        """Test that shadow expval measurements work as expected"""
        mp = qml.shadow_expval(qml.PauliX(0) @ qml.PauliX(1), seed=200)
        state = np.ones((2, 2)) / 2
        dm = np.outer(state, state).reshape((2,) * 4)
        res = mp.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=1000, rng=100)

        assert res.shape == ()
        assert np.allclose(res, 1.0, atol=0.05)

    def test_expval_wire_order(self):
        """Test that shadow expval respects the wire order"""
        state = np.array([[1, 1], [0, 0]]) / np.sqrt(2)

        dm = np.outer(state, state).reshape((2,) * 4)

        mp = qml.shadow_expval(qml.PauliZ(0), seed=200)
        res = mp.process_density_matrix_with_shots(dm, qml.wires.Wires([0, 1]), shots=3000, rng=100)

        assert res.shape == ()
        assert np.allclose(res, 1.0, atol=0.05)

        res = mp.process_density_matrix_with_shots(dm, qml.wires.Wires([1, 0]), shots=3000, rng=100)

        assert res.shape == ()
        assert np.allclose(res, 0.0, atol=0.05)

    def test_expval_same_rng(self):
        """Test expval results when the rng is the same"""
        state = np.ones((2, 2)) / 2

        dm = np.outer(state, state).reshape((2,) * 4)

        mp1 = qml.shadow_expval(qml.PauliZ(0) @ qml.PauliZ(1), seed=123)
        mp2 = qml.shadow_expval(qml.PauliZ(0) @ qml.PauliZ(1), seed=123)

        res1 = mp1.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=1000, rng=100
        )
        res2 = mp2.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=1000, rng=200
        )

        # test results are different
        assert res1 != res2

        res1 = mp1.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=1000, rng=456
        )
        res2 = mp2.process_density_matrix_with_shots(
            dm, qml.wires.Wires([0, 1]), shots=1000, rng=456
        )

        # now test that results are the same
        assert res1 == res2


@pytest.mark.parametrize("wires", wires_list)
class TestClassicalShadow:
    """Unit tests for classical_shadow measurement"""

    shots_list = [1, 100]
    seed_recipes_list = [None, 74]  # random seed

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_numeric_type(self, wires, seed):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        res = qml.classical_shadow(wires=range(wires), seed=seed)
        assert res.numeric_type == int

    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_shape(self, wires, shots, seed):
        """Test that the shape of the MeasurementProcess instance is correct"""
        res = qml.classical_shadow(wires=range(wires), seed=seed)
        assert res.shape(shots, wires) == (2, shots, wires)

        # test an error is raised when device is None
        msg = "Shots must be specified to obtain the shape of a classical shadow measurement"
        with pytest.raises(qml.measurements.MeasurementShapeError, match=msg):
            res.shape(None, wires)

    def test_shape_matches(self, wires):
        """Test that the shape of the MeasurementProcess matches the shape
        of the tape execution"""
        shots = 100

        circuit = get_circuit(wires, shots, True)
        tape = qml.workflow.construct_tape(circuit)()

        res = qml.execute([tape], circuit.device, None)[0]
        expected_shape = qml.classical_shadow(wires=range(wires)).shape(shots, wires)

        assert res.shape == expected_shape

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_copy(self, wires, seed):
        """Test that the attributes of the MeasurementProcess instance are
        correctly copied"""
        res = qml.classical_shadow(wires=range(wires), seed=seed)

        copied_res = copy.copy(res)
        assert isinstance(copied_res, ClassicalShadowMP)
        assert isinstance(res, ClassicalShadowMP)
        assert copied_res.wires == res.wires
        assert copied_res.seed == res.seed

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("seed", seed_recipes_list)
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    def test_format(self, wires, shots, seed, interface, device):
        """Test that the format of the returned classical shadow
        measurement is correct"""
        import torch

        circuit = get_circuit(wires, shots, seed, interface, device)
        shadow = circuit()

        # test shape is correct
        assert shadow.shape == (2, shots, wires)

        # test dtype is correct
        expected_dtype = np.int8
        if interface == "torch":
            expected_dtype = torch.int8

        assert shadow.dtype == expected_dtype

        bits, recipes = shadow  # pylint: disable=unpacking-non-sequence

        # test allowed values of bits and recipes
        assert qml.math.all(np.logical_or(bits == 0, bits == 1))
        assert qml.math.all(np.logical_or(recipes == 0, np.logical_or(recipes == 1, recipes == 2)))

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize(
        "circuit_fn, basis_recipe",
        [(get_x_basis_circuit, 0), (get_y_basis_circuit, 1), (get_z_basis_circuit, 2)],
    )
    def test_return_distribution(self, wires, interface, circuit_fn, basis_recipe):
        """Test that the distribution of the bits and recipes are correct for a circuit
        that prepares all qubits in a Pauli basis"""
        # high number of shots to prevent true negatives
        shots = 1000

        circuit = circuit_fn(wires, shots=shots, interface=interface)
        bits, recipes = circuit()

        # test that the recipes follow a rough uniform distribution
        ratios = np.unique(recipes, return_counts=True)[1] / (wires * shots)
        assert np.allclose(ratios, 1 / 3, atol=1e-1)

        # test that the bit is 0 for all X measurements
        assert qml.math.allequal(bits[recipes == basis_recipe], 0)

        # test that the bits are uniformly distributed for all Y and Z measurements
        bits1 = bits[recipes == (basis_recipe + 1) % 3]
        ratios1 = np.unique(bits1, return_counts=True)[1] / bits1.shape[0]
        assert np.allclose(ratios1, 1 / 2, atol=1e-1)

        bits2 = bits[recipes == (basis_recipe + 2) % 3]
        ratios2 = np.unique(bits2, return_counts=True)[1] / bits2.shape[0]
        assert np.allclose(ratios2, 1 / 2, atol=1e-1)

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_shots_none_error(self, wires, seed):
        """Test that an error is raised when a device with shots=None is used
        to obtain classical shadows"""
        circuit = get_circuit(wires, None, seed)

        msg = "not accepted for analytic simulation on default.qubit"
        with pytest.raises(DeviceError, match=msg):
            circuit()

    @pytest.mark.parametrize("shots", shots_list)
    def test_multi_measurement_error(self, wires, shots):
        """Test that an error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=wires)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)

            for target in range(1, wires):
                qml.CNOT(wires=[0, target])

            return qml.classical_shadow(wires=range(wires)), qml.expval(qml.PauliZ(0))

        res = circuit()
        assert isinstance(res, tuple) and len(res) == 2
        assert qml.math.shape(res[0]) == (2, shots, wires)
        assert qml.math.shape(res[1]) == ()

    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("params", [[0.1, 0.2], [0.1, 0.2, 0.3]])
    def test_parameter_broadcasting(self, wires, shots, params):
        """Test that the classical_shadow measurement process supports parameter broadcasting"""

        @qml.set_shots(shots)
        @qml.qnode(qml.device("default.qubit", wires=wires))
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.classical_shadow(wires=range(wires))

        result = circuit(params)
        sequential_result = [circuit(i) for i in params]

        assert isinstance(result, np.ndarray)
        assert qml.math.shape(result) == (len(params), 2, shots, wires)
        for seq_res, res in zip(sequential_result, result):
            assert qml.math.shape(seq_res) == qml.math.shape(res)


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.shadow_expval(obs, k=k)

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.shadow_expval(obs, k=k)

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.shadow_expval(obs, k=k)

    return circuit


@pytest.mark.autograd
class TestExpvalMeasurement:
    def test_measurement_process_numeric_type(self):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        H = qml.PauliZ(0)
        res = qml.shadow_expval(H)
        assert res.numeric_type == float

    @pytest.mark.parametrize("wires", [1, 2])
    @pytest.mark.parametrize("shots", [1, 10])
    def test_measurement_process_shape(self, wires, shots):
        """Test that the shape of the MeasurementProcess instance is correct"""
        H = qml.PauliZ(0)
        res = qml.shadow_expval(H)
        assert len(res.shape(shots, wires)) == 0

    def test_shape_matches(self):
        """Test that the shape of the MeasurementProcess matches the shape
        of the tape execution"""
        wires = 2
        shots = 100
        H = qml.PauliZ(0)

        circuit = hadamard_circuit(wires, shots)
        tape = qml.workflow.construct_tape(circuit)(H)

        res = qml.execute([tape], circuit.device, None)[0]
        expected_shape = qml.shadow_expval(H).shape(shots, wires)

        assert res.shape == expected_shape

    def test_measurement_process_copy(self):
        """Test that the attributes of the MeasurementProcess instance are
        correctly copied"""
        H = qml.PauliZ(0)
        res = qml.shadow_expval(H, k=10)

        copied_res = copy.copy(res)
        assert type(copied_res) == type(res)  # pylint: disable=unidiomatic-typecheck
        assert copied_res._shortname == res._shortname  # pylint: disable=protected-access
        qml.assert_equal(copied_res.H, res.H)
        assert copied_res.k == res.k
        assert copied_res.seed == res.seed

    def test_shots_none_error(self):
        """Test that an error is raised when a device with shots=None is used
        to obtain classical shadows"""
        circuit = hadamard_circuit(2, None)
        H = qml.PauliZ(0)

        msg = "not accepted for analytic simulation on default.qubit"
        with pytest.raises(DeviceError, match=msg):
            _ = circuit(H, k=10)

    def test_multi_measurement_allowed(self, seed):
        """Test that no error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=2, seed=seed)

        @qml.set_shots(10000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.shadow_expval(qml.PauliZ(0), seed=seed), qml.expval(qml.PauliZ(0))

        res = circuit()
        assert isinstance(res, tuple)
        assert qml.math.allclose(res, 0, atol=0.05)

    def test_obs_not_queued(self):
        """Test that the observable passed to qml.shadow_expval is not queued"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliY(0)
            qml.shadow_expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "PauliY"
        assert len(tape.measurements) == 1
        assert isinstance(tape.measurements[0], ShadowExpvalMP)

    @pytest.mark.parametrize("params", [[0.1, 0.2], [0.1, 0.2, 0.3]])
    def test_expval_parameter_broadcasting(self, params):
        """Test that the shadow_expval measurement process supports parameter broadcasting"""

        @qml.set_shots(10)
        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RX(x, wires=1)
            qml.Hadamard(wires=0)
            return qml.shadow_expval([qml.PauliZ(0), qml.PauliZ(1)])

        result = circuit(params)
        sequential_result = [circuit(i) for i in params]

        assert isinstance(result, np.ndarray)
        assert qml.math.shape(result)[0] == len(params)
        for seq_res, res in zip(sequential_result, result):
            assert qml.math.shape(seq_res) == qml.math.shape(res)


obs_hadamard = [
    qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
    qml.PauliY(2),
    qml.PauliY(1) @ qml.PauliZ(2),
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
]
expected_hadamard = [1, 1, 1, 0, 0, 0, 0]

obs_max_entangled = [
    qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(2),
    qml.PauliZ(2),
    qml.Identity(1) @ qml.PauliZ(2),
    qml.PauliZ(1) @ qml.PauliZ(2),
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
    qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(2),
]
expected_max_entangled = [0, 0, 0, 0, 1, 0, 0, -1]

obs_qft = [
    qml.PauliX(0),
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
    qml.PauliZ(2),
    qml.PauliX(1) @ qml.PauliY(2),
    qml.PauliY(1) @ qml.PauliX(2),
    qml.Identity(0) @ qml.PauliY(1) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2),
    qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2),
]
expected_qft = [
    -1,
    0,
    -1 / np.sqrt(2),
    -1 / np.sqrt(2),
    0,
    0,
    1 / np.sqrt(2),
    1 / np.sqrt(2),
    -1 / np.sqrt(2),
    0,
]


@pytest.mark.autograd
class TestExpvalForward:
    """Test the shadow_expval measurement process forward pass"""

    def test_hadamard_expval(self, k=1, obs=obs_hadamard, expected=expected_hadamard):
        """Test that the expval estimation is correct for a uniform
        superposition of qubits"""
        circuit = hadamard_circuit(3, shots=100000)
        actual = circuit(obs, k=k)

        assert actual.shape == (len(obs_hadamard),)
        assert actual.dtype == np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_max_entangled_expval(
        self, k=1, obs=obs_max_entangled, expected=expected_max_entangled
    ):
        """Test that the expval estimation is correct for a maximally
        entangled state"""
        circuit = max_entangled_circuit(3, shots=100000)
        actual = circuit(obs, k=k)

        assert actual.shape == (len(obs_max_entangled),)
        assert actual.dtype == np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_non_pauli_error(self):
        """Test that an error is raised when a non-Pauli observable is passed"""
        circuit = hadamard_circuit(3)

        with pytest.raises(ValueError, match="Observable must have a valid pauli representation."):
            circuit(qml.Hadamard(0) @ qml.Hadamard(2))


# pylint: disable=too-few-public-methods
@pytest.mark.all_interfaces
class TestExpvalForwardInterfaces:

    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_qft_expval(self, interface, k=1, obs=obs_qft, expected=expected_qft):
        """Test that the expval estimation is correct for a QFT state"""
        import torch

        circuit = qft_circuit(3, shots=100000, interface=interface)
        actual = circuit(obs, k=k)

        assert actual.shape == (len(obs_qft),)
        assert actual.dtype == torch.float64 if interface == "torch" else np.float64
        assert qml.math.allclose(actual, expected, atol=1e-1)


obs_strongly_entangled = [
    qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2),
    qml.PauliY(2),
    qml.PauliY(1) @ qml.PauliZ(2),
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.Identity(2),
]


def strongly_entangling_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(x, obs, k):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return qml.shadow_expval(obs, k=k)

    return circuit


def strongly_entangling_circuit_exact(wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x, obs):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return [qml.expval(o) for o in obs]

    return circuit


class TestExpvalBackward:
    """Test the shadow_expval measurement process backward pass"""

    @pytest.mark.autograd
    def test_backward_autograd(self, obs=obs_strongly_entangled):
        """Test that the gradient of the expval estimation is correct for
        the autograd interface"""
        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="autograd")
        exact_circuit = strongly_entangling_circuit_exact(3, "autograd")

        def cost_exact(x, obs):
            return autograd.numpy.hstack(exact_circuit(x, obs))

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = np.random.uniform(
            0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
        )
        actual = qml.jacobian(shadow_circuit)(x, obs, k=1)
        expected = qml.jacobian(cost_exact, argnum=0)(x, obs)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.jax
    def test_backward_jax(self, obs=obs_strongly_entangled):
        """Test that the gradient of the expval estimation is correct for
        the jax interface"""
        import jax
        from jax import numpy as jnp

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="jax")
        exact_circuit = strongly_entangling_circuit_exact(3, "jax")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = jnp.array(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            )
        )

        actual = jax.jacrev(shadow_circuit)(x, obs, k=1)
        expected = jax.jacrev(exact_circuit)(x, obs)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.tf
    def test_backward_tf(self, obs=obs_strongly_entangled):
        """Test that the gradient of the expval estimation is correct for
        the tensorflow interface"""
        import tensorflow as tf

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="tf")
        exact_circuit = strongly_entangling_circuit_exact(3, "tf")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = tf.Variable(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            )
        )

        with tf.GradientTape() as tape:
            out = shadow_circuit(x, obs, k=10)

        actual = tape.jacobian(out, x)

        with tf.GradientTape() as tape2:
            out2 = qml.math.hstack(exact_circuit(x, obs))

        expected = tape2.jacobian(out2, x)

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.torch
    def test_backward_torch(self, obs=obs_strongly_entangled):
        """Test that the gradient of the expval estimation is correct for
        the pytorch interface"""
        import torch

        shadow_circuit = strongly_entangling_circuit(3, shots=20000, interface="torch")
        exact_circuit = strongly_entangling_circuit_exact(3, "torch")

        # make rotations close to pi / 2 to ensure gradients are not too small
        x = torch.tensor(
            np.random.uniform(
                0.8, 2, size=qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
            ),
            requires_grad=True,
        )

        actual = torch.autograd.functional.jacobian(lambda x: shadow_circuit(x, obs, k=10), x)
        expected = torch.autograd.functional.jacobian(lambda x: tuple(exact_circuit(x, obs)), x)

        assert qml.math.allclose(actual, qml.math.stack(expected), atol=1e-1)


def get_basis_circuit(wires, shots, basis, interface="autograd", device="default.mixed"):
    """
    Return a QNode that prepares a state in a given computational basis
    and performs a classical shadow measurement
    """
    dev = qml.device(device or "default.mixed", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        for wire in range(wires):
            if basis in ("x", "y"):
                qml.Hadamard(wire)
            if basis == "y":
                qml.RZ(np.pi / 2, wire)

        return qml.classical_shadow(wires=range(wires))

    return circuit


wires_list = [1, 3]


@pytest.mark.parametrize("wires", [1, 3])
@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
@pytest.mark.parametrize("circuit_basis, basis_recipe", [("x", 0), ("y", 1), ("z", 2)])
def test_return_distribution(wires, interface, circuit_basis, basis_recipe):
    """Test that the distribution of the bits and recipes are correct for a circuit
    that prepares all qubits in a Pauli basis"""
    # high number of shots to prevent true negatives
    shots = 1000

    device = "default.mixed"

    circuit = get_basis_circuit(
        wires, basis=circuit_basis, shots=shots, interface=interface, device=device
    )
    bits, recipes = circuit()  # pylint: disable=unpacking-non-sequence
    new_bits, new_recipes = circuit()  # pylint: disable=unpacking-non-sequence

    # test that the recipes follow a rough uniform distribution
    ratios = np.unique(recipes, return_counts=True)[1] / (wires * shots)
    assert np.allclose(ratios, 1 / 3, atol=1e-1)
    new_ratios = np.unique(new_recipes, return_counts=True)[1] / (wires * shots)
    assert np.allclose(new_ratios, 1 / 3, atol=1e-1)

    # test that the bit is 0 for all X measurements
    assert qml.math.allequal(bits[recipes == basis_recipe], 0)
    assert qml.math.allequal(new_bits[new_recipes == basis_recipe], 0)

    # test that the bits are uniformly distributed for all Y and Z measurements
    bits1 = bits[recipes == (basis_recipe + 1) % 3]
    ratios1 = np.unique(bits1, return_counts=True)[1] / bits1.shape[0]
    assert np.allclose(ratios1, 1 / 2, atol=1e-1)
    new_bits1 = new_bits[new_recipes == (basis_recipe + 1) % 3]
    new_ratios1 = np.unique(new_bits1, return_counts=True)[1] / new_bits1.shape[0]
    assert np.allclose(new_ratios1, 1 / 2, atol=1e-1)

    bits2 = bits[recipes == (basis_recipe + 2) % 3]
    ratios2 = np.unique(bits2, return_counts=True)[1] / bits2.shape[0]
    assert np.allclose(ratios2, 1 / 2, atol=1e-1)

    new_bits2 = new_bits[new_recipes == (basis_recipe + 2) % 3]
    new_ratios2 = np.unique(new_bits2, return_counts=True)[1] / new_bits2.shape[0]
    assert np.allclose(new_ratios2, 1 / 2, atol=1e-1)


@pytest.mark.parametrize("wires", [1, 3])
@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch"])
@pytest.mark.parametrize("circuit_basis, basis_recipe", [("x", 0), ("y", 1), ("z", 2)])
def test_return_distribution_legacy(wires, interface, circuit_basis, basis_recipe, seed):
    """Test that the distribution of the bits and recipes are correct for a circuit
    that prepares all qubits in a Pauli basis"""
    # high number of shots to prevent true negatives
    shots = 1000

    dev = DefaultQubitLegacy(wires=wires, shots=shots, seed=seed)

    with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):

        @qml.qnode(dev, interface=interface)
        def circuit():
            for wire in range(wires):
                if circuit_basis in ("x", "y"):
                    qml.Hadamard(wire)
                if circuit_basis == "y":
                    qml.RZ(np.pi / 2, wire)

            return qml.classical_shadow(wires=range(wires), seed=seed)

    bits, recipes = circuit()  # pylint: disable=unpacking-non-sequence
    tape = qml.workflow.construct_tape(circuit)()
    new_bits, new_recipes = tape.measurements[0].process(tape, circuit.device.target_device)

    # test that the recipes follow a rough uniform distribution
    ratios = np.unique(recipes, return_counts=True)[1] / (wires * shots)
    assert np.allclose(ratios, 1 / 3, atol=1e-1)
    new_ratios = np.unique(new_recipes, return_counts=True)[1] / (wires * shots)
    assert np.allclose(new_ratios, 1 / 3, atol=1e-1)

    # test that the bit is 0 for all X measurements
    assert qml.math.allequal(bits[recipes == basis_recipe], 0)
    assert qml.math.allequal(new_bits[new_recipes == basis_recipe], 0)

    # test that the bits are uniformly distributed for all Y and Z measurements
    bits1 = bits[recipes == (basis_recipe + 1) % 3]
    ratios1 = np.unique(bits1, return_counts=True)[1] / bits1.shape[0]
    assert np.allclose(ratios1, 1 / 2, atol=1e-1)
    new_bits1 = new_bits[new_recipes == (basis_recipe + 1) % 3]
    new_ratios1 = np.unique(new_bits1, return_counts=True)[1] / new_bits1.shape[0]
    assert np.allclose(new_ratios1, 1 / 2, atol=1e-1)

    bits2 = bits[recipes == (basis_recipe + 2) % 3]
    ratios2 = np.unique(bits2, return_counts=True)[1] / bits2.shape[0]
    assert np.allclose(ratios2, 1 / 2, atol=1e-1)

    new_bits2 = new_bits[new_recipes == (basis_recipe + 2) % 3]
    new_ratios2 = np.unique(new_bits2, return_counts=True)[1] / new_bits2.shape[0]
    assert np.allclose(new_ratios2, 1 / 2, atol=1e-1)


def hadamard_circuit_legacy(wires, shots=10000, interface="autograd"):
    dev = DefaultQubitLegacy(wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.shadow_expval(obs, k=k)

    return circuit


def test_hadamard_expval_legacy(k=1, obs=obs_hadamard, expected=expected_hadamard):
    """Test that the expval estimation is correct for a uniform
    superposition of qubits"""
    with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):
        circuit = hadamard_circuit_legacy(3, shots=50000)
    actual = circuit(obs, k=k)

    tape = qml.workflow.construct_tape(circuit)(obs)
    new_actual = tape.measurements[0].process(tape, circuit.device.target_device)

    assert actual.shape == (len(obs_hadamard),)
    assert actual.dtype == np.float64
    assert qml.math.allclose(actual, expected, atol=1e-1)
    assert qml.math.allclose(new_actual, expected, atol=1e-1)


def hadamard_circuit_mixed(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.mixed", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.shadow_expval(obs, k=k)

    return circuit


@pytest.mark.slow
def test_hadamard_expval_mixed(k=1, obs=obs_hadamard, expected=expected_hadamard):
    """Test that the expval estimation is correct for a uniform
    superposition of qubits"""
    circuit = hadamard_circuit_mixed(3, shots=50000)
    actual = circuit(obs, k=k)
    new_actual = circuit(obs, k=k)

    assert actual.shape == (len(obs_hadamard),)
    assert actual.dtype == np.float64
    assert qml.math.allclose(actual, expected, atol=1e-1)
    assert qml.math.allclose(new_actual, expected, atol=1e-1)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch"])
@pytest.mark.parametrize("circuit_basis", ["x", "y", "z"])
def test_partitioned_shots(interface, circuit_basis):
    """Test that mixed device works for partitioned shots"""
    wires = 3
    shot = 100
    shots = (shot, shot)

    device = "default.mixed"
    circuit = get_basis_circuit(
        wires, basis=circuit_basis, shots=shots, interface=interface, device=device
    )
    bits, recipes = circuit()  # pylint: disable=unpacking-non-sequence
    assert bits.shape == recipes.shape == (2, shot, 3)
