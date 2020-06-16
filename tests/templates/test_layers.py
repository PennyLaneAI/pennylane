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
Unit tests for the :mod:`pennylane.template.layers` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import (
    CVNeuralNetLayers,
    StronglyEntanglingLayers,
    RandomLayers,
    BasicEntanglerLayers,
    SimplifiedTwoDesign,
)
from pennylane.templates.layers.random import random_layer
from pennylane import RX, RY, RZ, CZ, CNOT
from pennylane.wires import Wires

TOLERANCE = 1e-8


class TestCVNeuralNet:
    """Tests for the CVNeuralNet from the pennylane.template module."""

    # Have a fixed number of subsystems in this handcoded test
    @pytest.fixture(scope="class")
    def num_subsystems(self):
        return 4

    @pytest.fixture(scope="class")
    def weights(self):
        return [
            np.array(
                [
                    [5.48791879, 6.08552046, 5.46131036, 3.33546468, 1.46227521, 0.0716208],
                    [3.36869403, 0.63074883, 4.59400392, 5.9040016, 5.92704296, 2.35455147],
                ]
            ),
            np.array(
                [
                    [2.70471535, 2.52804815, 3.28406182, 3.0058243, 3.48940764, 3.41419504],
                    [3.74320919, 4.15936005, 3.20807161, 2.95870535, 0.05574621, 0.42660569],
                ]
            ),
            np.array(
                [
                    [4.7808479, 4.47598146, 3.89357744, 2.67721355],
                    [2.73203094, 2.71115444, 1.16794164, 3.32823666],
                ]
            ),
            np.array(
                [
                    [0.27344502, 0.68431314, 0.30026443, 0.23128064],
                    [0.45945175, 0.53255468, 0.28383751, 0.34263728],
                ]
            ),
            np.array(
                [
                    [2.3936353, 4.80135971, 5.89867895, 2.00867023],
                    [5.14552399, 3.31578667, 5.90119363, 4.54515204],
                ]
            ),
            np.array(
                [
                    [0.4134863, 6.17555778, 0.80334114, 2.02400747, 0.44574704, 1.41227118],
                    [5.16969442, 3.6890488, 4.43916808, 3.20808287, 5.21543123, 4.52815349],
                ]
            ),
            np.array(
                [
                    [2.47328111, 5.63064513, 2.17059932, 6.1873632, 0.18052879, 2.20970037],
                    [5.44288268, 1.27806129, 1.87574979, 2.98956484, 3.10140853, 3.81814174],
                ]
            ),
            np.array(
                [
                    [5.03318258, 4.01017269, 0.43159284, 3.7928101],
                    [3.5329307, 4.79661266, 5.0683084, 1.87631749],
                ]
            ),
            np.array(
                [
                    [1.61159166, 0.1608155, 0.96535086, 1.60132783],
                    [0.36293094, 1.30725604, 0.11578591, 1.5983082],
                ]
            ),
            np.array(
                [
                    [6.21267547, 3.71076099, 0.34060195, 2.86031556],
                    [3.20443756, 6.26536946, 6.18450567, 1.50406923],
                ]
            ),
            np.array(
                [
                    [0.1376345, 0.22541113, 0.14306356, 0.13019402],
                    [0.26999146, 0.26256351, 0.14722687, 0.23137066],
                ]
            ),
        ]

    def test_cvneuralnet_uses_correct_weights(self, weights):
        """Tests that the CVNeuralNetLayers template uses the weigh parameters correctly."""

        with qml.utils.OperationRecorder() as rec:
            CVNeuralNetLayers(*weights, wires=range(4))

        # Test that gates appear in the right order for each layer:
        # BS-R-S-BS-R-D-K
        for l in range(2):
            gates = [
                qml.Beamsplitter,
                qml.Rotation,
                qml.Squeezing,
                qml.Beamsplitter,
                qml.Rotation,
                qml.Displacement,
            ]

            # count the position of each group of gates in the layer
            num_gates_per_type = [0, 6, 4, 4, 6, 4, 4, 4]
            s = np.cumsum(num_gates_per_type)
            gc = l * sum(num_gates_per_type) + np.array(list(zip(s[:-1], s[1:])))

            # loop through expected gates
            for idx, g in enumerate(gates):
                # loop through where these gates should be in the queue
                for opidx, op in enumerate(rec.queue[gc[idx, 0] : gc[idx, 1]]):
                    # check that op in queue is correct gate
                    assert isinstance(op, g)

                    # test that the parameters are correct
                    res_params = op.parameters

                    if idx == 0:
                        # first BS
                        exp_params = [weights[0][l][opidx], weights[1][l][opidx]]
                    elif idx == 1:
                        # first rot
                        exp_params = [weights[2][l][opidx]]
                    elif idx == 2:
                        # squeezing
                        exp_params = [weights[3][l][opidx], weights[4][l][opidx]]
                    elif idx == 3:
                        # second BS
                        exp_params = [weights[5][l][opidx], weights[6][l][opidx]]
                    elif idx == 4:
                        # second rot
                        exp_params = [weights[7][l][opidx]]
                    elif idx == 5:
                        # displacement
                        exp_params = [weights[8][l][opidx], weights[9][l][opidx]]

                    assert res_params == exp_params

    def test_cvqnn_layers_exception_nlayers(self, gaussian_device_4modes):
        """Integration test for the CVNeuralNetLayers method."""

        def circuit(weights):
            CVNeuralNetLayers(*weights, wires=range(4))
            return qml.expval(qml.X(0))

        qnode = qml.QNode(circuit, gaussian_device_4modes)

        wrong_weights = [np.array([1]) if i < 10 else np.array([1, 1]) for i in range(11)]
        with pytest.raises(ValueError, match="the first dimension of the weight parameters"):
            qnode(wrong_weights)


class TestStronglyEntangling:
    """Tests for the StronglyEntanglingLayers method from the pennylane.templates.layers module."""

    @pytest.mark.parametrize("n_layers", range(1, 4))
    def test_single_qubit(self, n_layers):
        weights = np.zeros((n_layers, 1, 3))
        with qml.utils.OperationRecorder() as rec:
            StronglyEntanglingLayers(weights, wires=range(1))

        assert len(rec.queue) == n_layers
        assert all([isinstance(q, qml.Rot) for q in rec.queue])
        assert all([q._wires[0] == 0 for q in rec.queue]) #Wires(0) for q in rec.queue])

    def test_strong_ent_layers_uses_correct_weights(self, n_subsystems):
        """Test that StronglyEntanglingLayers uses the correct weights in the circuit."""
        np.random.seed(12)
        n_layers = 2
        num_wires = n_subsystems

        weights = np.random.randn(n_layers, num_wires, 3)

        with qml.utils.OperationRecorder() as rec:
            StronglyEntanglingLayers(weights, wires=range(num_wires))

        # Test that gates appear in the right order
        exp_gates = [qml.Rot] * num_wires + [qml.CNOT] * num_wires
        exp_gates *= n_layers
        res_gates = rec.queue

        for op1, op2 in zip(res_gates, exp_gates):
            assert isinstance(op1, op2)

        # test the device parameters
        for l in range(n_layers):

            layer_ops = rec.queue[2 * l * num_wires : 2 * (l + 1) * num_wires]

            # check each rotation gate parameter
            for n in range(num_wires):
                res_params = layer_ops[n].parameters
                exp_params = weights[l, n, :]
                assert sum([r == e for r, e in zip(res_params, exp_params)])

    def test_strong_ent_layers_uses_correct_number_of_imprimitives(self, n_layers, n_subsystems):
        """Test that StronglyEntanglingLayers uses the correct number of imprimitives."""
        imprimitive = CZ
        weights = np.random.randn(n_layers, n_subsystems, 3)

        with qml.utils.OperationRecorder() as rec:
            StronglyEntanglingLayers(
                weights=weights, wires=range(n_subsystems), imprimitive=imprimitive
            )

        types = [type(q) for q in rec.queue]
        assert types.count(imprimitive) == n_subsystems * n_layers

    @pytest.mark.parametrize("n_wires, n_layers, ranges", [(2, 2, [2, 1]), (3, 1, [5])])
    def test_strong_ent_layers_ranges_equals_wires_exception(self, n_layers, n_wires, ranges):
        """Test that StronglyEntanglingLayers throws and exception if a range is equal to or
        larger than the number of wires."""
        dev = qml.device("default.qubit", wires=n_wires)
        weights = np.random.randn(n_layers, n_wires, 3)

        def circuit(weights):
            StronglyEntanglingLayers(weights=weights, wires=range(n_wires), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match="the range for all layers needs to be smaller than"):
            qnode(weights)

    def test_strong_ent_layers_illegal_ranges_exception(self):
        """Test that StronglyEntanglingLayers throws and exception if ``ranges`` parameter of illegal type."""
        n_wires = 2
        n_layers = 2
        dev = qml.device("default.qubit", wires=n_wires)
        weights = np.random.randn(n_layers, n_wires, 3)

        def circuit(weights):
            StronglyEntanglingLayers(weights=weights, wires=range(n_wires), ranges=["a", "a"])
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match="'ranges' must be a list of integers"):
            qnode(weights)

    @pytest.mark.parametrize("n_layers, ranges", [(2, [1, 2, 4]), (5, [2])])
    def test_strong_ent_layers_wrong_size_ranges_exception(self, n_layers, ranges):
        """Test that StronglyEntanglingLayers throws and exception if ``ranges`` parameter
        not of shape (len(wires),)."""
        n_wires = 5
        dev = qml.device("default.qubit", wires=n_wires)
        weights = np.random.randn(n_layers, n_wires, 3)

        def circuit(weights):
            StronglyEntanglingLayers(weights=weights, wires=range(n_wires), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match="'ranges' must be of shape"):
            qnode(weights)


class TestRandomLayers:
    """Tests for the RandomLayers method from the pennylane.templates module."""

    @pytest.fixture(scope="class", params=[0.2, 0.6])
    def ratio(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[CNOT, CZ])
    def impr(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[[RX], [RY, RZ]])
    def rots(self, request):
        return request.param

    def test_same_circuit_for_same_seed(self, tol, seed):
        """Test that RandomLayers() creates the same circuit when using the same seed."""
        dev = qml.device("default.qubit", wires=2)
        weights = [[0.1, 0.2, 0.3]]

        def circuit(weights):
            RandomLayers(weights=weights, wires=range(2), seed=seed)
            return qml.expval(qml.PauliZ(0))

        qnode1 = qml.QNode(circuit, dev)
        qnode2 = qml.QNode(circuit, dev)
        assert np.allclose(qnode1(weights), qnode2(weights), atol=tol)

    def test_different_circuits_for_different_seeds(self, tol):
        """Test that RandomLayers() does not necessarily have the same output for two different seeds."""
        dev = qml.device("default.qubit", wires=2)
        weights = [[0.1, 0.2, 0.3]]

        def circuit1(weights):
            RandomLayers(weights=weights, wires=range(2), seed=10)
            return qml.expval(qml.PauliZ(0))

        def circuit2(weights):
            RandomLayers(weights=weights, wires=range(2), seed=20)
            return qml.expval(qml.PauliZ(0))

        qnode1 = qml.QNode(circuit1, dev)
        qnode2 = qml.QNode(circuit2, dev)

        assert not np.allclose(qnode1(weights), qnode2(weights), atol=tol)

    @pytest.mark.parametrize("mutable", [True, False])
    def test_same_circuit_in_each_qnode_call(self, mutable, tol):
        """Test that RandomLayers() creates the same circuit in two calls of a qnode."""
        dev = qml.device("default.qubit", wires=2)
        weights = [[0.1, 0.2, 0.3]]

        @qml.qnode(dev, mutable=mutable)
        def circuit(weights):
            RandomLayers(weights=weights, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        first_call = circuit(weights)
        second_call = circuit(weights)
        assert np.allclose(first_call, second_call, atol=tol)

    def test_no_seed(self, tol):
        """Test that two calls to a qnode with RandomLayers() for 'seed=None' option create the
        same circuit for immutable qnodes."""

        dev = qml.device("default.qubit", wires=2)
        weights = [[0.1] * 100]

        @qml.qnode(dev, mutable=False)
        def circuit(weights):
            RandomLayers(weights=weights, wires=range(2), seed=None)
            return qml.expval(qml.PauliZ(0))

        first_call = circuit(weights)
        second_call = circuit(weights)
        assert np.allclose(first_call, second_call, atol=tol)

    def test_random_layers_nlayers(self, n_layers):
        """Test that RandomLayers() picks the correct number of gates."""
        np.random.seed(12)
        n_rots = 1
        n_wires = 2
        impr = CNOT
        weights = np.random.randn(n_layers, n_rots)

        with qml.utils.OperationRecorder() as rec:
            RandomLayers(weights=weights, wires=range(n_wires))

        types = [type(q) for q in rec.queue]
        assert len(types) - types.count(impr) == n_layers

    def test_random_layer_ratio_imprimitive(self, ratio):
        """Test that  random_layer() has the right ratio of imprimitive gates."""
        n_rots = 500
        n_wires = 2
        impr = CNOT
        weights = np.random.randn(n_rots)

        with qml.utils.OperationRecorder() as rec:
            random_layer(
                weights=weights,
                wires=Wires(range(n_wires)),
                ratio_imprim=ratio,
                imprimitive=CNOT,
                rotations=[RX, RY, RZ],
                seed=42,
            )

        types = [type(q) for q in rec.queue]
        ratio_impr = types.count(impr) / len(types)
        assert np.isclose(ratio_impr, ratio, atol=0.05)

    def test_random_layer_gate_types(self, n_subsystems, impr, rots):
        """Test that  random_layer() uses the correct types of gates."""
        n_rots = 20
        weights = np.random.randn(n_rots)

        with qml.utils.OperationRecorder() as rec:
            random_layer(
                weights=weights,
                wires=Wires(range(n_subsystems)),
                ratio_imprim=0.3,
                imprimitive=impr,
                rotations=rots,
                seed=42,
            )

        types = [type(q) for q in rec.queue]
        unique = set(types)
        gates = {impr, *rots}
        assert unique == gates

    def test_random_layer_numgates(self, n_subsystems):
        """Test that random_layer() uses the correct number of gates."""
        n_rots = 5
        weights = np.random.randn(n_rots)

        with qml.utils.OperationRecorder() as rec:
            random_layer(
                weights=weights,
                wires=Wires(range(n_subsystems)),
                ratio_imprim=0.3,
                imprimitive=qml.CNOT,
                rotations=[RX, RY, RZ],
                seed=42,
            )

        types = [type(q) for q in rec.queue]
        assert len(types) - types.count(qml.CNOT) == n_rots

    def test_random_layer_randomwires(self, n_subsystems):
        """Test that  random_layer() picks random wires."""
        n_rots = 500
        weights = np.random.randn(n_rots)

        with qml.utils.OperationRecorder() as rec:
            random_layer(
                weights=weights,
                wires=Wires(range(n_subsystems)),
                ratio_imprim=0.3,
                imprimitive=qml.CNOT,
                rotations=[RX, RY, RZ],
                seed=42,
            )

        wires = [q._wires for q in rec.queue]
        wires_flat = [item for w in wires for item in w]
        mean_wire = np.mean(wires_flat)
        assert np.isclose(mean_wire, (n_subsystems - 1) / 2, atol=0.05)

    def test_random_layer_weights(self, n_subsystems, tol):
        """Test that random_layer() uses the correct weights."""
        np.random.seed(12)
        n_rots = 5
        weights = np.random.randn(n_rots)

        with qml.utils.OperationRecorder() as rec:
            random_layer(
                weights=weights,
                wires=Wires(range(n_subsystems)),
                ratio_imprim=0.3,
                imprimitive=qml.CNOT,
                rotations=[RX, RY, RZ],
                seed=4,
            )

        params = [q.parameters for q in rec.queue]
        params_flat = [item for p in params for item in p]
        assert np.allclose(weights.flatten(), params_flat, atol=tol)


class TestSimplifiedTwoDesign:
    """Tests for the SimplifiedTwoDesign method from the pennylane.templates.layers module."""

    @pytest.mark.parametrize(
        "n_wires, n_layers, shape_weights",
        [(1, 2, (0,)), (2, 2, (2, 1, 2)), (3, 2, (2, 2, 2)), (4, 2, (2, 3, 2))],
    )
    def test_circuit_queue(self, n_wires, n_layers, shape_weights):
        """Tests the gate types in the circuit."""
        np.random.seed(42)
        initial_layer = np.random.randn(n_wires)
        weights = np.random.randn(*shape_weights)

        with qml.utils.OperationRecorder() as rec:
            SimplifiedTwoDesign(initial_layer, weights, wires=range(n_wires))

        # Test that gates appear in the right order
        exp_gates = [qml.CZ, qml.RY, qml.RY] * ((n_wires // 2) + (n_wires - 1) // 2)
        exp_gates *= n_layers
        exp_gates = [qml.RY] * n_wires + exp_gates

        res_gates = rec.queue

        for op1, op2 in zip(res_gates, exp_gates):
            assert isinstance(op1, op2)

    @pytest.mark.parametrize(
        "n_wires, n_layers, shape_weights",
        [(1, 2, (0,)), (2, 2, (2, 1, 2)), (3, 2, (2, 2, 2)), (4, 2, (2, 3, 2))],
    )
    def test_circuit_parameters(self, n_wires, n_layers, shape_weights):
        """Tests the parameter values in the circuit."""
        np.random.seed(42)
        initial_layer = np.random.randn(n_wires)
        weights = np.random.randn(*shape_weights)

        with qml.utils.OperationRecorder() as rec:
            SimplifiedTwoDesign(initial_layer, weights, wires=range(n_wires))

        # test the device parameters
        for l in range(n_layers):
            # only select the rotation gates
            ops = [gate for gate in rec.queue if isinstance(gate, qml.RY)]

            # check each initial_layer gate parameters
            for n in range(n_wires):
                res_param = ops[n].parameters[0]
                exp_param = initial_layer[n]
                assert res_param == exp_param

            # check layer gate parameters
            ops = ops[n_wires:]
            exp_params = weights.flatten()
            for o, exp_param in zip(ops, exp_params):
                res_param = o.parameters[0]
                assert res_param == exp_param

    @pytest.mark.parametrize(
        "initial_layer_weights, weights, n_wires, target",
        [
            ([np.pi], [], 1, [-1]),
            ([np.pi] * 2, [[[np.pi] * 2]], 2, [1, 1]),
            ([np.pi] * 3, [[[np.pi] * 2] * 2], 3, [1, -1, 1]),
            ([np.pi] * 4, [[[np.pi] * 2] * 3], 4, [1, -1, -1, 1]),
        ],
    )
    def test_correct_target_output(self, initial_layer_weights, weights, n_wires, target):
        """Tests the result of the template for simple cases."""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(initial_layer, weights):
            SimplifiedTwoDesign(
                initial_layer_weights=initial_layer, weights=weights, wires=range(n_wires)
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        expectations = circuit(initial_layer_weights, weights)
        for exp, target_exp in zip(expectations, target):
            assert pytest.approx(exp, target_exp, abs=TOLERANCE)


class TestBasicEntangler:
    """Tests for the BasicEntanglerLayers method from the pennylane.templates.layers module."""

    @pytest.mark.parametrize("n_wires, n_cnots", [(1, 0), (2, 1), (3, 3), (4, 4)])
    def test_circuit_queue(self, n_wires, n_cnots):
        """Tests the gate types in the circuit."""
        np.random.seed(42)
        n_layers = 2

        weights = np.random.randn(n_layers, n_wires)

        with qml.utils.OperationRecorder() as rec:
            BasicEntanglerLayers(weights, wires=range(n_wires))

        # Test that gates appear in the right order
        exp_gates = [qml.RX] * n_wires + [qml.CNOT] * n_cnots
        exp_gates *= n_layers
        res_gates = rec.queue

        for op1, op2 in zip(res_gates, exp_gates):
            assert isinstance(op1, op2)

    @pytest.mark.parametrize("n_wires, n_cnots", [(1, 0), (2, 1), (3, 3), (4, 4)])
    def test_circuit_parameters(self, n_wires, n_cnots):
        """Tests the parameter values in the circuit."""
        np.random.seed(42)
        n_layers = 2

        weights = np.random.randn(n_layers, n_wires)

        with qml.utils.OperationRecorder() as rec:
            BasicEntanglerLayers(weights, wires=range(n_wires))

        # test the device parameters
        for l in range(n_layers):
            # only select the rotation gates
            layer_ops = rec.queue[l * (n_wires + n_cnots) : l * (n_wires + n_cnots) + n_wires]

            # check each rotation gate parameter
            for n in range(n_wires):
                res_param = layer_ops[n].parameters[0]
                exp_param = weights[l, n]
                assert res_param == exp_param

    @pytest.mark.parametrize("rotation", [RX, RY, RZ])
    def test_custom_rotation(self, rotation):
        """Tests that non-default rotation gates are used correctly."""
        n_layers = 2
        n_wires = 4
        weights = np.ones(shape=(n_layers, n_wires))

        with qml.utils.OperationRecorder() as rec:
            BasicEntanglerLayers(weights, wires=range(n_wires), rotation=rotation)

        # assert queue contains the custom rotations and CNOTs only
        gates = rec.queue
        for op in gates:
            if not isinstance(op, CNOT):
                assert isinstance(op, rotation)

    @pytest.mark.parametrize(
        "weights, n_wires, target",
        [
            ([[np.pi]], 1, [-1]),
            ([[np.pi] * 2], 2, [-1, 1]),
            ([[np.pi] * 3], 3, [1, 1, -1]),
            ([[np.pi] * 4], 4, [-1, 1, -1, 1]),
        ],
    )
    def test_simple_target_outputs(self, weights, n_wires, target):
        """Tests the result of the template for simple cases."""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(weights=weights, wires=range(n_wires), rotation=RX)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        expectations = circuit(weights)
        for exp, target_exp in zip(expectations, target):
            assert exp == target_exp

    def test_interactions_type(self):
        """Tests that an error is raised when 'interactions' is not a list"""

        n_wires = 3
        weights = [[1, 2, 1]]
        interactions = (0, 1, 2)

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(
                weights=weights, wires=range(n_wires), rotation=RX, interactions=interactions
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as info:
            output = circuit(weights)
        assert "'interactions' must be list of wire index pairs" in str(info.value)

    def test_interactions_shape(self):
        """Tests that an error is raised when the shape of elements in 'interactions' is not (2,)"""

        n_wires = 3
        weights = [[1, 2, 1]]
        interactions = [1, 1, 1]

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(
                weights=weights, wires=range(n_wires), rotation=RX, interactions=interactions
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as info:
            output = circuit(weights)
        assert "Elements of 'interactions' must be of shape (2,)" in str(info.value)

    def test_interactions_same_wires(self):
        """Tests that an error is raised when a 'interactions' contains a pair of the form [a, b] with a = b"""

        n_wires = 3
        weights = [[1, 2, 1]]
        interactions = [[0, 1], [1, 2], [1, 1]]

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(
                weights=weights, wires=range(n_wires), rotation=RX, interactions=interactions
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as info:
            output = circuit(weights)
        assert "CNOT gates must be applied between two different wires" in str(info.value)

    @pytest.mark.parametrize(
        "interactions, index",
        [
            ([[0, 1], [1, 2], [2, 3]], [2, 3]),
            ([[5, 1], [1, 2]], [5, 1]),
            ([[5, 4], [1, 2], [2, 1]], [5, 4]),
        ],
    )
    def test_interactions_range(self, interactions, index):
        """Tests that an error is raised when a 'interactions' contains a pair of wires that is out of range"""

        n_wires = 3
        weights = [[1, 2, 1]]

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(
                weights=weights, wires=range(n_wires), rotation=RX, interactions=interactions
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as info:
            output = circuit(weights)
        assert "Wire index pair {} is out of range".format(index) in str(info.value)

    @pytest.mark.parametrize(
        "weights, n_wires, target, interactions",
        [
            ([[np.pi] * 2], 2, [-1, 1], [[0, 1]]),
            ([[np.pi] * 3], 3, [-1, 1, 1], [[0, 1], [0, 2]]),
            ([[np.pi] * 4], 4, [-1, 1, -1, 1], [[0, 1], [2, 3]]),
        ],
    )
    def test_simple_interactions(self, weights, n_wires, target, interactions):
        """Tests the result of the template for simple cases, with 'interactions' != None."""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            BasicEntanglerLayers(
                weights=weights, wires=range(n_wires), rotation=RX, interactions=interactions
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        expectations = circuit(weights)
        for exp, target_exp in zip(expectations, target):
            assert exp == target_exp