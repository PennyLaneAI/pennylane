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
"""
Unit tests for the StronglyEntanglingLayers template.
"""
import numpy as np

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import ops as qml_ops
from pennylane.capture.autograph import run_autograph
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
@pytest.mark.external
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    n_wires = 2
    weight_shape = (1, 2, 3)

    weights = np.random.random(size=weight_shape)

    op = qml.StronglyEntanglingLayers(weights, wires=range(n_wires))

    qml.ops.functions.assert_valid(op)


@pytest.mark.parametrize("batch_dim", [None, 1, 3])
class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    QUEUES = [
        (1, (1, 1, 3), ["Rot"], [[0]]),
        (2, (1, 2, 3), ["Rot", "Rot", "CNOT", "CNOT"], [[0], [1], [0, 1], [1, 0]]),
        (
            2,
            (2, 2, 3),
            ["Rot", "Rot", "CNOT", "CNOT", "Rot", "Rot", "CNOT", "CNOT"],
            [[0], [1], [0, 1], [1, 0], [0], [1], [0, 1], [1, 0]],
        ),
        (
            3,
            (1, 3, 3),
            ["Rot", "Rot", "Rot", "CNOT", "CNOT", "CNOT"],
            [[0], [1], [2], [0, 1], [1, 2], [2, 0]],
        ),
    ]

    @pytest.mark.parametrize(
        "n_wires, imprimitive", [(2, qml_ops.CNOT), (3, qml_ops.CZ), (4, qml_ops.CY)]
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(
        self, n_wires, imprimitive, batch_dim
    ):  # pylint: disable=unused-argument
        """Tests the decomposition rule implemented with the new system."""
        weights = np.random.random(
            size=(1, n_wires, 3),
        )
        op = qml.StronglyEntanglingLayers(weights, wires=range(n_wires), imprimitive=imprimitive)

        for rule in qml.list_decomps(qml.StronglyEntanglingLayers):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        "n_wires, imprimitive", [(2, qml_ops.CNOT), (3, qml_ops.CZ), (4, qml_ops.CY)]
    )
    def test_decomposition_new(
        self, n_wires, imprimitive, batch_dim
    ):  # pylint: disable=unused-argument
        """Tests the decomposition rule implemented with the new system."""
        weights = np.random.random(
            size=(1, n_wires, 3),
        )
        op = qml.StronglyEntanglingLayers(weights, wires=range(n_wires), imprimitive=imprimitive)

        for rule in qml.list_decomps(qml.StronglyEntanglingLayers):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("n_wires, weight_shape, expected_names, expected_wires", QUEUES)
    def test_expansion(self, n_wires, weight_shape, expected_names, expected_wires, batch_dim):
        """Checks the queue for the default settings."""
        # pylint: disable=too-many-arguments

        if batch_dim is not None:
            weight_shape = (batch_dim,) + weight_shape
        weights = np.random.random(size=weight_shape)

        op = qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
        tape = qml.tape.QuantumScript(op.decomposition())

        if batch_dim is None:
            param_sets = iter(weights.reshape((-1, 3)))
        else:
            param_sets = iter(weights.reshape((batch_dim, -1, 3)).transpose([1, 2, 0]))

        for i, gate in enumerate(tape.operations):
            assert gate.name == expected_names[i]
            if gate.name == "Rot":
                assert gate.batch_size == batch_dim
                assert qml.math.allclose(gate.data, next(param_sets))
            else:
                assert gate.batch_size is None
            assert gate.wires.labels == tuple(expected_wires[i])

    @pytest.mark.parametrize("n_layers, n_wires", [(2, 2), (1, 3), (2, 4)])
    def test_uses_correct_imprimitive(self, n_layers, n_wires, batch_dim):
        """Test that correct number of entanglers are used in the circuit."""

        shape = (n_layers, n_wires, 3)
        if batch_dim is not None:
            shape = (batch_dim,) + shape
        weights = np.random.randn(*shape)

        op = qml.StronglyEntanglingLayers(weights=weights, wires=range(n_wires), imprimitive=qml.CZ)
        ops = op.decomposition()

        gate_names = [gate.name for gate in ops]
        assert gate_names.count("CZ") == n_wires * n_layers

    def test_custom_wire_labels(self, tol, batch_dim):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        shape = (1, 3, 3) if batch_dim is None else (batch_dim, 1, 3, 3)
        weights = np.random.random(size=shape)

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.StronglyEntanglingLayers(weights, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.StronglyEntanglingLayers(weights, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_layers, n_wires, ranges", [(2, 2, [1, 1]), (1, 3, [2]), (4, 4, [2, 3, 1, 3])]
    )
    def test_custom_range_sequence(self, n_layers, n_wires, ranges, batch_dim):
        """Test that correct sequence of custom ranges are used in the circuit."""

        shape = (n_layers, n_wires, 3)
        if batch_dim is not None:
            shape = (batch_dim,) + shape
        weights = np.random.randn(*shape)

        op = qml.StronglyEntanglingLayers(weights=weights, wires=range(n_wires), ranges=ranges)
        ops = op.decomposition()

        gate_wires = [gate.wires.labels for gate in ops]
        range_idx = 0
        for idx, i in enumerate(gate_wires):
            if idx % (n_wires * 2) // n_wires == 1:
                expected_wire = (
                    idx % n_wires,
                    (ranges[range_idx % len(ranges)] + idx % n_wires) % n_wires,
                )
                assert i == expected_wire
                if idx % n_wires == n_wires - 1:
                    range_idx += 1


@pytest.mark.jax
@pytest.mark.capture
# pylint:disable=protected-access
class TestDynamicDecomposition:
    """Tests that dynamic decomposition via compute_qfunc_decomposition works correctly."""

    def test_strongly_entangling_plxpr(self):
        """Test that the dynamic decomposition of StronglyEntanglingLayer has the correct plxpr"""
        import jax
        from jax import numpy as jnp

        from pennylane.capture.primitives import cond_prim, for_loop_prim
        from pennylane.tape.plxpr_conversion import CollectOpsandMeas
        from pennylane.transforms.decompose import DecomposeInterpreter

        layers = 5
        n_wires = 3
        gate_set = None
        imprimitive = qml.CNOT
        max_expansion = 1

        weight_shape = (layers, n_wires, 3)
        weights = np.random.random(size=weight_shape)
        wires = list(range(n_wires))

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        def circuit(weights, wires):
            qml.StronglyEntanglingLayers(weights, wires=wires, imprimitive=imprimitive)
            return qml.state()

        jaxpr = jax.make_jaxpr(circuit)(weights, wires=wires)

        # Validate Jaxpr
        jaxpr_eqns = jaxpr.eqns
        layer_loop_eqn = [eqn for eqn in jaxpr_eqns if eqn.primitive == for_loop_prim]
        assert layer_loop_eqn[0].primitive == for_loop_prim
        layer_inner_eqn = layer_loop_eqn[0].params["jaxpr_body_fn"].eqns

        rot_loop_eqn = [eqn for eqn in layer_inner_eqn if eqn.primitive == for_loop_prim]
        assert rot_loop_eqn[0].primitive == for_loop_prim
        rot_inner_eqn = rot_loop_eqn[0].params["jaxpr_body_fn"].eqns
        assert rot_inner_eqn[-1].primitive == qml.Rot._primitive

        cond_eqn = [eqn for eqn in layer_inner_eqn if eqn.primitive == cond_prim]
        assert cond_eqn[0].primitive == cond_prim
        true_branch_eqns = cond_eqn[0].params["jaxpr_branches"][0].eqns
        false_branch_eqns = cond_eqn[0].params["jaxpr_branches"][1].eqns
        assert false_branch_eqns == []

        imprimitive_loop_eqn = [eqn for eqn in true_branch_eqns if eqn.primitive == for_loop_prim]
        assert imprimitive_loop_eqn[0].primitive == for_loop_prim
        imprimitive_inner_eqn = imprimitive_loop_eqn[0].params["jaxpr_body_fn"].eqns
        assert imprimitive_inner_eqn[-1].primitive == imprimitive._primitive

        # Validate Ops
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, weights, *wires)
        ops_list = collector.state["ops"]
        tape = qml.tape.QuantumScript(
            [qml.StronglyEntanglingLayers(jnp.array(weights), wires=wires, imprimitive=imprimitive)]
        )
        [decomp_tape], _ = qml.transforms.decompose(
            tape, max_expansion=max_expansion, gate_set=gate_set
        )
        assert ops_list == decomp_tape.operations

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize(
        "n_layers, n_wires, ranges",
        [(2, 2, [1, 1]), (1, 3, [2]), (4, 4, [2, 3, 1, 3]), (4, 4, None)],
    )
    @pytest.mark.parametrize("imprimitive", [qml.CNOT, qml.CZ, None])
    @pytest.mark.parametrize("max_expansion", [1, 2, 3, 4, 5, None])
    @pytest.mark.parametrize(
        "gate_set", [[qml.RX, qml.RY, qml.RZ, qml.CNOT, qml.GlobalPhase], None]
    )
    def test_strongly_entangling_state(
        self, n_layers, n_wires, ranges, imprimitive, max_expansion, gate_set, autograph
    ):  # pylint:disable=too-many-arguments
        """Test that the StronglyEntanglingLayer gives correct result after dynamic decomposition."""

        from functools import partial

        import jax

        from pennylane.transforms.decompose import DecomposeInterpreter

        weight_shape = (n_layers, n_wires, 3)
        weights = np.random.random(size=weight_shape)
        wires = list(range(n_wires))

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        @qml.qnode(device=qml.device("default.qubit", wires=n_wires))
        def circuit(weights, wires):
            qml.StronglyEntanglingLayers(
                weights, wires=wires, ranges=ranges, imprimitive=imprimitive
            )
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(circuit)(weights, wires=wires)
        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, weights, *wires)

        with qml.capture.pause():

            @partial(qml.transforms.decompose, max_expansion=max_expansion, gate_set=gate_set)
            @qml.qnode(device=qml.device("default.qubit", wires=n_wires))
            def circuit_comparison():
                qml.StronglyEntanglingLayers(
                    weights, wires=range(n_wires), ranges=ranges, imprimitive=imprimitive
                )
                return qml.state()

            result_comparison = circuit_comparison()

        assert qml.math.allclose(*result, result_comparison)


class TestInputs:
    """Test inputs and pre-processing."""

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights, ranges=None):
            qml.StronglyEntanglingLayers(weights, wires=range(2), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Weights tensor must have second dimension"):
            weights = np.random.randn(2, 1, 3)
            circuit(weights)

        with pytest.raises(ValueError, match="Weights tensor must have third dimension"):
            weights = np.random.randn(2, 2, 1)
            circuit(weights)

        with pytest.raises(ValueError, match="Range sequence must be of length"):
            weights = np.random.randn(2, 2, 3)
            circuit(weights, ranges=[1])

    def test_exception_wrong_ranges(self):
        """Verifies that exception is raised if the
        value of ranges is incorrect."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights, ranges=None):
            qml.StronglyEntanglingLayers(weights, wires=range(2), ranges=ranges)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Ranges must not be zero nor"):
            weights = np.random.randn(1, 2, 3)
            circuit(weights, ranges=[0])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.StronglyEntanglingLayers(np.array([[[1, 2, 3]]]), wires=[0], id="a")
        assert template.id == "a"


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 3, (2, 3, 3)),
            (2, 1, (2, 1, 3)),
            (2, 2, (2, 2, 3)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.StronglyEntanglingLayers.shape(n_layers, n_wires)
        assert shape == expected_shape


def circuit_template(weights):
    qml.StronglyEntanglingLayers(weights, range(3))
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    qml.Rot(weights[0, 0, 0], weights[0, 0, 1], weights[0, 0, 2], wires=0)
    qml.Rot(weights[0, 1, 0], weights[0, 1, 1], weights[0, 1, 2], wires=1)
    qml.Rot(weights[0, 2, 0], weights[0, 2, 1], weights[0, 2, 2], wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 3, 3))
        weights = pnp.array(weights, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 3, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 3, 3)))

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(1, 3, 3)), requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
