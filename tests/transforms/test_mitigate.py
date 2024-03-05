# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for mitigation transforms.
"""
# pylint:disable=no-self-use
from functools import partial
import pytest

from packaging import version

import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumScript
from pennylane.transforms import mitigate_with_zne, richardson_extrapolate, fold_global

with qml.queuing.AnnotatedQueue() as q_tape:
    qml.BasisState([1], wires=0)
    qml.RX(0.9, wires=0)
    qml.RY(0.4, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=0)
    qml.RX(0.6, wires=1)
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

tape = QuantumScript.from_queue(q_tape)
with qml.queuing.AnnotatedQueue() as q_tape_base:
    qml.RX(0.9, wires=0)
    qml.RY(0.4, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=0)
    qml.RX(0.6, wires=1)


tape_base = QuantumScript.from_queue(q_tape_base)


def same_tape(tape1, tape2):
    """Raises an error if tapes are not identical"""
    assert all(o1.name == o2.name for o1, o2 in zip(tape1.operations, tape2.operations))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape1.operations, tape2.operations))
    assert all(
        np.allclose(o1.parameters, o2.parameters)
        for o1, o2 in zip(tape1.operations, tape2.operations)
    )
    assert len(tape1.measurements) == len(tape2.measurements)
    assert all(
        m1.return_type == m2.return_type for m1, m2 in zip(tape1.measurements, tape2.measurements)
    )
    assert all(o1.name == o2.name for o1, o2 in zip(tape1.observables, tape2.observables))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape1.observables, tape2.observables))


class TestMitigateWithZNE:
    """Tests for the mitigate_with_zne function"""

    folding = lambda *args, **kwargs: tape_base
    extrapolate = lambda *args, **kwargs: [3.141]

    def test_folding_call(self, mocker):
        """Tests that arguments are passed to the folding function as expected"""
        spy = mocker.spy(self, "folding")
        scale_factors = [1, 2, -4]
        folding_kwargs = {"Hello": "goodbye"}

        mitigate_with_zne(tape, scale_factors, self.folding, self.extrapolate, folding_kwargs)

        args = spy.call_args_list

        for i in range(3):
            same_tape(args[i][0][0], tape_base)
        assert [args[i][0][1] for i in range(3)] == scale_factors
        assert all(args[i][1] == folding_kwargs for i in range(3))

    def test_extrapolate_call(self, mocker):
        """Tests that arguments are passed to the extrapolate function as expected"""
        spy = mocker.spy(self, "extrapolate")
        scale_factors = [1, 2, -4]
        random_results = [0.1, 0.2, 0.3]
        extrapolate_kwargs = {"Hello": "goodbye"}

        tapes, fn = mitigate_with_zne(
            tape,
            scale_factors,
            self.folding,
            self.extrapolate,
            extrapolate_kwargs=extrapolate_kwargs,
        )
        res = fn(random_results)
        assert res == 3.141

        args = spy.call_args
        assert args[0][0] == scale_factors
        assert np.allclose(args[0][1], random_results)

        assert args[1] == extrapolate_kwargs

        for t in tapes:
            same_tape(t, tape)

    def test_multi_returns(self):
        """Tests if the expected shape is returned when mitigating a circuit with two returns"""
        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(dev_noise_free, qml.AmplitudeDamping, noise_strength)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_global,
            extrapolate=richardson_extrapolate,
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        # check shapes
        assert isinstance(res_mitigated, tuple)
        assert len(res_mitigated) == 2
        assert all(res.shape == () for res in res_mitigated)

        assert isinstance(res_ideal, tuple)
        assert len(res_ideal) == 2
        assert all(res.shape == () for res in res_ideal)

        res_mitigated = qml.math.stack(res_mitigated)
        res_ideal = qml.math.stack(res_ideal)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_reps_per_factor_not_1(self, mocker):
        """Tests if mitigation proceeds as expected when reps_per_factor is not 1 (default)"""
        scale_factors = [1, 2, -4]
        spy_fold = mocker.spy(self, "folding")
        spy_extrapolate = mocker.spy(self, "extrapolate")
        _, fn = mitigate_with_zne(
            tape, scale_factors, self.folding, self.extrapolate, reps_per_factor=2
        )
        random_results = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        args = spy_fold.call_args_list
        for i in range(6):
            same_tape(args[i][0][0], tape_base)
        assert [args[i][0][1] for i in range(6)] == [1, 1, 2, 2, -4, -4]

        fn(random_results)

        args = spy_extrapolate.call_args

        assert args[0][0] == scale_factors
        assert np.allclose(args[0][1], np.mean(np.reshape(random_results, (3, 2)), axis=1))

    def test_broadcasting(self):
        """Tests that mitigate_with_zne supports batch arguments"""

        batch_size = 2

        @qml.qnode(dev_noisy)
        def original_qnode(inputs):
            qml.AmplitudeEmbedding(features=inputs, wires=range(2), normalize=True)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

        expanded_qnode = qml.transforms.broadcast_expand(original_qnode)

        mitigated_qnode_orig = qml.transforms.mitigate_with_zne(
            original_qnode, [1, 2, 3], fold_global, richardson_extrapolate
        )
        mitigated_qnode_expanded = qml.transforms.mitigate_with_zne(
            expanded_qnode, [1, 2, 3], fold_global, richardson_extrapolate
        )
        rng = np.random.default_rng(seed=18954959)
        inputs = rng.uniform(0, 1, size=(batch_size, 2**2))
        result_orig = mitigated_qnode_orig(inputs)
        result_expanded = mitigated_qnode_expanded(inputs)
        assert qml.math.allclose(result_orig, result_expanded)


@pytest.fixture
def skip_if_no_mitiq_support():
    """Fixture to skip if minimum version of mitiq is not available"""
    try:
        import mitiq

        v = version.parse(mitiq.__version__)
        t = version.parse("0.11.0")
        if v.major < t.major and v.minor < t.minor:
            pytest.skip("Mitiq version too low")
    except ImportError:
        pytest.skip("Mitiq not available")


@pytest.fixture
def skip_if_no_pl_qiskit_support():
    """Fixture to skip if pennylane_qiskit is not available"""
    pytest.importorskip("pennylane_qiskit")


@pytest.mark.usefixtures("skip_if_no_pl_qiskit_support")
@pytest.mark.usefixtures("skip_if_no_mitiq_support")
class TestMitiqIntegration:
    """Tests if the mitigate_with_zne transform is compatible with using mitiq as a backend"""

    def test_multiple_returns(self):
        """Tests if the expected shape is returned when mitigating a circuit with two returns"""
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_global,
            extrapolate=RichardsonFactory.extrapolate,
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        # check shapes
        assert isinstance(res_mitigated, tuple)
        assert len(res_mitigated) == 2
        assert all(res.shape == () for res in res_mitigated)

        assert isinstance(res_ideal, tuple)
        assert len(res_ideal) == 2
        assert all(res.shape == () for res in res_ideal)

        res_mitigated = qml.math.stack(res_mitigated)
        res_ideal = qml.math.stack(res_ideal)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_single_return(self):
        """Tests if the expected shape is returned when mitigating a circuit with a single return"""
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_global,
            extrapolate=RichardsonFactory.extrapolate,
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_with_reps_per_factor(self):
        """Tests if the expected shape is returned when mitigating a circuit with a reps_per_factor
        set not equal to 1"""
        from mitiq.zne.scaling import fold_gates_at_random
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_gates_at_random,
            extrapolate=RichardsonFactory.extrapolate,
            reps_per_factor=2,
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_integration(self):
        """Test if the error of the mitigated result is less than the error of the unmitigated
        result for a circuit with known expectation values"""
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            qml.adjoint(qml.SimplifiedTwoDesign)(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        exact_qnode = qml.QNode(circuit, dev_noise_free)
        noisy_qnode = qml.QNode(circuit, dev)

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_global,
            extrapolate=RichardsonFactory.extrapolate,
        )
        @qml.qnode(dev)
        def mitigated_qnode(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            qml.adjoint(qml.SimplifiedTwoDesign)(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        exact_val = exact_qnode(w1, w2)
        noisy_val = noisy_qnode(w1, w2)
        mitigated_val = mitigated_qnode(w1, w2)

        for res in [exact_val, noisy_val, mitigated_val]:
            assert isinstance(res, tuple)
            assert len(res) == 2
            assert all(r.shape == () for r in res)

        exact_val = qml.math.stack(exact_val)
        noisy_val = qml.math.stack(noisy_val)
        mitigated_val = qml.math.stack(mitigated_val)

        mitigated_err = np.abs(exact_val - mitigated_val)
        noisy_err = np.abs(exact_val - noisy_val)

        assert np.allclose(exact_val, [1, 1])
        assert all(mitigated_err < noisy_err)

    @pytest.mark.xfail(
        reason="Using external tape transforms breaks differentiability",
    )
    def test_grad(self):
        """Tests if the gradient is calculated successfully."""
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
        np.random.seed(0)
        w1, w2 = [np.random.random(s, requires_grad=True) for s in shapes]

        @partial(
            qml.transforms.mitigate_with_zne,
            scale_factors=[1, 2, 3],
            folding=fold_global,
            extrapolate=RichardsonFactory.extrapolate,
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        g = qml.grad(mitigated_circuit)(w1, w2)
        for g_ in g:
            assert not np.allclose(g_, 0)


# qnodes for the diffable ZNE error mitigation
def qfunc(theta):
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    return qml.expval(1 * qml.PauliZ(0) + 2 * qml.PauliZ(1))


def qfunc_multi(theta):
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))


# Describe noise
noise_gate = qml.PhaseDamping

# Load devices
dev_ideal = qml.device("default.mixed", wires=2)
dev_noisy = qml.transforms.insert(dev_ideal, noise_gate, 0.05)

out_ideal = np.sqrt(2) / 2 + np.sqrt(2)
grad_ideal_0 = [-np.sqrt(2) / 2, -np.sqrt(2)]

out_ideal_multi = np.array([np.sqrt(2) / 2, np.sqrt(3) / 2])
grad_ideal_0_multi = np.array([[-np.sqrt(2) / 2, 0], [0, -0.5]])


class TestDifferentiableZNE:
    """Testing differentiable ZNE"""

    def test_global_fold_constant_result(self):
        """Ensuring that the folded circuits always yields the same results."""

        dev = qml.device("default.qubit", wires=5)

        # Select template to use within circuit and generate parameters
        n_layers = 2
        n_wires = 3
        template = qml.SimplifiedTwoDesign
        weights_shape = template.shape(n_layers, n_wires)
        w1, w2 = [np.arange(np.prod(s)).reshape(s) for s in weights_shape]

        dev = qml.device("default.qubit", wires=range(n_wires))

        # This circuit itself produces the identity by construction
        @qml.qnode(dev)
        def circuit(w1, w2):
            template(w1, w2, wires=range(n_wires))
            qml.adjoint(template(w1, w2, wires=range(n_wires)))
            return qml.expval(qml.PauliZ(0))

        res = [
            fold_global(circuit, scale_factor=scale_factor)(w1, w2) for scale_factor in range(1, 5)
        ]
        assert np.allclose(res, 1)

    def test_polyfit(self):
        """Testing the custom diffable _polyfit function"""
        # pylint: disable=protected-access
        x = np.linspace(1, 4, 4)
        y = 3.0 * x**2 + 2.0 * x + 1.0
        coeffs = qml.transforms.mitigate._polyfit(x, y, 2)
        assert qml.math.allclose(qml.math.squeeze(coeffs), [3, 2, 1])

    @pytest.mark.autograd
    def test_diffability_autograd(self):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in autograd"""
        qnode_noisy = qml.QNode(qfunc, dev_noisy)
        qnode_ideal = qml.QNode(qfunc, dev_ideal)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = np.array([np.pi / 4, np.pi / 4], requires_grad=True)

        res = mitigated_qnode(theta)
        assert qml.math.allclose(res, out_ideal, atol=1e-2)
        grad = qml.grad(mitigated_qnode)(theta)
        grad_ideal = qml.grad(qnode_ideal)(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)
        qml.grad(qnode_noisy)(theta)

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_diffability_jax(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in jax"""
        import jax
        import jax.numpy as jnp

        qnode_noisy = qml.QNode(qfunc, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = jnp.array(
            [np.pi / 4, np.pi / 4],
        )

        res = mitigated_qnode(theta)
        assert qml.math.allclose(res, out_ideal, atol=1e-2)
        grad = jax.grad(mitigated_qnode)(theta)
        grad_ideal = jax.grad(qnode_ideal)(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)
        jax.grad(qnode_noisy)(theta)

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_diffability_jaxjit(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in jax-jit"""
        import jax
        import jax.numpy as jnp

        qnode_noisy = qml.QNode(qfunc, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = jax.jit(
            mitigate_with_zne(qnode_noisy, scale_factors, fold_global, richardson_extrapolate)
        )

        theta = jnp.array(
            [np.pi / 4, np.pi / 4],
        )

        res = mitigated_qnode(theta)
        assert qml.math.allclose(res, out_ideal, atol=1e-2)
        grad = jax.grad(mitigated_qnode)(theta)
        grad_ideal = jax.grad(qnode_ideal)(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)
        jax.grad(qnode_noisy)(theta)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_diffability_torch(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in torch"""
        import torch

        qnode_noisy = qml.QNode(qfunc, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = torch.tensor([np.pi / 4, np.pi / 4], requires_grad=True)

        res = mitigated_qnode(theta)

        assert qml.math.allclose(res, out_ideal, atol=1e-2)
        res.backward()
        grad = theta.grad
        theta0 = torch.tensor([np.pi / 4, np.pi / 4], requires_grad=True)
        res_ideal = qnode_ideal(theta0)
        res_ideal.backward()
        grad_ideal = theta0.grad
        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["auto", "tf"])
    def test_diffability_tf(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns the correct gradient in tf"""
        import tensorflow as tf

        qnode_noisy = qml.QNode(qfunc, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = tf.Variable([np.pi / 4, np.pi / 4])

        with tf.GradientTape() as t:
            res = mitigated_qnode(theta)

        assert qml.math.allclose(res, out_ideal, atol=1e-2)

        grad = t.gradient(res, theta)
        with tf.GradientTape() as t:
            res_ideal = qnode_ideal(theta)
        grad_ideal = t.gradient(res_ideal, theta)

        assert qml.math.allclose(grad_ideal, grad_ideal_0)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.autograd
    def test_diffability_autograd_multi(self):
        """Testing that the mitigated qnode can be differentiated and returns
        the correct gradient in autograd for multiple measurements"""
        qnode_noisy = qml.QNode(qfunc_multi, dev_noisy)
        qnode_ideal = qml.QNode(qfunc_multi, dev_ideal)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = np.array([np.pi / 4, np.pi / 6], requires_grad=True)

        res = qml.math.stack(mitigated_qnode(theta))
        assert qml.math.allclose(res, out_ideal_multi, atol=1e-2)

        grad = qml.jacobian(lambda t: qml.math.stack(mitigated_qnode(t)))(theta)
        grad_ideal = qml.jacobian(lambda t: qml.math.stack(qnode_ideal(t)))(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0_multi, atol=1e-6)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["auto", "jax"])
    def test_diffability_jax_multi(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns
        the correct gradient in jax for multiple measurements"""
        import jax
        import jax.numpy as jnp

        qnode_noisy = qml.QNode(qfunc_multi, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc_multi, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = jnp.array(
            [np.pi / 4, np.pi / 6],
        )

        res = qml.math.stack(mitigated_qnode(theta))
        assert qml.math.allclose(res, out_ideal_multi, atol=1e-2)

        grad = jax.jacobian(lambda t: qml.math.stack(mitigated_qnode(t)))(theta)
        grad_ideal = jax.jacobian(lambda t: qml.math.stack(qnode_ideal(t)))(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0_multi, atol=1e-6)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
    def test_diffability_jaxjit_multi(self, interface):
        """Testing that the mitigated qnode can be differentiated and
        returns the correct gradient in jax-jit for multiple measurements"""
        import jax
        import jax.numpy as jnp

        qnode_noisy = qml.QNode(qfunc_multi, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc_multi, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = jax.jit(
            mitigate_with_zne(qnode_noisy, scale_factors, fold_global, richardson_extrapolate)
        )

        theta = jnp.array(
            [np.pi / 4, np.pi / 6],
        )

        res = qml.math.stack(mitigated_qnode(theta))
        assert qml.math.allclose(res, out_ideal_multi, atol=1e-2)

        grad = jax.jacobian(lambda t: qml.math.stack(mitigated_qnode(t)))(theta)
        grad_ideal = jax.jacobian(lambda t: qml.math.stack(qnode_ideal(t)))(theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0_multi, atol=1e-6)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["auto", "torch"])
    def test_diffability_torch_multi(self, interface):
        """Testing that the mitigated qnode can be differentiated and returns
        the correct gradient in torch for multiple measurements"""
        import torch

        qnode_noisy = qml.QNode(qfunc_multi, dev_noisy, interface=interface)
        qnode_ideal = qml.QNode(qfunc_multi, dev_ideal, interface=interface)

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = torch.tensor([np.pi / 4, np.pi / 6], requires_grad=True)

        res = qml.math.stack(mitigated_qnode(theta))
        assert qml.math.allclose(res, out_ideal_multi, atol=1e-2)

        grad = torch.autograd.functional.jacobian(
            lambda t: qml.math.stack(mitigated_qnode(t)), theta
        )
        grad_ideal = torch.autograd.functional.jacobian(
            lambda t: qml.math.stack(qnode_ideal(t)), theta
        )
        assert qml.math.allclose(grad_ideal, grad_ideal_0_multi, atol=1e-6)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)

    @pytest.mark.tf
    def test_diffability_tf_multi(self):
        """Testing that the mitigated qnode can be differentiated and returns
        the correct gradient in tf for multiple measurements"""
        import tensorflow as tf

        qnode_noisy = qml.QNode(qfunc_multi, dev_noisy, interface="tf")
        qnode_ideal = qml.QNode(qfunc_multi, dev_ideal, interface="tf")

        scale_factors = [1.0, 2.0, 3.0]

        mitigated_qnode = mitigate_with_zne(
            qnode_noisy, scale_factors, fold_global, richardson_extrapolate
        )

        theta = tf.Variable([np.pi / 4, np.pi / 6])

        with tf.GradientTape() as t:
            res = qml.math.stack(mitigated_qnode(theta))

        assert qml.math.allclose(res, out_ideal_multi, atol=1e-2)

        grad = t.jacobian(res, theta)
        with tf.GradientTape() as t:
            res_ideal = qml.math.stack(qnode_ideal(theta))

        grad_ideal = t.jacobian(res_ideal, theta)
        assert qml.math.allclose(grad_ideal, grad_ideal_0_multi, atol=1e-6)
        assert qml.math.allclose(grad, grad_ideal, atol=1e-2)
