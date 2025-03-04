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
"""Tests for the gradients.parameter_shift module using mid-circuit measurements."""
import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qml

seeds = [8521, 962, 81]


def random_state(seed, num_wires):
    """Create a random normalized pure state."""
    return unitary_group.rvs(2**num_wires, seed)[0][0]


def random_observable(seed, num_wires):
    """Create a random Hermitian operator."""
    rng = np.random.default_rng(seed)
    H = rng.random((2**num_wires, 2**num_wires, 2)) @ np.array([1, 1j])
    H += H.conj().T
    return H


@pytest.mark.parametrize("mcm_version", [0, 1])
class TestParameterShiftMCM:

    @pytest.mark.autograd
    @pytest.mark.parametrize("seed", seeds)
    @pytest.mark.parametrize("reset", [False, True])
    def test_measure_no_postselect(self, reset, seed, mcm_version):
        """Test that a mid-circuit measurement without postselection alone does
        not modify parameter-shift rules."""

        wires = [0, 1, 2]
        state = random_state(seed, len(wires))
        H = qml.Hermitian(random_observable(seed, len(wires)), wires=wires)

        def circuit(x, y):
            qml.QubitStateVector(state, wires)
            qml.RX(x[0], 0)
            qml.RX(x[1], 1)
            qml.measure(0, reset=reset)
            qml.RY(y[0], 0)
            qml.RY(y[1], 1)
            return qml.expval(H)

        dev = qml.device("default.qubit")
        psr_kwargs = {"diff_method": "parameter-shift", "mcm_version": mcm_version}
        nodes = [
            qml.QNode(circuit, dev),
            qml.QNode(circuit, dev, **psr_kwargs),
            qml.QNode(circuit, dev, **psr_kwargs, deactivate_mcms=True),
        ]

        x, y = qml.numpy.array([[0.512, -0.732], [1.973, -2.632]], requires_grad=True)

        grads = [qml.jacobian(node)(x, y) for node in nodes]
        for grad in grads:
            assert isinstance(grad, tuple)
            assert all(isinstance(g, qml.numpy.ndarray) for g in grad)
            assert qml.math.shape(grad) == (2, 2)

        assert qml.math.allclose(grads[1:], grads[0])

    @pytest.mark.autograd
    @pytest.mark.parametrize("seed", seeds)
    @pytest.mark.parametrize("reset", [False, True])
    def test_measure_postselect(self, reset, seed, mcm_version):
        """Test that a mid-circuit measurement with postselection
        modifies parameter-shift rules for parameters in the backwards cone."""

        wires = [0, 1, 2]
        state = random_state(seed, len(wires))
        H = qml.Hermitian(random_observable(seed, len(wires)), wires=wires)

        def circuit(x, y):
            qml.QubitStateVector(state, wires)
            qml.RX(x[0], 0)
            qml.RX(x[1], 1)
            qml.measure(0, reset=reset, postselect=1)
            qml.RY(y[0], 0)
            qml.RY(y[1], 1)
            return qml.expval(H)

        dev = qml.device("default.qubit")
        psr_kwargs = {"diff_method": "parameter-shift", "mcm_version": mcm_version}
        nodes = [
            qml.QNode(circuit, dev),
            qml.QNode(circuit, dev, **psr_kwargs),
            qml.QNode(circuit, dev, **psr_kwargs, deactivate_mcms=True),
        ]

        x, y = qml.numpy.array([[0.512, -0.732], [1.973, -2.632]], requires_grad=True)

        grads = [qml.jacobian(node)(x, y) for node in nodes]
        for grad in grads:
            assert isinstance(grad, tuple)
            assert all(isinstance(g, qml.numpy.ndarray) for g in grad)
            assert qml.math.shape(grad) == (2, 2)

        assert qml.math.allclose(grads[1], grads[0])
        assert not qml.math.allclose(grads[2][0][0], grads[0][0][0])
        grads[2][0][0] = grads[0][0][0]
        assert qml.math.allclose(grads[2], grads[0])

    @pytest.mark.autograd
    @pytest.mark.parametrize("seed", seeds)
    @pytest.mark.parametrize("measure_wire", [0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_conditioned_parameter(self, reset, measure_wire, seed, mcm_version):
        """Test that a parameter feeding into a classically controlled gate is
        differentiated correctly with the original parameter-shift rule."""
        wires = [0, 1, 2]
        state = random_state(seed, len(wires))
        H = qml.Hermitian(random_observable(seed, len(wires)), wires=wires)

        def circuit(x, y):
            qml.QubitStateVector(state, wires)
            mcm1 = qml.measure(measure_wire, reset=reset, postselect=None)
            qml.cond(mcm1, qml.RX)(x, 1)
            mcm2 = qml.measure(2, reset=reset, postselect=None)
            qml.cond(mcm2, qml.IsingXY)(y, [0, 1])
            return qml.probs(op=H)

        dev = qml.device("default.qubit")
        psr_kwargs = {"diff_method": "parameter-shift", "mcm_version": mcm_version}
        x, y = qml.numpy.array([0.512, -0.732], requires_grad=True)

        grads = [
            qml.jacobian(qml.QNode(circuit, dev))(x, y),
            qml.jacobian(qml.QNode(circuit, dev, **psr_kwargs))(x, y),
            qml.jacobian(qml.QNode(circuit, dev, **psr_kwargs, deactivate_mcms=True))(x, y),
        ]
        for grad in grads:
            assert isinstance(grad, tuple)
            assert all(isinstance(g, qml.numpy.ndarray) for g in grad)
            assert qml.math.shape(grad) == (2, 8)

        assert qml.math.allclose(grads[1:], grads[0])

    @pytest.mark.autograd
    @pytest.mark.parametrize("seed", seeds)
    @pytest.mark.parametrize("measure_wire", [0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_conditioned_and_postselect(self, reset, measure_wire, seed, mcm_version):
        """Test that a parameter feeding into a classically controlled gate is
        differentiated correctly in conjunction with postselection."""
        wires = [0, 1, 2]
        state = random_state(seed, len(wires))
        H = qml.Hermitian(random_observable(seed, len(wires)), wires=wires)

        def circuit(x, y):
            qml.QubitStateVector(state, wires)
            mcm1 = qml.measure(measure_wire, reset=reset, postselect=None)
            qml.cond(mcm1, qml.IsingXX)(x, [1, 2])
            mcm2 = qml.measure(2, reset=reset, postselect=1)
            qml.cond(mcm2, qml.IsingXY)(y, [0, 1])
            return qml.probs(op=H)

        dev = qml.device("default.qubit")
        psr_kwargs = {"diff_method": "parameter-shift", "mcm_version": mcm_version}
        x, y = qml.numpy.array([0.512, -0.732], requires_grad=True)

        grads = [
            qml.jacobian(qml.QNode(circuit, dev))(x, y),
            qml.jacobian(qml.QNode(circuit, dev, **psr_kwargs))(x, y),
            qml.jacobian(qml.QNode(circuit, dev, **psr_kwargs, deactivate_mcms=True))(x, y),
        ]
        for grad in grads:
            assert isinstance(grad, tuple)
            assert all(isinstance(g, qml.numpy.ndarray) for g in grad)
            assert qml.math.shape(grad) == (2, 8)

        assert qml.math.allclose(grads[1], grads[0])
        assert not qml.math.allclose(grads[2][0], grads[0][0])
        grads[2] = (grads[0][0], grads[2][1])
        assert qml.math.allclose(grads[2], grads[0])

    @pytest.mark.jax
    @pytest.mark.parametrize("seed", seeds)
    @pytest.mark.parametrize("measure_wire", [0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_mcm_stats(self, reset, measure_wire, seed, mcm_version):
        """Test that statistics of mid-circuit measurements can be
        differentiated correctly with the original PSRs."""
        # This test uses JAX because Autograd does not work with the
        # mixed-shape return types
        import jax
        from jax import numpy as jnp

        wires = [0, 1, 2, 3]
        state = random_state(seed, len(wires))
        H = qml.Hermitian(random_observable(seed, 2), wires=[0, 1])

        def circuit(x, y):
            qml.QubitStateVector(state, wires)
            [qml.RX(x[i] * (i + 1), i) for i in wires]
            mcms1 = [qml.measure(w) for w in wires]
            [qml.RY(y[i] * (i + 1), i) for i in wires]
            mcms2 = [qml.measure(w) for w in wires]
            postproc = (2.0 ** np.arange(4)) @ mcms2
            return qml.probs(op=H), qml.probs(op=mcms1), qml.expval(postproc)

        dev = qml.device("default.qubit")
        psr_kwargs = {"diff_method": "parameter-shift", "mcm_version": mcm_version}
        x, y = jnp.array(
            [[0.512, -0.732, 0.61, -0.2], [0.62, -1.5, -0.263, 0.7453]],
        )

        grads = [
            jax.jacobian(qml.QNode(circuit, dev), argnums=[0, 1])(x, y),
            jax.jacobian(qml.QNode(circuit, dev, **psr_kwargs), argnums=[0, 1])(x, y),
            jax.jacobian(
                qml.QNode(circuit, dev, **psr_kwargs, deactivate_mcms=True), argnums=[0, 1]
            )(x, y),
        ]
        for grad in grads:  # Iterate over QNodes
            assert isinstance(grad, tuple)
            assert len(grad) == 3
            for meas_dim, g in zip([(4,), (16,), ()], grad):  # Iterate over measurements
                assert isinstance(g, tuple)
                assert len(g) == 2  # Two parameters
                for _g in g:  # Iterate over parameters
                    assert isinstance(_g, jnp.ndarray)
                    assert _g.shape == (*meas_dim, 4)

        for g0, g1, g2 in zip(*grads):
            assert qml.math.allclose(np.stack([g1, g2]), g0)
