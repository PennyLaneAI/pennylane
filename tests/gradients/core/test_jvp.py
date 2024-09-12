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
"""Tests for the gradients.jvp module."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.measurements.shots import Shots

_x = np.arange(12).reshape((2, 3, 2))

tests_compute_jvp_single = [
    # Single scalar parameter, scalar output
    (np.array(2), np.array([4]), np.array(8)),
    (np.array(2), [np.array(4)], np.array(8)),
    (np.array(2), (np.array(4),), np.array(8)),
    # Single scalar parameter, tensor output
    (np.array([1, 2, 3]), np.array([4]), np.array([4, 8, 12])),
    (np.array([1, 2, 3]), [np.array(4)], np.array([4, 8, 12])),
    (np.array([1, 2, 3]), (np.array(4),), np.array([4, 8, 12])),
    (_x.reshape((4, 3)), np.array([4]), 4 * _x.reshape((4, 3))),
    (_x.reshape((4, 3)), [np.array(4)], 4 * _x.reshape((4, 3))),
    (_x.reshape((4, 3)), (np.array(4),), 4 * _x.reshape((4, 3))),
    (_x, np.array([4]), _x * 4),
    (_x, [np.array(4)], _x * 4),
    (_x, (np.array(4),), _x * 4),
    # Single tensor parameter, scalar output
    (np.array([5, -2]), [np.array([4, 9])], np.array(2)),
    (np.array([[5, -2]]), [np.array([[4, 9]])], np.array(2)),
    (np.array([[4, 3], [5, -2]]), [np.array([[-1, 2], [4, 9]])], np.array(4)),
    # Single tensor parameter, tensor output
    (np.outer([-1, 3], [5, -2]), [np.array([4, 9])], np.array([-2, 6])),
    (np.array([[[3, 2]], [[5, -2]]]), [np.array([[4, 9]])], np.array([30, 2])),
    (
        np.tensordot(_x.reshape((4, 3)), [[4, 3], [5, -2]], axes=0),
        [np.array([[-1, 2], [4, 9]])],
        4 * _x.reshape((4, 3)),
    ),
    # Multiple scalar parameters, scalar output
    (tuple(np.array(x) for x in [2, 1, -9]), np.array([4, -3, 2]), np.array(-13)),
    (tuple(np.array(x) for x in [2, 1, -9]), [np.array(x) for x in [4, -3, 2]], np.array(-13)),
    (tuple(np.array(x) for x in [2, 1, -9]), tuple(np.array(x) for x in [4, -3, 2]), np.array(-13)),
    # Multiple scalar parameters, tensor output
    (
        tuple(np.array([1, 3, -2]) * x for x in [2, 1, -9]),
        np.array([4, -3, 2]),
        np.array([-13, -39, 26]),
    ),
    (
        tuple(np.array([1, 3, -2]) * x for x in [2, 1, -9]),
        [np.array(x) for x in [4, -3, 2]],
        np.array([-13, -39, 26]),
    ),
    (
        tuple(np.array([1, 3, -2]) * x for x in [2, 1, -9]),
        tuple(np.array(x) for x in [4, -3, 2]),
        np.array([-13, -39, 26]),
    ),
    (
        tuple(np.array([[1, 3, -2], [0, -4, 2]]) * x for x in [2, 1, -9]),
        np.array([4, -3, 2]),
        np.array([[-13, -39, 26], [0, 52, -26]]),
    ),
    (
        tuple(np.array([[1, 3, -2], [0, -4, 2]]) * x for x in [2, 1, -9]),
        [np.array(x) for x in [4, -3, 2]],
        np.array([[-13, -39, 26], [0, 52, -26]]),
    ),
    (
        tuple(np.array([[1, 3, -2], [0, -4, 2]]) * x for x in [2, 1, -9]),
        tuple(np.array(x) for x in [4, -3, 2]),
        np.array([[-13, -39, 26], [0, 52, -26]]),
    ),
    (tuple(_x * x for x in [2, 1, -9]), np.array([4, -3, 2]), -13 * _x),
    (tuple(_x * x for x in [2, 1, -9]), [np.array(x) for x in [4, -3, 2]], -13 * _x),
    (tuple(_x * x for x in [2, 1, -9]), tuple(np.array(x) for x in [4, -3, 2]), -13 * _x),
    # Multiple same-shape tensor parameters, scalar output
    (tuple(np.array([1, 2, 3]) * x for x in [2, 1, -9]), [np.array([4, -3, 2])] * 3, np.array(-24)),
    (
        tuple(np.array([[1, 2, 3], [-2, 1, 2]]) * x for x in [2, 1, -9]),
        [np.array([[4, -3, 2], [0, 2, 1]])] * 3,
        np.array(-48),
    ),
    (tuple(_x * x for x in [2, 1, -9]), [_x] * 3, -6 * np.sum(_x**2)),
    # Multiple mixed parameters, scalar output
    (
        (np.array([1, 3]), np.array(2), np.array([[0, 5, 2, 1]])),
        [np.array([2, -1]), np.array(-5), np.array([[1, 2, 3, -2]])],
        np.array(3),
    ),
    ((np.array(2), np.array(2), _x), [np.array(-1), np.array(5), _x], 8 + np.sum(_x**2)),
    # Multiple same-shape tensor parameters, tensor output
    (
        tuple(np.outer([-4, 5, 2], [1, 2, 3]) * x for x in [2, 1, -9]),
        [np.array([4, -3, 2])] * 3,
        -24 * np.array([-4, 5, 2]),
    ),
    (
        tuple(np.tensordot([-4, 5, 2], [[1, 2, 3], [-2, 1, 2]], axes=0) * x for x in [2, 1, -9]),
        [np.array([[4, -3, 2], [0, 2, 1]])] * 3,
        -48 * np.array([-4, 5, 2]),
    ),
    (
        tuple(np.tensordot([[-4, 5, 2], [1, 3, -2]], _x, axes=0) * x for x in [2, 1, -9]),
        [_x] * 3,
        -6 * np.sum(_x**2) * np.array([[-4, 5, 2], [1, 3, -2]]),
    ),
    # Multiple mixed parameters, tensor output
    (
        tuple(np.tensordot([-4, 5, 2], v, axes=0) for v in ([1, 3], 2, [[0, 5, 2, 1]])),
        [np.array([2, -1]), np.array(-5), np.array([[1, 2, 3, -2]])],
        3 * np.array([-4, 5, 2]),
    ),
    (
        tuple(np.tensordot([[-4, 5, 2], [1, 3, -2]], v, axes=0) for v in [2, 2, _x]),
        [np.array(-1), np.array(5), _x],
        (8 + np.sum(_x**2)) * np.array([[-4, 5, 2], [1, 3, -2]]),
    ),
]

jacs, tangs, expects = list(zip(*tests_compute_jvp_single))
tests_compute_jvp_multi = [
    (tuple(jacs[:3]), tangs[0], tuple(expects[:3])),  # scalar return types, scalar parameter
    (tuple(jacs[:3]), tangs[1], tuple(expects[:3])),  # scalar return types, scalar parameter
    (tuple(jacs[2:5]), tangs[0], tuple(expects[2:5])),  # mixed return types, scalar parameter
    (
        (jacs[2], jacs[4], jacs[8]),
        tangs[2],
        (expects[2], expects[4], expects[8]),
    ),  # mixed return types, scalar parameter
    (
        (jacs[12], jacs[12]),
        tangs[12],
        (expects[12], expects[12]),
    ),  # scalar return types, tensor parameter
    (
        (jacs[12], jacs[15]),
        tangs[12],
        (expects[12], expects[15]),
    ),  # mixed return types, tensor parameter
    (
        tuple(jacs[18:20]),
        tangs[18],
        tuple(expects[18:20]),
    ),  # scalar return types, multiple scalar parameters
    (
        tuple(jacs[21:23]),
        tangs[18],
        tuple(expects[21:23]),
    ),  # tensor return types, multiple scalar parameters
    (
        tuple(jacs[24:26]),
        tangs[24],
        tuple(expects[24:26]),
    ),  # tensor return types, multiple scalar parameters
    (
        (jacs[18], jacs[19], jacs[22]),
        tangs[18],
        (expects[18], expects[19], expects[22]),
    ),  # mixed return types, multiple scalar parameters
    (
        (jacs[30], jacs[30]),
        tangs[30],
        (expects[30], expects[30]),
    ),  # scalar return types, multiple tensor parameter
    (
        (jacs[35], jacs[35]),
        tangs[35],
        (expects[35], expects[35]),
    ),  # tensor return types, multiple tensor parameters
    (
        (jacs[30], jacs[35]),
        tangs[30],
        (expects[30], expects[35]),
    ),  # mixed return types, multiple tensor parameters
    (
        (jacs[33], jacs[33]),
        tangs[33],
        (expects[33], expects[33]),
    ),  # scalar return types, mixed parameters
    (
        (jacs[38], jacs[38]),
        tangs[38],
        (expects[38], expects[38]),
    ),  # tensor return types, mixed parameters
    (
        (jacs[33], jacs[38]),
        tangs[33],
        (expects[33], expects[38]),
    ),  # mixed return types, mixed parameters
]


class TestComputeJVPSingle:
    """Tests for the numeric computation of JVPs for single measurements."""

    @pytest.mark.parametrize("jac, tangent, exp", tests_compute_jvp_single)
    def test_compute_jvp_single(self, jac, tangent, exp):
        """Unit test for compute_jvp_single."""
        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert isinstance(jvp, np.ndarray)
        assert np.array_equal(jvp, exp)

    @pytest.mark.parametrize("jac, tangent, exp", tests_compute_jvp_multi)
    def test_compute_jvp_multi(self, jac, tangent, exp):
        """Unit test for compute_jvp_multi."""
        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert all(isinstance(_jvp, np.ndarray) for _jvp in jvp)
        assert all(np.array_equal(_jvp, _exp) for _jvp, _exp in zip(jvp, exp))

    def test_jacobian_is_none_single(self):
        """A None Jacobian returns a None JVP"""

        tangent = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert jvp is None

    def test_jacobian_is_none_multi(self):
        """A None Jacobian returns a None JVP"""

        tangent = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert jvp is None

        jac = (None, None, None)

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert all(_jvp is None for _jvp in jvp)

    def test_zero_tangent_single_measurement_single_params(self):
        """A zero dy vector will return a zero matrix"""
        tangent = np.zeros([1])
        jac = np.array(0.1)

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.all(jvp == np.zeros([1]))

    def test_zero_tangent_single_measurement_multi_params(self):
        """A zero tangent vector will return a zero matrix"""
        tangent = np.zeros([2])
        jac = tuple([np.array(0.1), np.array(0.2)])

        jvp = qml.gradients.compute_jvp_single(tangent, jac)

        assert np.all(jvp == np.zeros([2]))

    def test_zero_dy_multi(self):
        """A zero tangent vector will return a zero matrix"""
        tangent = np.array([0.0, 0.0, 0.0])
        jac = tuple(
            [
                tuple([np.array(0.1), np.array(0.1), np.array(0.1)]),
                tuple([np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([0.1, 0.2])]),
            ]
        )

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)

        assert isinstance(jvp, tuple)
        assert np.all(jvp[0] == np.zeros([1]))
        assert np.all(jvp[1] == np.zeros([2]))

    @pytest.mark.jax
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_jax(self, dtype1, dtype2):
        """Test that using the JAX interface the dtype of the result is
        determined by the dtype of the dy."""
        import jax

        jax.config.update("jax_enable_x64", True)
        dtype = dtype1
        dtype1 = getattr(jax.numpy, dtype1)
        dtype2 = getattr(jax.numpy, dtype2)

        tangent = jax.numpy.array([1], dtype=dtype1)
        jac = tuple([jax.numpy.array(1, dtype=dtype2), jax.numpy.array([1, 1], dtype=dtype2)])
        assert qml.gradients.compute_jvp_multi(tangent, jac)[0].dtype == dtype

    def test_no_trainable_params_adjoint_single(self):
        """An empty jacobian return empty array."""
        tangent = np.array([1.0, 2.0])
        jac = tuple()

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.allclose(jvp, qml.math.zeros(0))

    def test_no_trainable_params_adjoint_multi_measurement(self):
        """An empty jacobian return an empty tuple."""
        tangent = np.array([1.0, 2.0])
        jac = tuple()

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert len(jvp) == 0

    def test_no_trainable_params_gradient_transform_single(self):
        """An empty jacobian return empty array."""
        tangent = np.array([1.0, 2.0])
        jac = qml.math.zeros(0)

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.allclose(jvp, qml.math.zeros(0))

    def test_no_trainable_params_gradient_transform_multi_measurement(self):
        """An empty jacobian return an empty tuple."""
        tangent = np.array([1.0, 2.0])
        jac = tuple([qml.math.zeros(0), qml.math.zeros(0)])

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert len(jvp) == 2
        for j in jvp:
            assert np.allclose(j, qml.math.zeros(0))


@pytest.mark.parametrize("batch_dim", [None, 1, 3])
class TestJVP:
    """Tests for the jvp function"""

    def test_no_trainable_parameters(self, batch_dim):
        """A tape with no trainable parameters will simply return None"""
        x = 0.4 if batch_dim is None else 0.4 * np.arange(1, 1 + batch_dim)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {}
        tangent = np.array([1.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == tuple()
        assert qml.math.allclose(fn(tapes), np.array(0.0))

    def test_zero_tangent_single_measurement_single_param(self, batch_dim):
        """A zero tangent vector will return no tapes and a zero matrix"""

        x = 0.4 if batch_dim is None else 0.4 * np.arange(1, 1 + batch_dim)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}
        tangent = np.array([0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert np.all(fn(tapes) == np.zeros([1]))

    def test_zero_tangent_single_measurement_multiple_param(self, batch_dim):
        """A zero tangent vector will return no tapes and a zero matrix"""

        x = 0.4 if batch_dim is None else 0.4 * np.arange(1, 1 + batch_dim)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([0.0, 0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert np.all(fn(tapes) == np.zeros([1]))

    def test_zero_tangent_single_measurement_probs_multiple_param(self, batch_dim):
        """A zero tangent vector will return no tapes and a zero matrix"""

        x = 0.6 if batch_dim is None else 0.6 * np.arange(1, 1 + batch_dim)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.4, wires=0)
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([0.0, 0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert np.all(fn(tapes) == np.zeros([4]))

    def test_zero_tangent_multiple_measurement_single_param(self, batch_dim):
        """A zero tangent vector will return no tapes and a zero matrix"""

        x = 0.4 if batch_dim is None else 0.4 * np.arange(1, 1 + batch_dim)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}
        tangent = np.array([0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        res = fn(tapes)

        assert tapes == []

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert np.allclose(res[0], 0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], [0, 0])

    def test_zero_tangent_multiple_measurement_multiple_param(self, batch_dim):
        """A zero tangent vector will return no tapes and a zero matrix"""

        x = 0.4 if batch_dim is None else 0.4 * np.arange(1, 1 + batch_dim)
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([0.0, 0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        res = fn(tapes)

        assert tapes == []

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert np.allclose(res[0], 0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], [0, 0])

    # Unskip batch_dim!=None cases once #4462 is resolved
    def test_single_expectation_value(self, tol, batch_dim):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        if batch_dim is not None:
            pytest.skip(reason="JVP computation of batched tapes is disallowed, see #4462")
        dev = qml.device("default.qubit", wires=2)
        x = 0.543 if batch_dim is None else 0.543 * np.arange(1, 1 + batch_dim)
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.execute(tapes))
        assert res.shape == () if batch_dim is None else (batch_dim,)

        exp = np.sum(np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]), axis=0)
        assert np.allclose(res, exp, atol=tol, rtol=0)

    # Unskip batch_dim!=None cases once #4462 is resolved
    def test_multiple_expectation_values(self, tol, batch_dim):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        if batch_dim is not None:
            pytest.skip(reason="JVP computation of batched tapes is disallowed, see #4462")
        dev = qml.device("default.qubit", wires=2)
        x = 0.543 if batch_dim is None else 0.543 * np.arange(1, 1 + batch_dim)
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 2.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(r.shape == () if batch_dim is None else (batch_dim,) for r in res)

        exp = [-np.sin(x), 2 * np.cos(y)]
        if batch_dim is not None:
            exp[1] = np.tensordot(np.ones(batch_dim), exp[1], axes=0)
        assert np.allclose(res, exp, atol=tol, rtol=0)

    # Unskip batch_dim!=None cases once #4462 is resolved
    def test_prob_expval_single_param(self, tol, batch_dim):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs and a single parameter"""
        if batch_dim is not None:
            pytest.skip(reason="JVP computation of batched tapes is disallowed, see #4462")
        dev = qml.device("default.qubit", wires=2)
        x = 0.543 if batch_dim is None else 0.543 * np.arange(1, 1 + batch_dim)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0}
        tangent = np.array([1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 2

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0].shape == () if batch_dim is None else (batch_dim,)
        assert res[1].shape == (2,) if batch_dim is None else (batch_dim, 2)

        expected_0 = -np.sin(x)
        assert np.allclose(res[0], expected_0, atol=tol, rtol=0)

        # Transpose for batch-dimension to be first if present
        expected_1 = np.array([-np.sin(x) / 2, np.sin(x) / 2]).T
        assert np.allclose(res[1], expected_1, atol=tol, rtol=0)

    # Unskip batch_dim!=None cases once #4462 is resolved
    def test_prob_expval_multi_param(self, tol, batch_dim):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs and multiple parameters"""
        if batch_dim is not None:
            pytest.skip(reason="JVP computation of batched tapes is disallowed, see #4462")
        dev = qml.device("default.qubit", wires=2)
        x = 0.543 if batch_dim is None else 0.543 * np.arange(1, 1 + batch_dim)
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2

        exp = [
            -1 * np.sin(x),
            np.array(
                [
                    -(np.cos(y / 2) ** 2 * np.sin(x)) - (np.cos(x / 2) ** 2 * np.sin(y)),
                    -(np.sin(x) * np.sin(y / 2) ** 2) + (np.cos(x / 2) ** 2 * np.sin(y)),
                    (np.sin(x) * np.sin(y / 2) ** 2) + (np.sin(x / 2) ** 2 * np.sin(y)),
                    (np.cos(y / 2) ** 2 * np.sin(x)) - (np.sin(x / 2) ** 2 * np.sin(y)),
                ]
            ).T
            / 2,
        ]

        assert np.allclose(res[0], exp[0], atol=tol, rtol=0)
        assert np.allclose(res[1], exp[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_matches_tangent(self, dtype, batch_dim):
        """Tests that the jvp function matches the dtype of tangent when tangent is
        zero-like."""
        x = np.array([0.1], dtype=np.float64)
        x = x if batch_dim is None else np.outer(x, np.arange(1, 1 + batch_dim))

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x[0], wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dy = np.zeros(1, dtype=dtype)
        _, func = qml.gradients.jvp(tape, dy, qml.gradients.param_shift)

        assert func([]).dtype == dtype


def expected_probs(params):
    """Expected result of the below circuit ansatz."""
    x, y = params[..., 0], params[..., 1]
    c_x, s_x = np.cos(x / 2), np.sin(x / 2)
    c_y, s_y = np.cos(y / 2), np.sin(y / 2)
    # Transpose to put potential broadcasting axis first
    return np.array([c_x * c_y, c_x * s_y, s_x * s_y, s_x * c_y]).T ** 2


def expected_jvp(params, tangent):
    """Expected result of the JVP of the below circuit ansatz."""
    j = qml.jacobian(expected_probs)(params)
    if qml.math.ndim(params) > 1:
        # If there is broadcasting, take the diagonal over
        # the two axes corresponding to broadcasting
        j = np.stack([j[i, :, i, :] for i in range(len(j))])

    return np.tensordot(j, tangent, axes=1)


def ansatz(x, y):
    """Circuit ansatz for gradient tests"""
    qml.RX(x, wires=[0])
    qml.RY(y, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.probs(wires=[0, 1])


class TestJVPGradients:
    """Gradient tests for the jvp function"""

    # Include batch_dim!=None cases once #4462 is resolved
    @pytest.mark.autograd
    @pytest.mark.parametrize("batch_dim", [None])  # , 1, 3])
    def test_autograd(self, tol, batch_dim):
        """Tests that the output of the JVP transform
        can be differentiated using autograd."""
        dev = qml.device("default.qubit", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)
        if batch_dim is not None:
            params = np.outer(np.arange(1, 1 + batch_dim), params, requires_grad=True)
        tangent = np.array([1.0, 0.3], requires_grad=False)

        def cost_fn(params, tangent):
            with qml.queuing.AnnotatedQueue() as q:
                ansatz(params[..., 0], params[..., 1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            return fn(dev.execute(tapes))

        res = cost_fn(params, tangent)
        exp = expected_jvp(params, tangent)
        assert np.allclose(res, exp, atol=tol, rtol=0)

        res = qml.jacobian(cost_fn)(params, tangent)
        exp = qml.jacobian(expected_jvp)(params, tangent)
        assert np.allclose(res, exp, atol=tol, rtol=0)

    # Include batch_dim!=None cases once #4462 is resolved
    @pytest.mark.torch
    @pytest.mark.parametrize("batch_dim", [None])  # , 1, 3])
    def test_torch(self, tol, batch_dim):
        """Tests that the output of the JVP transform
        can be differentiated using Torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        params_np = np.array([0.543, -0.654], requires_grad=True)
        if batch_dim is not None:
            params_np = np.outer(np.arange(1, 1 + batch_dim), params_np, requires_grad=True)
        tangent_np = np.array([1.2, -0.3], requires_grad=False)
        params = torch.tensor(params_np, requires_grad=True, dtype=torch.float64)
        tangent = torch.tensor(tangent_np, requires_grad=False, dtype=torch.float64)

        def cost_fn(params, tangent):
            with qml.queuing.AnnotatedQueue() as q:
                ansatz(params[..., 0], params[..., 1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            return fn(dev.execute(tapes))

        res = cost_fn(params, tangent)
        exp = expected_jvp(params_np, tangent_np)
        assert np.allclose(res.detach(), exp, atol=tol, rtol=0)

        res = torch.autograd.functional.jacobian(cost_fn, (params, tangent))[0]
        exp = qml.jacobian(expected_jvp)(params_np, tangent_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)

    # Include batch_dim!=None cases once #4462 is resolved
    @pytest.mark.tf
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_dim", [None])  # , 1, 3])
    def test_tf(self, tol, batch_dim):
        """Tests that the output of the JVP transform
        can be differentiated using Tensorflow."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        params_np = np.array([0.543, -0.654], requires_grad=True)
        if batch_dim is not None:
            params_np = np.outer(np.arange(1, 1 + batch_dim), params_np, requires_grad=True)
        tangent_np = np.array([1.2, -0.3], requires_grad=False)
        params = tf.Variable(params_np, dtype=tf.float64)
        tangent = tf.constant(tangent_np, dtype=tf.float64)

        def cost_fn(params, tangent):
            with qml.queuing.AnnotatedQueue() as q:
                ansatz(params[..., 0], params[..., 1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            return fn(dev.execute(tapes))

        with tf.GradientTape() as t:
            res = cost_fn(params, tangent)

        exp = expected_jvp(params_np, tangent_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)

        res = t.jacobian(res, params)
        exp = qml.jacobian(expected_jvp)(params_np, tangent_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)

    # Include batch_dim!=None cases once #4462 is resolved
    @pytest.mark.jax
    @pytest.mark.parametrize("batch_dim", [None])  # , 1, 3])
    def test_jax(self, tol, batch_dim):
        """Tests that the output of the JVP transform
        can be differentiated using JAX."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit")
        params_np = np.array([0.543, -0.654], requires_grad=True)
        if batch_dim is not None:
            params_np = np.outer(np.arange(1, 1 + batch_dim), params_np, requires_grad=True)
        tangent_np = np.array([1.2, -0.3], requires_grad=False)
        params = jnp.array(params_np)
        tangent = jnp.array(tangent_np)

        def cost_fn(params, tangent):
            with qml.queuing.AnnotatedQueue() as q:
                ansatz(params[..., 0], params[..., 1])

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            return fn(dev.execute(tapes))

        res = cost_fn(params, tangent)
        exp = expected_jvp(params_np, tangent_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)

        res = jax.jacobian(cost_fn)(params, tangent)
        exp = qml.jacobian(expected_jvp)(params_np, tangent_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)


class TestBatchJVP:
    """Tests for the batch JVP function"""

    @pytest.mark.parametrize("shots", [Shots(None), Shots(10), Shots([20, 10])])
    def test_one_tape_no_trainable_parameters(self, shots):
        """A tape with no trainable parameters will simply return None"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1, shots=shots)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2, shots=shots)
        tape1.trainable_params = {}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)

        # Even though there are 3 parameters, only two contribute
        # to the JVP, so only 2*2=4 quantum evals
        assert len(v_tapes) == 4
        res = fn(dev.execute(v_tapes))

        assert qml.math.allclose(res[0], np.array(0.0))
        assert res[1] is not None

    @pytest.mark.parametrize("shots", [Shots(None), Shots(10), Shots([20, 10])])
    def test_all_tapes_no_trainable_parameters(self, shots):
        """If all tapes have no trainable parameters all outputs will be None"""

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1, shots=shots)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2, shots=shots)
        tape1.trainable_params = set()
        tape2.trainable_params = set()

        tapes = [tape1, tape2]
        tangents = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)

        assert v_tapes == []
        assert qml.math.allclose(fn([])[0], np.array(0.0))
        assert qml.math.allclose(fn([])[1], np.array(0.0))

    def test_zero_tangent(self):
        """A zero dy vector will return no tapes and a zero matrix"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([0.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)
        res = fn(dev.execute(v_tapes))

        # Even though there are 3 parameters, only two contribute
        # to the JVP, so only 2*2=4 quantum evals

        assert len(v_tapes) == 4
        assert np.allclose(res[0], 0)

    def test_reduction_append(self):
        """Test the 'append' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift, reduction="append")
        res = fn(dev.execute(v_tapes))

        # Returned JVPs will be appended to a list, one JVP per tape

        assert len(res) == 2
        assert all(isinstance(r, np.ndarray) for r in res)
        assert res[0].shape == ()
        assert res[1].shape == ()

    def test_reduction_extend(self):
        """Test the 'extend' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=1)

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1.trainable_params = {0}
        tape2.trainable_params = {1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift, reduction="extend")
        res = fn(dev.execute(v_tapes))
        assert len(res) == 4

    def test_reduction_extend_special(self):
        """Test the 'extend' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(
            tapes,
            tangents,
            param_shift,
            reduction=lambda jvps, x: (
                jvps.extend(qml.math.reshape(x, (1,)))
                if not isinstance(x, tuple) and x.shape == ()
                else jvps.extend(x)
            ),
        )
        res = fn(dev.execute(v_tapes))

        assert len(res) == 3

    def test_reduction_callable(self):
        """Test the callable reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1 = qml.tape.QuantumScript.from_queue(q1)
        with qml.queuing.AnnotatedQueue() as q2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(
            tapes, tangents, param_shift, reduction=lambda jvps, x: jvps.append(x)
        )
        res = fn(dev.execute(v_tapes))
        # Returned JVPs will be appended to a list, one JVP per tape
        assert len(res) == 2
