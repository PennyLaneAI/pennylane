# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for using the TF interface with shot vectors and with a QNode"""
# pylint: disable=too-many-arguments,unexpected-keyword-arg
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode
from pennylane.devices import DefaultQubit

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow")

shots_and_num_copies = [(((5, 2), 1, 10), 4), ((1, 10, (5, 2)), 4)]
shots_and_num_copies_hess = [((10, (5, 1)), 2)]

qubit_device_and_diff_method = [
    [DefaultQubit(), "finite-diff", {"h": 10e-2}],
    [DefaultQubit(), "parameter-shift", {}],
    [DefaultQubit(), "spsa", {"h": 10e-2, "num_directions": 20}],
]

TOLS = {
    "finite-diff": 0.3,
    "parameter-shift": 1e-2,
    "spsa": 0.5,
}

interface_and_qubit_device_and_diff_method = [
    ["tf"] + inner_list for inner_list in qubit_device_and_diff_method
]


@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies)
@pytest.mark.parametrize(
    "interface,dev,diff_method,gradient_kwargs", interface_and_qubit_device_and_diff_method
)
class TestReturnWithShotVectors:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types and shot vectors."""

    def test_jac_single_measurement_param(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For one measurement and one param, the gradient is a float."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies,)

    def test_jac_single_measurement_multiple_param(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, (a, b))

        assert isinstance(jac, tuple)
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, tf.Tensor)
            assert j.shape == (num_copies,)

    def test_jacobian_single_measurement_multiple_param_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 2)

    def test_jacobian_single_measurement_param_probs(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 4)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, (a, b))

        assert isinstance(jac, tuple)
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, tf.Tensor)
            assert j.shape == (num_copies, 4)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The gradient of multiple measurements with multiple params return a tuple of arrays."""

        par_0 = tf.Variable(0.1)
        par_1 = tf.Variable(0.2)

        @qnode(dev, diff_method=diff_method, interface=interface, max_diff=1, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(par_0, par_1, shots=shots)
            res = qml.math.stack([qml.math.stack(r) for r in res])

        jac = tape.jacobian(res, (par_0, par_1))

        assert isinstance(jac, tuple)
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, tf.Tensor)
            assert j.shape == (num_copies, 2)

    def test_jacobian_expval_expval_multiple_params_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.RY(a[2], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([qml.math.stack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 2, 3)

    def test_jacobian_multiple_measurement_single_param(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 5)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        jac = tape.jacobian(res, (a, b))

        assert isinstance(jac, tuple)
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, tf.Tensor)
            assert j.shape == (num_copies, 5)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 5, 2)


@pytest.mark.slow
@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies_hess)
@pytest.mark.parametrize(
    "interface,dev,diff_method,gradient_kwargs", interface_and_qubit_device_and_diff_method
)
class TestReturnShotVectorHessian:
    """Class to test the shape of the Hessian with different return types and shot vectors."""

    def test_hessian_expval_multiple_params(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The hessian of a single measurement with multiple params return a tuple of arrays."""

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, max_diff=2, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(par_0, par_1, shots=shots)
                res = qml.math.stack(res)

            jac = tape2.jacobian(res, (par_0, par_1), experimental_use_pfor=False)
            jac = qml.math.stack(jac)

        hess = tape1.jacobian(jac, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2
        for h in hess:
            assert isinstance(h, tf.Tensor)
            assert h.shape == (2, num_copies)

    def test_hessian_expval_multiple_param_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, max_diff=2, **gradient_kwargs)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(params, shots=shots)
                res = qml.math.stack(res)

            jac = tape2.jacobian(res, params, experimental_use_pfor=False)

        hess = tape1.jacobian(jac, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (num_copies, 2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, max_diff=2, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(par_0, par_1, shots=shots)
                res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

            jac = tape2.jacobian(res, (par_0, par_1), experimental_use_pfor=False)
            jac = qml.math.stack(jac)

        hess = tape1.jacobian(jac, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2
        for h in hess:
            assert isinstance(h, tf.Tensor)
            assert h.shape == (2, num_copies, 3)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, max_diff=2, **gradient_kwargs)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(params, shots=shots)
                res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

            jac = tape2.jacobian(res, params, experimental_use_pfor=False)

        hess = tape1.jacobian(jac, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (num_copies, 3, 2, 2)


shots_and_num_copies = [((1000000, 900000, 800000), 3), ((1000000, (900000, 2)), 3)]


@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies)
@pytest.mark.parametrize(
    "interface,dev,diff_method,gradient_kwargs", interface_and_qubit_device_and_diff_method
)
class TestReturnShotVectorIntegration:
    """Tests for the integration of shots with the TF interface."""

    def test_single_expectation_value(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""

        x = tf.Variable(0.543)
        y = tf.Variable(-0.654)

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape:
            res = circuit(x, y, shots=shots)
            res = qml.math.stack(res)

        all_res = tape.jacobian(res, (x, y))

        assert isinstance(all_res, tuple)
        assert len(all_res) == 2

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        tol = TOLS[diff_method]

        for res, exp in zip(all_res, expected):
            assert isinstance(res, tf.Tensor)
            assert res.shape == (num_copies,)
            assert np.allclose(res, exp, atol=tol, rtol=0)

    def test_prob_expectation_values(
        self, dev, diff_method, gradient_kwargs, shots, num_copies, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        x = tf.Variable(0.543)
        y = tf.Variable(-0.654)

        @qnode(dev, diff_method=diff_method, interface=interface, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        with tf.GradientTape() as tape:
            res = circuit(x, y, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        all_res = tape.jacobian(res, (x, y))

        assert isinstance(all_res, tuple)
        assert len(all_res) == 2

        expected = np.array(
            [
                [
                    -np.sin(x),
                    -(np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                    -(np.sin(x) * np.sin(y / 2) ** 2) / 2,
                    (np.sin(x) * np.sin(y / 2) ** 2) / 2,
                    (np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                ],
                [
                    0,
                    -(np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                    (np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                    (np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                    -(np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                ],
            ]
        )

        tol = TOLS[diff_method]

        for res, exp in zip(all_res, expected):
            assert isinstance(res, tf.Tensor)
            assert res.shape == (num_copies, 5)
            assert np.allclose(res, exp, atol=tol, rtol=0)
