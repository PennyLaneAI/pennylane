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
# pylint: disable=too-many-arguments,too-few-public-methods,unexpected-keyword-arg,redefined-outer-name
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow")

shots_and_num_copies = [((1, (5, 2), 10), 4)]
shots_and_num_copies_hess = [((10, (5, 1)), 2)]

kwargs = {
    "finite-diff": {"h": 10e-2},
    "parameter-shift": {},
    "spsa": {"h": 10e-2, "num_directions": 20},
}

qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff"],
    ["default.qubit", "parameter-shift"],
    ["default.qubit", "spsa"],
]

TOLS = {
    "finite-diff": 0.3,
    "parameter-shift": 2e-2,
    "spsa": 0.5,
}


@pytest.fixture
def gradient_kwargs(request):
    diff_method = request.node.funcargs["diff_method"]
    seed = request.getfixturevalue("seed")
    return kwargs[diff_method] | (
        {"sampler_rng": np.random.default_rng(seed)} if diff_method == "spsa" else {}
    )


@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies)
@pytest.mark.parametrize("dev_name,diff_method", qubit_device_and_diff_method)
@pytest.mark.parametrize(
    "decorator,interface",
    [(tf.function, "tf"), (lambda x: x, "tf-autograph")],
)
class TestReturnWithShotVectors:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types and shot vectors."""

    def test_jac_single_measurement_param(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For one measurement and one param, the gradient is a float."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a, wires=0)
            qml.RX(0.7, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(1.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies,)

    def test_jac_single_measurement_multiple_param(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b, **_):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(1.5, dtype=tf.float64)
        b = tf.Variable(0.7, dtype=tf.float64)

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
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable([1.5, 0.7], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 2)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a, wires=0)
            qml.RX(0.7, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(1.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 4)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b, **_):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(1.5, dtype=tf.float64)
        b = tf.Variable(0.7, dtype=tf.float64)

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
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable([1.5, 0.7], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """The gradient of multiple measurements with multiple params return a tuple of arrays."""
        par_0 = tf.Variable(1.5, dtype=tf.float64)
        par_1 = tf.Variable(0.7, dtype=tf.float64)

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            max_diff=1,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y, **_):
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
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.RY(a[2], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = tf.Variable([0.7, 0.9, 1.1], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([qml.math.stack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 2, 3)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a, wires=0)
            qml.RX(0.7, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(1.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 5)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots, num_copies, decorator, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b, **_):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(1.5, dtype=tf.float64)
        b = tf.Variable(0.7, dtype=tf.float64)

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
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, **_):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable([1.5, 0.7], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, shots=shots)
            res = qml.math.stack([tf.experimental.numpy.hstack(r) for r in res])

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (num_copies, 5, 2)


@pytest.mark.slow
@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies_hess)
@pytest.mark.parametrize("dev_name,diff_method", qubit_device_and_diff_method)
@pytest.mark.parametrize(
    "decorator,interface",
    [(tf.function, "tf"), (lambda x: x, "tf-autograph")],
)
class TestReturnShotVectorHessian:
    """Class to test the shape of the Hessian with different return types and shot vectors."""

    def test_hessian_expval_multiple_params(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """The hessian of a single measurement with multiple params return a tuple of arrays."""

        if interface == "tf" and diff_method == "spsa":
            # TODO: Find out why.
            pytest.skip("SPSA gradient does not support this particular test case [sc-33150]")

        par_0 = tf.Variable(1.5, dtype=tf.float64)
        par_1 = tf.Variable(0.7, dtype=tf.float64)

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y, **_):
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


shots_and_num_copies = [((30000, 28000, 26000), 3), ((30000, (28000, 2)), 3)]


@pytest.mark.parametrize("shots,num_copies", shots_and_num_copies)
@pytest.mark.parametrize("dev_name,diff_method", qubit_device_and_diff_method)
@pytest.mark.parametrize(
    "decorator,interface",
    [(tf.function, "tf"), (lambda x: x, "tf-autograph")],
)
class TestReturnShotVectorIntegration:
    """Tests for the integration of shots with the TF interface."""

    def test_single_expectation_value(
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y, **_):
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
        self, dev_name, seed, diff_method, gradient_kwargs, shots, num_copies, decorator, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(
            qml.device(dev_name, seed=seed),
            diff_method=diff_method,
            interface=interface,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y, **_):
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
