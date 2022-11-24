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
"""Integration tests for using the Autograd interface with shot vectors and with a QNode"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode

pytestmark = pytest.mark.autograd

shots = [((5, 2), 1, 10), (1, 10, (5, 2))]

qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", {"h": 10e-2}],
    ["default.qubit", "parameter-shift", {}],
]


@pytest.mark.parametrize("shots", shots)
@pytest.mark.parametrize("dev_name,diff_method,gradient_kwargs", qubit_device_and_diff_method)
class TestReturnWithShotVectors:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types and shot vectors."""

    def test_jac_single_measurement_param(self, dev_name, diff_method, gradient_kwargs, shots):
        """For one measurement and one param, the gradient is a float."""
        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1)

        def cost(a):
            return qml.math.stack(circuit(a))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies,)

    def test_jac_single_measurement_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1)
        b = np.array(0.2)

        def cost(a, b):
            return qml.math.stack(circuit(a, b))

        jac = qml.jacobian(cost, argnum=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, np.ndarray)
            assert j.shape == (num_copies,)

    def test_jacobian_single_measurement_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        dev = qml.device(dev_name, wires=1, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array([0.1, 0.2])

        def cost(a):
            return qml.math.stack(circuit(a))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 2)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1)

        def cost(a):
            return qml.math.stack(circuit(a))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 4)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array(0.1)
        b = np.array(0.2)

        def cost(a, b):
            return qml.math.stack(circuit(a, b))

        jac = qml.jacobian(cost, argnum=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, np.ndarray)
            assert j.shape == (num_copies, 4)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2])

        def cost(a):
            return qml.math.stack(circuit(a))

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The gradient of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = np.array(0.1)
        par_1 = np.array(0.2)

        @qnode(dev, diff_method=diff_method, max_diff=1, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        def cost(x, y):
            res = circuit(x, y)
            return qml.math.stack([qml.math.stack(r) for r in res])

        jac = qml.jacobian(cost, argnum=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, np.ndarray)
            assert j.shape == (num_copies, 2)

    def test_jacobian_expval_expval_multiple_params_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.RY(a[2], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = np.array([0.1, 0.2, 0.3])

        def cost(a):
            res = circuit(a)
            return qml.math.stack([qml.math.stack(r) for r in res])

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 2, 3)

    def test_jacobian_var_var_multiple_params(self, dev_name, diff_method, gradient_kwargs, shots):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = np.array(0.1)
        par_1 = np.array(0.2)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        def cost(x, y):
            res = circuit(x, y)
            return qml.math.stack([qml.math.stack(r) for r in res])

        jac = qml.jacobian(cost, argnum=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, np.ndarray)
            assert j.shape == (num_copies, 2)

    def test_jacobian_var_var_multiple_params_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            qml.RY(a[2], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = np.array([0.1, 0.2, 0.3])

        def cost(a):
            res = circuit(a)
            return qml.math.stack([qml.math.stack(r) for r in res])

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 2, 3)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1)

        def cost(a):
            res = circuit(a)
            return qml.math.stack([qml.math.hstack(r) for r in res])

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 5)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        def cost(a, b):
            res = circuit(a, b)
            return qml.math.stack([qml.math.hstack(r) for r in res])

        jac = qml.jacobian(cost, argnum=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == 2
        for j in jac:
            assert isinstance(j, np.ndarray)
            assert j.shape == (num_copies, 5)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array([0.1, 0.2])

        def cost(a):
            res = circuit(a)
            return qml.math.stack([qml.math.hstack(r) for r in res])

        jac = qml.jacobian(cost)(a)

        assert isinstance(jac, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert jac.shape == (num_copies, 5, 2)

    def test_hessian_expval_multiple_params(self, dev_name, diff_method, gradient_kwargs, shots):
        """The hessian of a single measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = np.array(0.1)
        par_1 = np.array(0.2)

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x, y):
            def cost2(x, y):
                res = circuit(x, y)
                return qml.math.stack(res)

            return qml.math.stack(qml.jacobian(cost2, argnum=[0, 1])(x, y))

        hess = qml.jacobian(cost, argnum=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == 2
        for h in hess:
            assert isinstance(h, np.ndarray)
            assert h.shape == (2, num_copies)

    def test_hessian_expval_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        params = np.array([0.1, 0.2])

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            def cost2(x):
                res = circuit(x)
                return qml.math.stack(res)

            return qml.jacobian(cost2)(x)

        hess = qml.jacobian(cost)(params)

        assert isinstance(hess, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert hess.shape == (num_copies, 2, 2)

    def test_hessian_var_multiple_params(self, dev_name, diff_method, gradient_kwargs, shots):
        """The hessian of a single measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = np.array(0.1)
        par_1 = np.array(0.2)

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x, y):
            def cost2(x, y):
                res = circuit(x, y)
                return qml.math.stack(res)

            return qml.math.stack(qml.jacobian(cost2, argnum=[0, 1])(x, y))

        hess = qml.jacobian(cost, argnum=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == 2
        for h in hess:
            assert isinstance(h, np.ndarray)
            assert h.shape == (2, num_copies)

    def test_hessian_var_multiple_param_array(self, dev_name, diff_method, gradient_kwargs, shots):
        """The hessian of single measurement with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        params = np.array([0.1, 0.2])

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x):
            def cost2(x):
                res = circuit(x)
                return qml.math.stack(res)

            return qml.jacobian(cost2)(x)

        hess = qml.jacobian(cost)(params)

        assert isinstance(hess, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert hess.shape == (num_copies, 2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""

        # override shots to speed up the test
        shots = (5, 10)

        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = np.array(0.1)
        par_1 = np.array(0.2)

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def cost(x, y):
            def cost2(x, y):
                res = circuit(x, y)
                return qml.math.stack([qml.math.hstack(r) for r in res])

            return qml.math.stack(qml.jacobian(cost2, argnum=[0, 1])(x, y))

        hess = qml.jacobian(cost, argnum=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == 2
        for h in hess:
            assert isinstance(h, np.ndarray)
            assert h.shape == (2, num_copies, 5)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        # override shots to speed up the test
        shots = (5, 10)

        dev = qml.device(dev_name, wires=2, shots=shots)

        params = np.array([0.1, 0.2])

        @qnode(dev, diff_method=diff_method, max_diff=2, **gradient_kwargs)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        def cost(x):
            def cost2(x):
                res = circuit(x)
                return qml.math.stack([qml.math.hstack(r) for r in res])

            return qml.jacobian(cost2)(x)

        hess = qml.jacobian(cost)(params)

        assert isinstance(hess, np.ndarray)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert hess.shape == (num_copies, 5, 2, 2)


qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", {"h": 10e-2}],
    ["default.qubit", "parameter-shift", {}],
]

finite_diff_shot_vec_tol = 0.3
param_shift_shot_vec_tol = 10e-3

shots = [(1000000, 900000, 800000), (1000000, (900000, 2))]


@pytest.mark.parametrize("shots", shots)
@pytest.mark.parametrize("dev_name,diff_method,gradient_kwargs", qubit_device_and_diff_method)
class TestReturnShotVectorIntegration:
    """Tests for the integration of shots with the autograd interface."""

    def test_single_expectation_value(self, dev_name, diff_method, gradient_kwargs, shots):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device(dev_name, wires=2, shots=shots)
        x = np.array(0.543)
        y = np.array(-0.654)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        def cost(x, y):
            res = circuit(x, y)
            return qml.math.stack(res)

        all_res = qml.jacobian(cost, argnum=[0, 1])(x, y)

        assert isinstance(all_res, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(all_res) == 2

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        tol = finite_diff_shot_vec_tol if diff_method == "finite-diff" else param_shift_shot_vec_tol

        for res, exp in zip(all_res, expected):
            assert isinstance(res, np.ndarray)
            assert res.shape == (num_copies,)
            assert np.allclose(res, exp, atol=tol, rtol=0)

    def test_prob_expectation_values(self, dev_name, diff_method, gradient_kwargs, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device(dev_name, wires=2, shots=shots)
        x = np.array(0.543)
        y = np.array(-0.654)

        @qnode(dev, diff_method=diff_method, **gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        def cost(x, y):
            res = circuit(x, y)
            return qml.math.stack([qml.math.hstack(r) for r in res])

        all_res = qml.jacobian(cost, argnum=[0, 1])(x, y)

        assert isinstance(all_res, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
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

        tol = finite_diff_shot_vec_tol if diff_method == "finite-diff" else param_shift_shot_vec_tol

        for res, exp in zip(all_res, expected):
            assert isinstance(res, np.ndarray)
            assert res.shape == (num_copies, 5)
            assert np.allclose(res, exp, atol=tol, rtol=0)
