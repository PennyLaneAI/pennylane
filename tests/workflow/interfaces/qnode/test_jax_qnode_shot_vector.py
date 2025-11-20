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
"""Integration tests for using the jax interface with shot vectors and with a QNode"""
# pylint: disable=too-many-arguments,too-many-public-methods
import pytest
from flaky import flaky

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

all_shots = [(1, 20, 100), (1, (20, 1), 100), (1, (5, 4), 100)]

qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", {"h": 10e-2}],
    ["default.qubit", "parameter-shift", {}],
    ["default.qubit", "spsa", {"h": 10e-2, "num_directions": 20}],
]

interface_and_qubit_device_and_diff_method = [
    ["jax"] + inner_list for inner_list in qubit_device_and_diff_method
]

TOLS = {
    "finite-diff": 0.3,
    "parameter-shift": 1e-2,
    "spsa": 0.32,
}

jacobian_fn = [jax.jacobian, jax.jacrev, jax.jacfwd]


@pytest.mark.parametrize("shots", all_shots)
@pytest.mark.parametrize(
    "interface,dev_name,diff_method,gradient_kwargs", interface_and_qubit_device_and_diff_method
)
class TestReturnWithShotVectors:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types and shot vectors."""

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jac_single_measurement_param(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For one measurement and one param, the gradient is a float."""
        dev = qml.device(dev_name, wires=1)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, jax.numpy.ndarray)
            assert j.shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jac_single_measurement_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        dev = qml.device(dev_name, wires=1)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2
            assert j[0].shape == ()
            assert j[1].shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        dev = qml.device(dev_name, wires=1)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, jax.numpy.ndarray)
            assert j.shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, jax.numpy.ndarray)
            assert j.shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)

            assert isinstance(j[0], jax.numpy.ndarray)
            assert j[0].shape == (4,)

            assert isinstance(j[1], jax.numpy.ndarray)
            assert j[1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, jax.numpy.ndarray)
            assert j.shape == (4, 2)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=1,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)

            assert isinstance(j[0], tuple)
            assert len(j[0]) == 2
            assert isinstance(j[0][0], jax.numpy.ndarray)
            assert j[0][0].shape == ()
            assert isinstance(j[0][1], jax.numpy.ndarray)
            assert j[0][1].shape == ()

            assert isinstance(j[1], tuple)
            assert len(j[1]) == 2
            assert isinstance(j[1][0], jax.numpy.ndarray)
            assert j[1][0].shape == ()
            assert isinstance(j[1][1], jax.numpy.ndarray)
            assert j[1][1].shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_expval_expval_multiple_params_array(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2  # measurements

            assert isinstance(j[0], jax.numpy.ndarray)
            assert j[0].shape == (2,)

            assert isinstance(j[1], jax.numpy.ndarray)
            assert j[1].shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_var_var_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2

            assert isinstance(j[0], tuple)
            assert len(j[0]) == 2
            assert isinstance(j[0][0], jax.numpy.ndarray)
            assert j[0][0].shape == ()
            assert isinstance(j[0][1], jax.numpy.ndarray)
            assert j[0][1].shape == ()

            assert isinstance(j[1], tuple)
            assert len(j[1]) == 2
            assert isinstance(j[1][0], jax.numpy.ndarray)
            assert j[1][0].shape == ()
            assert isinstance(j[1][1], jax.numpy.ndarray)
            assert j[1][1].shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_var_var_multiple_params_array(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2  # measurements

            assert isinstance(j[0], jax.numpy.ndarray)
            assert j[0].shape == (2,)

            assert isinstance(j[1], jax.numpy.ndarray)
            assert j[1].shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(jac, tuple)
            assert len(j) == 2

            assert isinstance(j[0], jax.numpy.ndarray)
            assert j[0].shape == ()

            assert isinstance(j[1], jax.numpy.ndarray)
            assert j[1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2

            assert isinstance(j[0], tuple)
            assert len(j[0]) == 2
            assert isinstance(j[0][0], jax.numpy.ndarray)
            assert j[0][0].shape == ()
            assert isinstance(j[0][1], jax.numpy.ndarray)
            assert j[0][1].shape == ()

            assert isinstance(j[1], tuple)
            assert len(j[1]) == 2
            assert isinstance(j[1][0], jax.numpy.ndarray)
            assert j[1][0].shape == (4,)
            assert isinstance(j[1][1], jax.numpy.ndarray)
            assert j[1][1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(jac) == num_copies
        for j in jac:
            assert isinstance(j, tuple)
            assert len(j) == 2  # measurements

            assert isinstance(j[0], jax.numpy.ndarray)
            assert j[0].shape == (2,)

            assert isinstance(j[1], jax.numpy.ndarray)
            assert j[1].shape == (4, 2)

    def test_hessian_expval_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(hess, tuple)
            assert len(h) == 2

            assert isinstance(h[0], tuple)
            assert len(h[0]) == 2
            assert isinstance(h[0][0], jax.numpy.ndarray)
            assert h[0][0].shape == ()
            assert h[0][1].shape == ()

            assert isinstance(h[1], tuple)
            assert len(h[1]) == 2
            assert isinstance(h[1][0], jax.numpy.ndarray)
            assert h[1][0].shape == ()
            assert h[1][1].shape == ()

    def test_hessian_expval_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, jax.numpy.ndarray)
            assert h.shape == (2, 2)

    def test_hessian_var_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, tuple)
            assert len(h) == 2

            assert isinstance(h[0], tuple)
            assert len(h[0]) == 2
            assert isinstance(h[0][0], jax.numpy.ndarray)
            assert h[0][0].shape == ()
            assert h[0][1].shape == ()

            assert isinstance(h[1], tuple)
            assert len(h[1]) == 2
            assert isinstance(h[1][0], jax.numpy.ndarray)
            assert h[1][0].shape == ()
            assert h[1][1].shape == ()

    def test_hessian_var_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, jax.numpy.ndarray)
            assert h.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, tuple)
            assert len(h) == 2

            assert isinstance(h[0], tuple)
            assert len(h[0]) == 2
            assert isinstance(h[0][0], tuple)
            assert len(h[0][0]) == 2
            assert isinstance(h[0][0][0], jax.numpy.ndarray)
            assert h[0][0][0].shape == ()
            assert isinstance(h[0][0][1], jax.numpy.ndarray)
            assert h[0][0][1].shape == ()
            assert isinstance(h[0][1], tuple)
            assert len(h[0][1]) == 2
            assert isinstance(h[0][1][0], jax.numpy.ndarray)
            assert h[0][1][0].shape == ()
            assert isinstance(h[0][1][1], jax.numpy.ndarray)
            assert h[0][1][1].shape == ()

            assert isinstance(h[1], tuple)
            assert len(h[1]) == 2
            assert isinstance(h[1][0], tuple)
            assert len(h[1][0]) == 2
            assert isinstance(h[1][0][0], jax.numpy.ndarray)
            assert h[1][0][0].shape == (2,)
            assert isinstance(h[1][0][1], jax.numpy.ndarray)
            assert h[1][0][1].shape == (2,)
            assert isinstance(h[1][1], tuple)
            assert len(h[1][1]) == 2
            assert isinstance(h[1][1][0], jax.numpy.ndarray)
            assert h[1][1][0].shape == (2,)
            assert isinstance(h[1][1][1], jax.numpy.ndarray)
            assert h[1][1][1].shape == (2,)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit)(params)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, tuple)
            assert len(h) == 2

            assert isinstance(h[0], jax.numpy.ndarray)
            assert h[0].shape == (2, 2)

            assert isinstance(h[1], jax.numpy.ndarray)
            assert h[1].shape == (2, 2, 2)

    def test_hessian_probs_var_multiple_params(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        par_0 = qml.numpy.array(0.1)
        par_1 = qml.numpy.array(0.2)

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, tuple)
            assert len(h) == 2

            assert isinstance(h[0], tuple)
            assert len(h[0]) == 2
            assert isinstance(h[0][0], tuple)
            assert len(h[0][0]) == 2
            assert isinstance(h[0][0][0], jax.numpy.ndarray)
            assert h[0][0][0].shape == ()
            assert isinstance(h[0][0][1], jax.numpy.ndarray)
            assert h[0][0][1].shape == ()
            assert isinstance(h[0][1], tuple)
            assert len(h[0][1]) == 2
            assert isinstance(h[0][1][0], jax.numpy.ndarray)
            assert h[0][1][0].shape == ()
            assert isinstance(h[0][1][1], jax.numpy.ndarray)
            assert h[0][1][1].shape == ()

            assert isinstance(h[1], tuple)
            assert len(h[1]) == 2
            assert isinstance(h[1][0], tuple)
            assert len(h[1][0]) == 2
            assert isinstance(h[1][0][0], jax.numpy.ndarray)
            assert h[1][0][0].shape == (2,)
            assert isinstance(h[1][0][1], jax.numpy.ndarray)
            assert h[1][0][1].shape == (2,)
            assert isinstance(h[1][1], tuple)
            assert len(h[1][1]) == 2
            assert isinstance(h[1][1][0], jax.numpy.ndarray)
            assert h[1][1][0].shape == (2,)
            assert isinstance(h[1][1][1], jax.numpy.ndarray)
            assert h[1][1][1].shape == (2,)

    def test_hessian_var_probs_multiple_param_array(
        self, dev_name, diff_method, gradient_kwargs, shots, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots)
        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit)(params)

        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(hess) == num_copies
        for h in hess:
            assert isinstance(h, tuple)
            assert len(h) == 2

            assert isinstance(h[0], jax.numpy.ndarray)
            assert h[0].shape == (2, 2)

            assert isinstance(h[1], jax.numpy.ndarray)
            assert h[1].shape == (2, 2, 2)


qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", {"h": 10e-2}],
    ["default.qubit", "parameter-shift", {}],
]

shots_large = [(1000000, 900000, 800000), (1000000, (900000, 2))]


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", shots_large)
@pytest.mark.parametrize(
    "interface,dev_name,diff_method,gradient_kwargs", interface_and_qubit_device_and_diff_method
)
class TestReturnShotVectorIntegration:
    """Tests for the integration of shots with the Jax interface."""

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_single_expectation_value(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device(dev_name, wires=2)
        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
        all_res = jacobian(circuit, argnums=[0, 1])(x, y)

        assert isinstance(all_res, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(all_res) == num_copies

        for res in all_res:
            assert isinstance(res[0], jax.numpy.ndarray)
            assert res[0].shape == ()

            assert isinstance(res[1], jax.numpy.ndarray)
            assert res[1].shape == ()
            tol = TOLS[diff_method]
            assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_prob_expectation_values(
        self, dev_name, diff_method, gradient_kwargs, shots, jacobian, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device(dev_name, wires=2)
        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qml.set_shots(shots)
        @qnode(dev, interface=interface, diff_method=diff_method, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        all_res = jacobian(circuit, argnums=[0, 1])(x, y)

        tol = TOLS[diff_method]

        assert isinstance(all_res, tuple)
        num_copies = sum(
            [1 for x in shots if isinstance(x, int)] + [x[1] for x in shots if isinstance(x, tuple)]
        )
        assert len(all_res) == num_copies

        for res in all_res:
            assert isinstance(res, tuple)
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert len(res[0]) == 2
            assert np.allclose(res[0][0], -np.sin(x), atol=tol, rtol=0)
            assert isinstance(res[0][0], jax.numpy.ndarray)
            assert np.allclose(res[0][1], 0, atol=tol, rtol=0)
            assert isinstance(res[0][1], jax.numpy.ndarray)

            assert isinstance(res[1], tuple)
            assert len(res[1]) == 2
            assert np.allclose(
                res[1][0],
                [
                    -(np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                    -(np.sin(x) * np.sin(y / 2) ** 2) / 2,
                    (np.sin(x) * np.sin(y / 2) ** 2) / 2,
                    (np.cos(y / 2) ** 2 * np.sin(x)) / 2,
                ],
                atol=tol,
                rtol=0,
            )
            assert isinstance(res[1][0], jax.numpy.ndarray)
            assert np.allclose(
                res[1][1],
                [
                    -(np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                    (np.cos(x / 2) ** 2 * np.sin(y)) / 2,
                    (np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                    -(np.sin(x / 2) ** 2 * np.sin(y)) / 2,
                ],
                atol=tol,
                rtol=0,
            )
            assert isinstance(res[1][1], jax.numpy.ndarray)
