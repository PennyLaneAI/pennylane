import jax
import pytest
import autograd.numpy as anp
from pennylane import numpy as np

import pennylane as qml
from pennylane import qnode


qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", "backward"],
    ["default.qubit", "parameter-shift", "backward"],
    ["default.qubit", "backprop", "forward"],
    ["default.qubit", "adjoint", "forward"],
    ["default.qubit", "adjoint", "backward"],
]


@pytest.mark.parametrize("dev_name,diff_method,mode", qubit_device_and_diff_method)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_execution_single_measurement_param(self, dev_name, diff_method, mode):
        """For one measurement and one param, the gradient is a float."""
        # qml.disable_return()
        import jax

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)

        grad = jax.grad(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == ()

    def test_execution_single_measurement_multiple_param(self, dev_name, diff_method, mode):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        import jax

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        grad = jax.grad(circuit, argnums=[0, 1])(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_execution_single_measurement_multiple_param_array(self, dev_name, diff_method, mode):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        import jax

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        grad = jax.grad(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == (2,)

    def test_execution_single_measurement_param_probs(self, dev_name, diff_method, mode):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        import jax

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jax.jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4,)

    def test_execution_single_measurement_probs_multiple_param(self, dev_name, diff_method, mode):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jax.jacobian(circuit, argnums=[0, 1])(a, b)
        print(jac)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_execution_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, mode
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])
        jac = jax.jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4, 2)

    def test_execution_multiple_measurement_single_param(self, dev_name, diff_method, mode):
        """The jacobian of multiple measurements with a single params return an array."""
        import jax

        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jax.jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_execution_multiple_measurement_multiple_param(self, dev_name, diff_method, mode):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = jax.jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (4,)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (4,)

    def test_execution_multiple_measurement_multiple_param_array(self, dev_name, diff_method, mode):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jax.jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4, 2)

    def test_hessian_expval_multiple_params(self, dev_name, diff_method, mode):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        import jax

        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_multiple_derivative_expval_multiple_param_array(self, dev_name, diff_method, mode):
        """The hessian of single measurement with a multiple params array return a single array."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_multiple_derivative_var_multiple_params(self, dev_name, diff_method, mode):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        import jax

        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_multiple_derivative_var_multiple_param_array(self, dev_name, diff_method, mode):
        """The hessian of single measurement with a multiple params array return a single array."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_multiple_derivative_probs_expval_multiple_params(self, dev_name, diff_method, mode):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        import jax

        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], tuple)
        assert len(hess[0][0]) == 2
        assert isinstance(hess[0][0][0], jax.numpy.ndarray)
        assert hess[0][0][0].shape == ()
        assert isinstance(hess[0][0][1], jax.numpy.ndarray)
        assert hess[0][0][1].shape == ()
        assert isinstance(hess[0][1], tuple)
        assert len(hess[0][1]) == 2
        assert isinstance(hess[0][1][0], jax.numpy.ndarray)
        assert hess[0][1][0].shape == ()
        assert isinstance(hess[0][1][1], jax.numpy.ndarray)
        assert hess[0][1][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], tuple)
        assert len(hess[1][0]) == 2
        assert isinstance(hess[1][0][0], jax.numpy.ndarray)
        assert hess[1][0][0].shape == (4,)
        assert isinstance(hess[1][0][1], jax.numpy.ndarray)
        assert hess[1][0][1].shape == (4,)
        assert isinstance(hess[1][1], tuple)
        assert len(hess[1][1]) == 2
        assert isinstance(hess[1][1][0], jax.numpy.ndarray)
        assert hess[1][1][0].shape == (4,)
        assert isinstance(hess[1][1][1], jax.numpy.ndarray)
        assert hess[1][1][1].shape == (4,)

    def test_multiple_derivative_expval_probs_multiple_param_array(
        self, dev_name, diff_method, mode
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (4, 2, 2)

    def test_multiple_derivative_probs_var_multiple_params(self, dev_name, diff_method, mode):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        import jax

        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = qml.numpy.array(0.1)
        par_1 = qml.numpy.array(0.2)

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], tuple)
        assert len(hess[0][0]) == 2
        assert isinstance(hess[0][0][0], jax.numpy.ndarray)
        assert hess[0][0][0].shape == ()
        assert isinstance(hess[0][0][1], jax.numpy.ndarray)
        assert hess[0][0][1].shape == ()
        assert isinstance(hess[0][1], tuple)
        assert len(hess[0][1]) == 2
        assert isinstance(hess[0][1][0], jax.numpy.ndarray)
        assert hess[0][1][0].shape == ()
        assert isinstance(hess[0][1][1], jax.numpy.ndarray)
        assert hess[0][1][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], tuple)
        assert len(hess[1][0]) == 2
        assert isinstance(hess[1][0][0], jax.numpy.ndarray)
        assert hess[1][0][0].shape == (4,)
        assert isinstance(hess[1][0][1], jax.numpy.ndarray)
        assert hess[1][0][1].shape == (4,)
        assert isinstance(hess[1][1], tuple)
        assert len(hess[1][1]) == 2
        assert isinstance(hess[1][1][0], jax.numpy.ndarray)
        assert hess[1][1][0].shape == (4,)
        assert isinstance(hess[1][1][1], jax.numpy.ndarray)
        assert hess[1][1][1].shape == (4,)

    def test_multiple_derivative_var_probs_multiple_param_array(self, dev_name, diff_method, mode):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        import jax

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        dev = qml.device(dev_name, wires=2)

        params = jax.numpy.array([0.1, 0.2])

        @qnode(dev, interface="jax", diff_method=diff_method, max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0, 1])

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (4, 2, 2)
