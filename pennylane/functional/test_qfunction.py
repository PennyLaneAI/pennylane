from pennylane.functional import qfunction
from pennylane.devices.default_qubit_jax import DefaultQubitJax
from pennylane.devices.default_qubit_tf import DefaultQubitTF
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

def test_sanity_check():

    device = qfunction.functional_device(DefaultQubitJax(wires=1))
    # Make sure it still works similarly to a qnode.
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RZ(a, wires=0)
        return qml.expval(qml.Hadamard(0))
    val = circuit(0.0)
    np.testing.assert_allclose(val, 0.707107, rtol=1e-6)


def test_custom_gradient_jax():
    device = qfunction.functional_device(DefaultQubitJax(wires=1))

    @qfunction.device_transform
    def double_gradient(device):
        """Define a custom gradient that doubles the original gradient"""
        def execute(tape):
            """The execute function is the new device"""

            def run_device(params):
                return device(tape.with_parameters(params))

            @jax.custom_vjp
            def f(params):
                return run_device(params)

            def f_fwd(params):
                return run_device(params), params

            def f_bwd(params, grads):
                # This line is cheating :-)
                vals = jax.grad(run_device)(params)
                return vals[0] * 2.0 * grads,

            f.defvjp(f_fwd, f_bwd)
            return f(tape.get_parameters())
        return execute # Return `execute` as the new device method.

    # Example 1: Using the double_gradient on a qfunc
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    double_circuit = double_gradient(circuit)

    val = jax.grad(circuit)(jnp.array(3.14/2.0))
    val2 = jax.grad(double_circuit)(jnp.array(3.14/2.0))
    np.testing.assert_allclose(val * 2.0, val2)


    # Example 2: Using double_gradient directly on the device.
    device_doubler = double_gradient(device)
    @qfunction.qfunc(device_doubler)
    def device_doubler_circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    
    val = jax.grad(circuit)(jnp.array(3.14/2.0))
    val2 = jax.grad(device_doubler_circuit)(jnp.array(3.14/2.0))
    np.testing.assert_allclose(val * 2.0, val2)


def test_custom_gradient_tensorflow():
    @qfunction.device_transform
    def const_gradient(device):
        """A custom graident that always returns 2.0"""
        def execute(tape):
            # Below is all just boilerplate for tensorflow.
            @tf.custom_gradient
            def f(params):
                def grad(upstream):
                    return tf.constant(2.0)
                return device(tape.with_parameters(params)), grad
            return f(tape.get_parameters())
        return execute

    # Build a TF device.
    device = qfunction.functional_device(DefaultQubitTF(wires=1))

    # Build our circuit.
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))


    # Make sure it works normally.
    with tf.GradientTape() as g:
        x = tf.constant(3.14/2.0)
        g.watch(x)
        val = circuit(x)
    y = g.gradient(val, x)
    np.testing.assert_allclose(y, -0.707107, rtol=1e-6)

    # Ensure our custom gradient is now constant.
    with tf.GradientTape() as g:
        x = tf.constant(3.14/2.0)
        g.watch(x)
        val = const_gradient(circuit)(x)
    y = g.gradient(val, x)
    np.testing.assert_allclose(y, 2.0)
