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

    @qfunction.qfunc(device)
    def circuit(a):
        qml.RZ(a, wires=0)
        return qml.expval(qml.Hadamard(0))
    val = circuit(0.0)
    np.testing.assert_allclose(val, 0.707107, rtol=1e-6)


def test_custom_gradient_jax():
    device = qfunction.functional_device(DefaultQubitJax(wires=1))

    @qfunction.device_transform
    def custom_gradient(device):
        def execute(tape):

            def run_device(params):
                return device(tape.with_parameters(params))

            @jax.custom_vjp
            def f(params):
                return run_device(params)

            def f_fwd(params):
                result = run_device(params)
                return result, params

            def f_bwd(params, grads):
                # This line is cheating :-)
                vals = jax.grad(run_device)(params)
                return vals[0] * 2.0 * grads,

            f.defvjp(f_fwd, f_bwd)
            return f(tape.get_parameters())
        return execute

    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    val = jax.grad(circuit)(jnp.array(3.14/2.0))
    val2 = jax.grad(custom_gradient(circuit))(jnp.array(3.14/2.0))
    np.testing.assert_allclose(val * 2.0, val2)


def test_custom_gradient_tensorflow():
    device = qfunction.functional_device(DefaultQubitTF(wires=1))

    @qfunction.device_transform
    def custom_gradient(device):
        def execute(tape):
            @tf.custom_gradient
            def f(params):
                def grad(upstream):
                    return 2.0
                return device(tape.with_parameters(params)), grad
            return f(tape.get_parameters())
        return execute

    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    with tf.GradientTape() as g:
        x = tf.constant(3.14/2.0)
        g.watch(x)
        val = circuit(x)
    y = g.gradient(val, x)
    with tf.GradientTape() as g:
        x = tf.constant(3.14/2.0)
        g.watch(x)
        val = custom_gradient(circuit)(x)
    y2 = g.gradient(val, x)
    np.testing.assert_allclose(y, -0.707107, rtol=1e-6)
    np.testing.assert_allclose(y2, 2.0)


