from pennylane.functional import qfunction
from pennylane.functional.qfunction import single_tape
from pennylane.devices.default_qubit_autograd import DefaultQubitAutograd
from pennylane.devices.default_qubit_jax import DefaultQubitJax
from pennylane.devices.default_qubit_tf import DefaultQubitTF
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from math import pi


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
                return single_tape(device)(tape.with_parameters(params))

            @jax.custom_vjp
            def f(params):
                return run_device(params)

            def f_fwd(params):
                return run_device(params), params

            def f_bwd(params, grads):
                # This line is cheating :-)
                vals = jax.grad(run_device)(params)
                return (vals[0] * 2.0 * grads,)

            f.defvjp(f_fwd, f_bwd)
            return f(tape.get_parameters())
        # Return a batched `execute` as the new device method.
        return lambda tapes: list(map(execute, tapes))

    # Example 1: Using the double_gradient on a qfunc
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    double_circuit = double_gradient(circuit)

    val = jax.grad(circuit)(jnp.array(pi / 2.0))
    val2 = jax.grad(double_circuit)(jnp.array(pi / 2.0))
    np.testing.assert_allclose(val * 2.0, val2)

    # Example 2: Using double_gradient directly on the device.
    device_doubler = double_gradient(device)

    @qfunction.qfunc(device_doubler)
    def device_doubler_circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    val = jax.grad(circuit)(jnp.array(pi / 2.0))
    val2 = jax.grad(device_doubler_circuit)(jnp.array(pi / 2.0))
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

                return single_tape(device)(tape.with_parameters(params)), grad

            return f(tape.get_parameters())

        return lambda tapes: list(map(execute, tapes))

    # Build a TF device.
    device = qfunction.functional_device(DefaultQubitTF(wires=1))

    # Build our circuit.
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    # Make sure it works normally.
    with tf.GradientTape() as g:
        x = tf.constant(pi / 2.0)
        g.watch(x)
        val = circuit(x)
    y = g.gradient(val, x)
    np.testing.assert_allclose(y, -0.707107, rtol=1e-6)

    # Ensure our custom gradient is now constant.
    with tf.GradientTape() as g:
        x = tf.constant(pi / 2.0)
        g.watch(x)
        val = const_gradient(circuit)(x)
    y = g.gradient(val, x)
    np.testing.assert_allclose(y, 2.0)


def test_tape_transform():

    @qfunction.tape_transform
    def cnots_to_czs(tape):
        new_ops = []
        for o in tape.operations:
            # here, we loop through all tape operations, and make
            # the transformation if a RY gate is encountered.
            if isinstance(o, qml.CNOT):
                new_ops.append(qml.Hadamard(wires=o.wires[1]))
                new_ops.append(qml.CZ(wires=o.wires))
                new_ops.append(qml.Hadamard(wires=o.wires[1]))
            else:
                new_ops.append(o)

        return tape.with_operations(new_ops)

    device = qfunction.functional_device(DefaultQubitJax(wires=2))

    @qfunction.qfunc(device)
    def circuit():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(1))

    val = circuit()
    cz_val = cnots_to_czs(circuit)()
    np.testing.assert_allclose(val, cz_val, atol=1e-6, rtol=1e-6)
    assert "H"  not in qfunction.draw(circuit)()
    assert "H" in qfunction.draw(cnots_to_czs(circuit))()

def test_learnable_transform():

    def add_rot(angle):
        """Create a tape transform function that adds `angle` of rotation after every gate"""
        @qfunction.tape_transform
        def transform(tape):
            new_ops = []
            for o in tape.operations:
                new_ops.append(o)
                if not isinstance(o, qml.RX):
                    new_ops.append(qml.RX(angle, wires=o.wires[0]))
            return tape.with_operations(new_ops)
        return transform

    def make_error_device(error):
        device = qfunction.functional_device(DefaultQubitJax(wires=2))
        return qfunction.with_preprocess(device, add_rot(error))

    def circuit():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(1))

    def loss(angle):
        real_device = qfunction.functional_device(DefaultQubitJax(wires=2))
        error_device = make_error_device(0.456045) # Give some rotation value

        real_val_fn = qfunction.qfunc(real_device)(circuit)
        corrected_error_fn = add_rot(angle)(qfunction.qfunc(error_device)(circuit))
        return (real_val_fn() - corrected_error_fn()) ** 2

    grad_loss = jax.grad(loss)
    np.testing.assert_allclose(grad_loss(-0.456045),  0.0)
    np.testing.assert_allclose(grad_loss(0.001), 0.090589, atol=1e-6, rtol=1e-6)


def test_metric_tensor_transform(): 
    device = qfunction.functional_device(DefaultQubitJax(wires=3))
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        qml.RX(0.2, wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        # layer 2
        qml.RZ(0.4, wires=0)
        qml.RZ(0.5, wires=2)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        qml.expval(qml.PauliY(2))

    a = qfunction.metric_tensor(circuit)(0.1)
    np.testing.assert_allclose(a, [[.25]])


def test_draw():
    device = qfunction.functional_device(DefaultQubitJax(wires=3))
    @qfunction.qfunc(device)
    def circuit(val):
        qml.RX(val, wires=0)

    result = qfunction.draw(circuit, charset='ascii')(0.123)
    assert result == ' 0: --RX(0.123)--|  \n'