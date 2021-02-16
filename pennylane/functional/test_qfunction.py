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
    """Check that the circuits execute correctly."""
    device = qfunction.functional_device(DefaultQubitJax(wires=1))
    # Make sure it still works similarly to a qnode.
    @qfunction.qfunc(device)
    def circuit(a):
        qml.RZ(a, wires=0)
        return qml.expval(qml.Hadamard(0))

    val = circuit(0.0)
    np.testing.assert_allclose(val, 0.707107, rtol=1e-6)


def test_custom_gradient_jax():
    """This test shows how one can do a device transformation to add a custom gradient in TF.
    """
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
    """This test shows how one can do a device transformation to add a custom gradient in TF.
    """
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
    """This test attempts to transform a tape with CNOT operations and turn it into 
    a tape with H and CZ operations instead.
    """
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

    # Here, we see that the drawing correctly takes into account the transformation.
    assert (
        qfunction.draw(circuit, charset='ascii')() == 
        ' 0: --X--+C--|     \n'
        ' 1: -----+X--| <Z> \n'
    )
    # Notice the added Hs and the CZ instead of CX
    assert (
        qfunction.draw(cnots_to_czs(circuit), charset='ascii')() == 
        ' 0: --X--+C-----|     \n'
        ' 1: --H--+Z--H--| <Z> \n'
    )

def test_learnable_transform():
    """Here, the goal is to test whether could learn a tape transformation in PL.

    The test is simple, just try to undo an added rotation gate after every gate. However, this
    example is actually very powerful, and showcases how a PL simulator could theortically be used
    to calibrate a real world QC.
    """

    # This is the angle we want to learn.
    error_angle = 0.456045

    def add_rot(angle):
        """Create a tape transform function that adds `angle` of rotation after every gate"""
        @qfunction.tape_transform
        def transform(tape):
            new_ops = []
            for o in tape.operations:
                new_ops.append(o)
                # Take each gate, and add an RX rotation afterwards unless the previous one
                # was also an RX gate.
                if not isinstance(o, qml.RX):
                    new_ops.append(qml.RX(angle, wires=o.wires[0]))
            return tape.with_operations(new_ops)
        return transform

    def make_error_device(error, device):
        """Create a quantum device that adds an `error` amout of RX dift after every gate."""
        return qfunction.with_preprocess(device, add_rot(error))

    def circuit():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(1))

    def loss(angle):
        # Create a normal device
        real_device = qfunction.functional_device(DefaultQubitJax(wires=2))

        # Take the real device, and create a new device that adds an error dift.
        error_device = make_error_device(error_angle, real_device)

        # This function will return the correct exepectation value
        real_expval_fn = qfunction.qfunc(real_device)(circuit)

        # This function will return the expectation value with the errors.
        error_expval_fn = qfunction.qfunc(error_device)(circuit)

        # Here, we transform our qfunc and add correcting rotation gates after every gate.
        # This function calculates the expectation value when correcting rotations after every gate.
        corrected_expval_fn = add_rot(angle)(error_expval_fn)

        # And create some loss value between what it should be and what our correction does.
        # One could imagine using a perfect super computing simulation to 
        # calibrate a real QC in a similar fasion.
        return (real_expval_fn() - corrected_expval_fn()) ** 2

    grad_loss = jax.grad(loss)
    # No gradient when the error is fully corrected
    np.testing.assert_allclose(grad_loss(-error_angle),  0.0)
    # Gradient when the correction is off.
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

    # Take the qfunc and draw the underlying
    result = qfunction.draw(circuit, charset='ascii')(0.123)
    assert result == ' 0: --RX(0.123)--|  \n'