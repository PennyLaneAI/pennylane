from pennylane.tape.tapes.tape import QuantumTape
from pennylane import math as qmath
import pennylane as qml
import jax
import numpy as np

def single_tape(device):
    """Transform a device from running a batch of tapes to a single tape"""
    def wrapper(tape):
        return device([tape])[0]
    return wrapper


class QFunc:
    """A Quantum Function."""

    def __init__(self, device, create_tape):
        self.device = device
        self.create_tape = create_tape

    def __call__(self, *args, **kwargs):
        return self.device([self.create_tape(*args, **kwargs)])[0]


def _get_tape(func):
    """Get the tape from a tape generating function.

    This function creates a tape context before calling
    `func` and the created tape is then returned after
    executing `func`.

    Args:
        func: A function that generates a tape.
    Returns:
        A function when called returns the generated tape.
    """
    def wrapper(*args, **kwargs):
        tape = QuantumTape()
        with tape:
            func(*args, **kwargs)
        return tape

    return wrapper


def qfunc(device):
    """A Quantum function. Used as a decorator similarly to `qml.qnode`.

    Example:


    device = functional_device(...)

    @qfunc(device)
    def circuit(a):
        qml.RX(a, wires=0)
        qml.expval(qml.PauliZ(0))

    print(circuit(0.1))

    Args:
        device: A functional device that executes batches of tapes.

    Returns:
        wrapper: A function that takes a tape generating method and 
            returns a Qfunc.
    """

    def wrapper(func):
        return QFunc(device, _get_tape(func))

    return wrapper


def device_transform(transform):
    """Transform a device into a new device. Used to add custom gradients

    Args:
        transform: A function with signature device -> device.

    returns:
        wrapper: A function that transforms either QFunc -> QFunc or device -> device.
    """

    def wrapper(obj):
        if isinstance(obj, QFunc):
            return QFunc(transform(obj.device), obj.create_tape)
        else:
            # Assume obj is a device, so we transform it directly.
            return transform(obj)

    return wrapper


def tape_transform(transform):
    """Define a function that transforms a QuantuMTape into another QuantumTape.

    This decorator allows the transform to be used as a decorator on a qfunc.
    """
    def wrapper(obj):
        if isinstance(obj, QFunc):
            return QFunc(
                obj.device, 
                lambda *args, **kwargs: transform(obj.create_tape(*args, **kwargs))
            )
        else:
            # Assume we have an actual tape.
            return transform(obj)
    return wrapper

def functional_device(device):
    """Transform old device to functional device

    Args:
        device: A pennylane.Device
    Returns:
        A function that tapes a list of tapes and executes
        them all with the given device.
    """
    def batch_execute(tapes):
        def execute(tape):
            device.reset()
            return qmath.squeeze(device.execute(tape))
        return list(map(execute, tapes))
    return batch_execute

def draw(qfunc, charset='unicode'):
    def wrapper(*args, **kwargs):
        return qfunc.create_tape(*args, **kwargs).draw(charset=charset)
    return wrapper


def with_preprocess(device, tape_transform):
    """Create a device that runs `tape_transform` on every tape before execution"""
    return lambda tapes: device(list(map(tape_transform, tapes)))

def with_batch_preprocess(device, tapes_transform):
    """Create a device that runs `tapes_transform` on the batch of tapes before execution"""
    return lambda tapes: devices(tapes_transform(tapes))

def _get_classical_jacobian_jax(_qfunc):
    """Helper function to extract the Jacobian
    matrix of the classical part of a QFunc"""

    def classical_preprocessing(*args, **kwargs):
        """Returns the trainable gate parameters for
        a given QNode input"""
        
        return qml.math.stack(_qfunc.create_tape(*args, **kwargs).get_parameters())

    return jax.jacobian(classical_preprocessing)

def metric_tensor(_qfunc, diag_approx=False, wrt=None):

    def _metric_tensor_fn(*args, **kwargs):
        jac = qml.math.stack(_get_classical_jacobian_jax(_qfunc)(*args, **kwargs))
        jac = qml.math.reshape(jac, [-1, 1])

        wrt, perm = np.nonzero(qml.math.toarray(jac))
        perm = np.argsort(np.argsort(perm))

        tape = _qfunc.create_tape(*args, **kwargs)

        metric_tensor_tapes, processing_fn = qml.tape.transforms.metric_tensor(
            tape,
            diag_approx=diag_approx,
            wrt=wrt.tolist(),
        )

        res = _qfunc.device(metric_tensor_tapes)
        mt = processing_fn(res)

        # permute rows ad columns
        mt = qml.math.gather(mt, perm)
        mt = qml.math.gather(qml.math.T(mt), perm)
        return mt

    return _metric_tensor_fn

