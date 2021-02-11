from pennylane.tape.tapes.tape import QuantumTape
from pennylane import math as qmath
from pennylane.tape.transforms import metric_tensor as mt
import pennylane as qml
import jax
import numpy as np

def single_tape(device):
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
    def wrapper(*args, **kwargs):
        tape = QuantumTape()
        with tape:
            func(*args, **kwargs)
        return tape

    return wrapper


def qfunc(device):
    """A Quantum function. Used as a decorator

    Args:
        device: A device.

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
    """Transform old device to functional device"""
    def batch_execute(circuits):
        def execute(circuit):
            device.reset()
            return qmath.squeeze(device.execute(circuit))
        return list(map(execute, circuits))
    return batch_execute

def draw(qfunc, charset='unicode'):
    def wrapper(*args, **kwargs):
        return qfunc.create_tape(*args, **kwargs).draw(charset=charset)
    return wrapper


def with_preprocess(device, tape_transform):
    return lambda tapes: device(list(map(tape_transform, tapes)))

def with_batch_preprocess(device, tapes_transform):
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

