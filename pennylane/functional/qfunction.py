from pennylane.tape.tapes.tape import QuantumTape


def squeeze(val):
    """Squeeze a tensor"""
    try:
        return val.squeeze()
    except AttributeError:
        # Because tensorflow just has to be the problem child doesn't it?
        import tensorflow as tf

        return tf.squeeze(val)


class QFunc:
    """A Quantum Function."""

    def __init__(self, device, func):
        self.device = device
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.device(self.func(*args, **kwargs))


def _get_tape(func):
    def wrapper(*args, **kwargs):
        tape = QuantumTape()
        with tape:
            func(*args, **kwargs)
        return tape

    return wrapper


def qfunc(device):
    """A Quantum function. Used as a decorato"""

    def wrapper(func):
        return QFunc(device, _get_tape(func))

    return wrapper


def device_transform(transform):
    """Transform a device into a new device. Used to add custom gradients"""

    def wrapper(obj):
        if isinstance(obj, QFunc):
            return QFunc(transform(obj.device), obj.func)
        else:
            return transform(obj)

    return wrapper


def functional_device(device):
    """Transform old device to functional device"""

    def execute(circuit):
        device.reset()
        return squeeze(device.execute(circuit))

    return execute
