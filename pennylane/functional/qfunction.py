from pennylane.tape.tapes.tape import QuantumTape

def squeeze(val):
    try:
        return val.squeeze()
    except AttributeError:
        # Because tensorflow just has to be the problem child doesn't it?
        import tensorflow as tf
        return tf.squeeze(val)

class QFunc():
    def __init__(self, device, func):
        self.device = device
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.device(self.func(*args, **kwargs))

    def device_transform(self, transform):
        return QFunc(transform(self.device), self.func)

def _get_tape(func):
    def wrapper(*args, **kwargs):
        tape = QuantumTape()
        with tape:
            func(*args, **kwargs)
        return tape     
    return wrapper

def qfunc(device):
    def wrapper(func):
        return QFunc(device, _get_tape(func))
    return wrapper

def device_transform(func):
    def wrapper(obj):
        if isinstance(obj, QFunc):
            return obj.device_transform(func)
        else:
            return func(obj)
    return wrapper

def functional_device(device):
    def execute(circuit):
        device.reset()
        return squeeze(device.execute(circuit))
    return execute

