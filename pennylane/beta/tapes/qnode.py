from functools import lru_cache

from pennylane.beta.tapes import QuantumTape

# from tf_interface import TFInterface
# from torch_interface import TorchInterface
from pennylane.beta.interfaces.autograd import AutogradInterface
from pennylane.beta.interfaces.tf import TFInterface
from pennylane.beta.interfaces.torch import TorchInterface


INTERFACE_MAP = {"tf": TFInterface, "torch": TorchInterface, "autograd": AutogradInterface}


class QNode:
    def __init__(self, func, dev, interface=None, diff_method="parameter-shift"):
        self.func = func
        self.dev = dev
        self.qtape = None
        self.interface = interface
        self.diff_method = diff_method

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context,
        ensuring the operations get queued."""
        with QuantumTape() as self.qtape:
            # Note that the quantum function doesn't
            # return anything. We assume that all classical
            # pre-processing happens *prior* to the quantum
            # tape, and the tape is the final step in the function.
            self.func(*args, **kwargs)

        if self.interface in INTERFACE_MAP:
            self.qtape = INTERFACE_MAP[self.interface].apply(self.qtape)

    def __call__(self, *args, **kwargs):
        # construct the tape
        self.construct(args, kwargs)

        # execute the tape
        return self.qtape.execute(device=self.dev)


def qnode(device, *, interface="autograd", diff_method="parameter-shift"):
    """Decorator for creating QNodes."""

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        return QNode(func, device, interface=interface, diff_method=diff_method)

    return qfunc_decorator
