import abc

from pennylane import tape
from pennylane.devices.experimental import Device
from pennylane.typing import ResultBatch, TensorLike

from typing import Callable, Union, Tuple

Batch = Tuple[tape.QuantumTape]


class Executor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, circuits: Batch) -> ResultBatch:
        pass

    @abc.abstractproperty
    @property
    def configuration(self):
        """ """
        pass

    def __hash__(self):
        return hash()


ExecuteFn = Callable[[Batch], ResultBatch]


Primals = TensorLike
Tangents = TensorLike
JVP = TensorLike
JvpFn = Callable[[Batch, Primals, Tangents], JVP]


InterfaceBoundaryJvp = Callable[[Batch, ExecuteFn, JvpFn], ResultBatch]

VJP = TensorLike
VjpFn = Callable[
    [
        Batch,
    ]
]

InterfaceBoundaryVjp = Callable[[Batch, ExecuteFn, VjpFn], ResultBatch]
