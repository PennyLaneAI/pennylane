from pennylane import tape
from pennylane.devices.experimental import Device
from pennylane.typing import ResultBatch, TensorLike

from typing import Callable, Union, Tuple

Batch = Tuple[tape.QuantumTape]

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
