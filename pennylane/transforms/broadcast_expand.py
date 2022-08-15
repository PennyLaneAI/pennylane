# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the tape expansion function for expanding a
broadcasted tape into multiple tapes."""
import pennylane as qml
from .batch_transform import batch_transform


@batch_transform
def broadcast_expand(tape):
    r"""Expand a broadcasted tape into multiple tapes
    and a function that stacks and squeezes the results.

    Args:
        tape (.QuantumTape): Broadcasted tape to be expanded

    .. warning::

        Currently, not all templates have been updated to support a batch
        dimension. If you run into an error attempting to use a template
        with this transform, please open a GitHub issue detailing
        the error.

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes that produce one of the results of the broadcasted tape each,
        and a function that stacks and squeezes the tape execution results.

    This expansion function is used internally whenever a device does not
    support broadcasting.

    **Example**

    We may use ``broadcast_expand`` on a ``QNode`` to separate it
    into multiple calculations. For this we will provide ``qml.RX`` with
    the ``ndim_params`` attribute that allows the operation to detect
    broadcasting, and set up a simple ``QNode`` with a single operation and
    returned expectation value:

    >>> qml.RX.ndim_params = (0,)
    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    >>> def circuit(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.PauliZ(0))

    We can then call ``broadcast_expand`` on the QNode and store the
    expanded ``QNode``:

    >>> expanded_circuit = qml.transforms.broadcast_expand(circuit)

    Let's use the expanded QNode and draw it for broadcasted parameters
    with broadcasting axis of length ``3`` passed to ``qml.RX``:

    >>> x = pnp.array([0.2, 0.6, 1.0], requires_grad=True)
    >>> print(qml.draw(expanded_circuit)(x))
    0: ──RX(0.20)─┤  <Z>
    0: ──RX(0.60)─┤  <Z>
    0: ──RX(1.00)─┤  <Z>

    Executing the expanded ``QNode`` results in three values, corresponding
    to the three parameters in the broadcasted input ``x``:

    >>> expanded_circuit(x)
    tensor([0.98006658, 0.82533561, 0.54030231], requires_grad=True)

    We also can call the transform manually on a tape:

    >>> with qml.tape.QuantumTape() as tape:
    >>>     qml.RX(pnp.array([0.2, 0.6, 1.0], requires_grad=True), wires=0)
    >>>     qml.expval(qml.PauliZ(0))
    >>> tapes, fn = qml.transforms.broadcast_expand(tape)
    >>> tapes
    [<QuantumTape: wires=[0], params=1>, <QuantumTape: wires=[0], params=1>, <QuantumTape: wires=[0], params=1>]
    >>> fn(qml.execute(tapes, qml.device("default.qubit", wires=1), None))
    array([0.98006658, 0.82533561, 0.54030231])
    """

    num_tapes = tape.batch_size
    if num_tapes is None:
        raise ValueError("The provided tape is not broadcasted.")

    # Note that these unbatched_params will have shape (#params, num_tapes)
    unbatched_params = []
    for op in tape.operations + tape.observables:
        for j, p in enumerate(op.data):
            if op.batch_size and qml.math.ndim(p) != op.ndim_params[j]:
                unbatched_params.append(qml.math.unstack(p))
            else:
                unbatched_params.append([p] * num_tapes)

    output_tapes = []
    for p in zip(*unbatched_params):
        new_tape = tape.copy(copy_operations=True)
        new_tape.set_parameters(p, trainable_only=False)
        output_tapes.append(new_tape)

    return output_tapes, lambda x: qml.math.squeeze(qml.math.stack(x))
