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
"""This module contains the tape expansion function for unbroadcasting a
broadcasted tape into multiple tapes."""
import pennylane as qml
from pennylane.transforms.batch_transform import batch_transform


@batch_transform
def unbroadcast_expand(tape):
    r"""Expand a broadcasted tape into multiple unbroadcasted tapes
    and a function that stacks and squeezes the results.

    Args:
        tape (.QuantumTape): Broadcasted tape to be expanded
    Returns:
        list[.QuantumTape]: Unbroadcasted tapes that produce one of the
        results of the broadcasted tape each
        callable: Function that stacks the results

    This expansion function is used internally whenever a device does not
    support broadcasting.

    **Example**

    We may use ``unbroadcast_expand`` manually on a tape to separate it
    into multiple calculations. For this we will provide ``qml.RX`` with
    the ``ndim_params`` attribute that allows the operation to detect
    broadcasting.

    >>> qml.RX.ndim_params = (0,)
    >>> with qml.tape.QuantumTape() as tape:
    >>>     qml.RX(np.array([0.2, 0.6, 1.0], requires_grad=True), wires=0)
    >>>     qml.expval(qml.PauliZ(0))
    >>> tapes, fn = qml.transforms.unbroadcast_expand(tape)
    >>> tapes
    [<QuantumTape: wires=[0], params=1>,
     <QuantumTape: wires=[0], params=1>,
     <QuantumTape: wires=[0], params=1>]
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
