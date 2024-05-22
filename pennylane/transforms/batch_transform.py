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
"""Contains tools and decorators for registering batch transforms."""
# pylint: disable=too-few-public-methods

import warnings
from typing import Callable, Tuple

import pennylane as qml
from pennylane.typing import ResultBatch

PostprocessingFn = Callable[[ResultBatch], ResultBatch]
QuantumTapeBatch = Tuple[qml.tape.QuantumScript]


def map_batch_transform(
    transform: Callable, tapes: QuantumTapeBatch
) -> Tuple[QuantumTapeBatch, PostprocessingFn]:
    """Map a transform over multiple tapes.

    Args:
        transform (Callable): the transform to be mapped
        tapes (Sequence[QuantumTape]): The sequence of tapes the
            transform should be applied to. Each tape in the sequence
            is transformed by the transform.

    **Example**

    Consider the following tapes:

    .. code-block:: python

        H = qml.Z(0) @ qml.Z(1) - qml.X(0)


        ops1 = [
            qml.RX(0.5, wires=0),
            qml.RY(0.1, wires=1),
            qml.CNOT(wires=(0,1))
        ]
        measurements1 = [qml.expval(H)]
        tape1 = qml.tape.QuantumTape(ops1, measurements1)

        ops2 = [qml.Hadamard(0), qml.CRX(0.5, wires=(0,1)), qml.CNOT((0,1))]
        measurements2 = [qml.expval(H + 0.5 * qml.Y(0))]
        tape2 = qml.tape.QuantumTape(ops2, measurements2)


    We can use ``map_batch_transform`` to map a single
    transform across both of the these tapes in such a way
    that allows us to submit a single job for execution:

    >>> tapes, fn = map_batch_transform(qml.transforms.hamiltonian_expand, [tape1, tape2])
    >>> dev = qml.device("default.qubit", wires=2)
    >>> fn(qml.execute(tapes, dev, qml.gradients.param_shift))
    [array(0.99500417), array(0.8150893)]

    .. warning::
        qml.transforms.map_batch_transform is deprecated and will be removed in a future release.
        Instead, a transform can be applied directly to a batch of tapes. See :func:`~.pennylane.transform` for more details.
    """

    warnings.warn(
        "qml.transforms.map_batch_transform is deprecated. "
        "Instead, a transform can be applied directly to a batch of tapes. "
        "See qml.transform for more details.",
        qml.PennyLaneDeprecationWarning,
    )

    execution_tapes = []
    batch_fns = []
    tape_counts = []

    for t in tapes:
        # Preprocess the tapes by applying transforms
        # to each tape, and storing corresponding tapes
        # for execution, processing functions, and list of tape lengths.
        new_tapes, fn = transform(t)
        execution_tapes.extend(new_tapes)
        batch_fns.append(fn)
        tape_counts.append(len(new_tapes))

    def processing_fn(res: ResultBatch) -> ResultBatch:
        """Applies a batch of post-processing functions to results.

        Args:
            res (ResultBatch): the results of executing a batch of circuits

        Returns:
            ResultBatch : results that have undergone classical post processing

        Closure variables:
            tape_counts: the number of tapes outputted from each application of the transform
            batch_fns: the post processing functions to apply to each sub-batch

        """
        count = 0
        final_results = []

        for idx, s in enumerate(tape_counts):
            # apply any transform post-processing
            new_res = batch_fns[idx](res[count : count + s])
            final_results.append(new_res)
            count += s

        return final_results

    return execution_tapes, processing_fn
