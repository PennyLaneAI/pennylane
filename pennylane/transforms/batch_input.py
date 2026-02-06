# Copyright 2018-2022 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Batch transformation for multiple (non-trainable) input examples following issue #2037
"""
from collections.abc import Sequence

import numpy as np

import pennylane as qp
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.batch_params import _nested_stack, _split_operations
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


@transform
def batch_input(
    tape: QuantumScript,
    argnum: Sequence[int] | int,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Transform a circuit to support an initial batch dimension for gate inputs.

    In a classical ML application one needs to batch the non-trainable inputs of the network.
    This function executes the same analogue for a quantum circuit:
    separate circuit executions are created for each input, which are then executed
    with the *same* trainable parameters.

    The batch dimension is assumed to be the first rank of the
    non trainable tensor object. For a rank 1 feature space, the shape needs to be ``(Nt, x)``
    where ``x`` indicates the dimension of the features and ``Nt`` being the number of examples
    within a batch.
    Based on `arXiv:2202.10471 <https://arxiv.org/abs/2202.10471>`__.

    Args:
        tape (QNode or QuantumTape or Callable): Input quantum circuit to batch
        argnum (Sequence[int] or int): One or several index values indicating the position of the
            non-trainable batched parameters in the quantum tape.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the batched results.

    .. seealso:: :func:`~.batch_params`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.batch_input(argnum=1)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
            qml.RY(weights[1], wires=1)
            return qml.expval(qml.Z(1))

    >>> rng = np.random.default_rng(seed=1234)
    >>> x = rng.random((10, 2))
    >>> w = rng.random((2, ))
    >>> circuit(x, w) # doctest: +SKIP
    array([0.4855, 0.5854, 0.6954, 0.5384, 0.5838, 0.2737, 0.0233, 0.2253,
           0.6166, 0.0167])

    """

    argnum = tuple(argnum) if isinstance(argnum, (list, tuple)) else (int(argnum),)

    all_parameters = tape.get_parameters(trainable_only=False)
    argnum_params = [all_parameters[i] for i in argnum]

    if any(num in tape.trainable_params for num in argnum):
        # JAX arrays can't be marked as non-trainable, so don't raise this error
        # if the interface is JAX
        if qml.math.get_interface(*argnum_params) != "jax":
            raise ValueError(
                "Batched inputs must be non-trainable. Please make sure that the parameters indexed by "
                + "'argnum' are not marked as trainable."
            )

    batch_dims = np.unique([qml.math.shape(x)[0] for x in argnum_params])
    if len(batch_dims) != 1:
        raise ValueError(
            "Batch dimension for all gate arguments specified by 'argnum' must be the same."
        )

    batch_size = batch_dims[0]

    output_tapes = []
    for ops in _split_operations(tape.operations, all_parameters, argnum, batch_size):
        new_tape = qml.tape.QuantumScript(
            ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
        )
        output_tapes.append(new_tape)

    def processing_fn(res):
        return _nested_stack(res)

    return output_tapes, processing_fn
