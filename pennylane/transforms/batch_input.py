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
from typing import Union, Sequence, Callable, Tuple

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.batch_transform import batch_transform


@batch_transform
def batch_input(
    tape: Union[qml.tape.JacobianTape, qml.QNode],
    argnum: Union[Sequence[int], int] = 0,
) -> Tuple[Sequence[qml.tape.JacobianTape], Callable]:
    """
    Transform a QNode to support an initial batch dimension for gate inputs.

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
        tape (.tape.QuantumTape or .QNode): Input quantum circuit to batch
        argnum (Sequence[int] or int): One or more index value on all gate parameters
            indicating the location of the non-trainable batched inputs within the input
            argument sequence of the circuit. By default first argument is assumed to be
            the only batched input.

    Returns:
        Sequence[Sequence[.tape.QuantumTape], Callable]: list of tapes arranged
        according to unbatched inputs and a callable function to batch the results.

    .. seealso:: :func:`~.batch_params`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires = 2, shots=None)

        @batch_input(argnum=0)
        @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires = range(2), rotation="Y")
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            return qml.expval(qml.PauliZ(1))

    >>> x = np.random.uniform(0,1,(10,2))
    >>> x.requires_grad = False
    >>> w = np.random.uniform(0,1,2)
    >>> circuit(x, w)
    <tf.Tensor: shape=(10,), dtype=float64, numpy=
    array([0.17926078, 0.7480163 , 0.47816999, 0.50381628, 0.349178  ,
           0.17511444, 0.03769436, 0.19180259, 0.75867188, 0.55335748])>
    """
    parameters = tape.get_parameters(trainable_only=False)

    argnum = tuple(argnum) if isinstance(argnum, (list, tuple)) else (int(argnum),)

    non_trainable, trainable = [], []
    for idx, param in enumerate(parameters):
        if idx in argnum:
            non_trainable.append(param)
        else:
            trainable.append(param)

    if len(np.unique([qml.math.shape(x)[0] for x in non_trainable])) != 1:
        raise ValueError(
            "Batch dimension for all gate arguments specified by 'argnum' must be the same."
        )

    outputs = []
    for inputs in zip(*non_trainable):
        outputs += [list(inputs) + trainable]

    # Construct new output tape with unstacked inputs
    output_tapes = []
    for params in outputs:
        new_tape = tape.copy(copy_operations=True)
        new_tape.set_parameters(params, trainable_only=False)
        output_tapes.append(new_tape)

    return output_tapes, lambda x: qml.math.squeeze(qml.math.stack(x))
