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
from typing import Callable, Sequence, Tuple, Union

import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape
from pennylane.transforms.batch_transform import batch_transform
from pennylane.transforms.batch_params import _nested_stack


@batch_transform
def batch_input(
    tape: Union[QuantumTape, qml.QNode],
    argnum: Union[Sequence[int], int],
) -> Tuple[Sequence[QuantumTape], Callable]:
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
        tape (.QuantumTape or .QNode): Input quantum circuit to batch
        argnum (Sequence[int] or int): One or several index values indicating the position of the
            non-trainable batched parameters in the quantum tape.

    Returns:
        Sequence[Sequence[.QuantumTape], Callable]: list of tapes arranged
        according to unbatched inputs and a callable function to batch the results.

    .. seealso:: :func:`~.batch_params`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2, shots=None)

        @qml.batch_input(argnum=1)
        @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
            qml.RY(weights[1], wires=1)
            return qml.expval(qml.PauliZ(1))

    >>> x = tf.random.uniform((10, 2), 0, 1)
    >>> w = tf.random.uniform((2,), 0, 1)
    >>> circuit(x, w)
    <tf.Tensor: shape=(10,), dtype=float64, numpy=
    array([0.46230079, 0.73971315, 0.95666004, 0.5355225 , 0.66180948,
            0.44519553, 0.93874261, 0.9483197 , 0.78737918, 0.90866411])>
    """
    # pylint: disable=too-many-branches
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

    new_preps = [[] for _ in range(batch_size)]
    new_ops = [[] for _ in range(batch_size)]
    new_measurements = [[] for _ in range(batch_size)]

    idx = 0
    for prep in tape.prep:
        # determine if any parameters of the operator are batched
        if any(i in argnum for i in range(idx, idx + len(prep.data))):
            for b in range(batch_size):
                new_params = tuple(
                    all_parameters[i][b] if i in argnum else all_parameters[i]
                    for i in range(idx, idx + len(prep.data))
                )
                new_prep = qml.ops.functions.bind_new_parameters(prep, new_params)
                new_preps[b].append(new_prep)
        else:
            # no batching in the operator; don't copy
            for b in range(batch_size):
                new_preps[b].append(prep)

        idx += len(prep.data)

    ops = [op for op in tape.operations if op not in tape.prep]
    for op in ops:
        # determine if any parameters of the operator are batched
        if any(i in argnum for i in range(idx, idx + len(op.data))):
            for b in range(batch_size):
                new_params = tuple(
                    all_parameters[i][b] if i in argnum else all_parameters[i]
                    for i in range(idx, idx + len(op.data))
                )
                new_op = qml.ops.functions.bind_new_parameters(op, new_params)
                new_ops[b].append(new_op)
        else:
            # no batching in the operator; don't copy
            for b in range(batch_size):
                new_ops[b].append(op)

        idx += len(op.data)

    for m in tape.measurements:
        # determine if any parameters of the measurement are batched
        if m.obs is not None and any(i in argnum for i in range(idx, idx + len(m.obs.data))):
            for b in range(batch_size):
                new_params = tuple(
                    all_parameters[i][b] if i in argnum else all_parameters[i]
                    for i in range(idx, idx + len(m.obs.data))
                )
                new_op = qml.ops.functions.bind_new_parameters(m.obs, new_params)
                new_m = m.copy()
                new_m.obs = new_op
                new_measurements[b].append(new_m)
        else:
            # no batching in the operator; don't copy
            for b in range(batch_size):
                new_measurements[b].append(m)

        idx += 0 if m.obs is None else len(m.obs.data)

    output_tapes = []
    for prep, ops, ms in zip(new_preps, new_ops, new_measurements):
        new_tape = qml.tape.QuantumScript(ops, ms, prep, shots=tape.shots)
        new_tape.trainable_params = tape.trainable_params
        output_tapes.append(new_tape)

    def processing_fn(res):
        if qml.active_return():
            return _nested_stack(res)

        return qml.math.squeeze(qml.math.stack(res))

    return output_tapes, processing_fn
