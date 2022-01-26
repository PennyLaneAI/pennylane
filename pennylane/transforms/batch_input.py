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
    argnum: Union[Sequence[int], int] = None,
) -> Tuple[Sequence[qml.tape.JacobianTape], Callable]:
    """
    In a classical ML application one needs to batch the non-trainable inputs of the network.
    This function executes the same analogue for a quantum circuit where separate circuit
    executions are created for each input examples  which will be executed with the same
    trainable inputs. The batch dimension assummed to be the first rank of the
    non trainable tensor object where for a rank 1 feature space the shape needs to be (Nt, x)
    where `x` indicates the dimension of the features and `Nt` being the number of examples
    within a batch.

    * Example *

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


    Parameters
    ----------
    tape (qml.tape.JacobianTape or qml.QNode): Record of the inputs to be executed
    argnum (Sequence[int] or int) : One or more argument numbers indicating the
        location of the batched inputs. As default first argument is assumed to be the only
        batched input.

    Returns
    -------
    Sequence[Sequence[qml.tape.JacobianTape], Callable]
        list of tapes arranged according to unbatched inputs and a callable function
        to batch the results.
    """
    parameters = tape.get_parameters(trainable_only=False)

    if argnum is None:
        return parameters, lambda x: x

    argnum = tuple(argnum) if isinstance(argnum, (list, tuple)) else (int(argnum),)

    non_trainable, trainable = [], []
    for idx, param in enumerate(parameters):
        if idx in argnum:
            non_trainable.append(param)
        else:
            trainable.append(param)

    assert (
        len(np.unique([qml.math.shape(x)[0] for x in non_trainable])) == 1
    ), "Batch dimension for all non-trainable inputs must be the same."

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
