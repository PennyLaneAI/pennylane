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
"""
Contains the tape transform that splits non-commuting terms
"""
# pylint: disable=protected-access
import numpy as np
import pennylane as qml

from .batch_transform import batch_transform


@batch_transform
def split_non_commuting(tape):
    r"""
    Splits a qnode measuring non-commuting observables into groups of commuting observables.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode that contains a list of
            non-commuting observables to measure.

    Returns:
        qnode (pennylane.QNode) or tuple[List[.QuantumTape], function]: If a QNode is passed,
        it returns a QNode capable of handling non-commuting groups.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to restore the ordering of the inputs.

    **Example**

    This transform allows us to transform a QNode that measures
    non-commuting observables to *multiple* circuit executions
    with qubit-wise commuting groups:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=1)

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x,wires=0)
            return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]

    Instead of decorating the QNode, we can also create a new function that yields the same result in the following way:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x,wires=0)
            return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]

        circuit = qml.transforms.split_non_commuting(circuit)

    Internally, the QNode is split into groups of commuting observables when executed:

    >>> print(qml.draw(circuit)(0.5))
    0: ──RX(0.50)─┤  <X>
    \
    0: ──RX(0.50)─┤  <Z>

    Note that while internally multiple QNodes are created, the end result has the same ordering as the user provides in the return statement.
    Here is a more involved example where we can see the different ordering at the execution level but restoring the original ordering in the output:

    .. code-block:: python3

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit0(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return [qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliZ(0)),
                    qml.expval(qml.PauliY(1)),
                    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                    ]

    Drawing this QNode unveils the separate executions in the background

    >>> print(qml.draw(circuit0)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)──RX(0.79)─┤  <X>
    1: ─────────────────────┤  <Y>
    \
    0: ──RY(0.79)──RX(0.79)─┤  <Z> ╭<Z@Z>
    1: ─────────────────────┤      ╰<Z@Z>

    Yet, executing it returns the original ordering of the expectation values. The outputs correspond to
    :math:`(\langle \sigma_x^0 \rangle, \langle \sigma_z^0 \rangle, \langle \sigma_y^1 \rangle, \langle \sigma_z^0\sigma_z^1 \rangle)`.

    >>> circuit0([np.pi/4, np.pi/4])
    tensor([0.70710678, 0.5       , 0.        , 0.5       ], requires_grad=True)


    .. details::
        :title: Usage Details

        Internally, this function works with tapes. We can create a tape with non-commuting observables:

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(0))

            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        Now ``tapes`` is a list of two tapes, each for one of the non-commuting terms:

        >>> [t.observables for t in tapes]
        [[expval(PauliZ(wires=[0]))], [expval(PauliY(wires=[0]))]]

        The processing function becomes important when creating the commuting groups as the order of the inputs has been modified:

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
                qml.expval(qml.PauliX(0) @ qml.PauliX(1))
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliX(0))

            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        In this example, the groupings are ``group_coeffs = [[0,2], [1,3]]`` and ``processing_fn`` makes sure that the final output is of the same shape and ordering:

        >>> processing_fn(tapes)
        tensor([tensor(expval(PauliZ(wires=[0]) @ PauliZ(wires=[1])), dtype=object, requires_grad=True),
            tensor(expval(PauliX(wires=[0]) @ PauliX(wires=[1])), dtype=object, requires_grad=True),
            tensor(expval(PauliZ(wires=[0])), dtype=object, requires_grad=True),
            tensor(expval(PauliX(wires=[0])), dtype=object, requires_grad=True)],
        dtype=object, requires_grad=True)

    """
    # TODO: allow for samples and probs
    obs_fn = {qml.measurements.Expectation: qml.expval, qml.measurements.Variance: qml.var}

    obs_list = tape.observables
    return_types = [m.return_type for m in obs_list]

    if qml.measurements.Sample in return_types or qml.measurements.Probability in return_types:
        raise NotImplementedError(
            "When non-commuting observables are used, only `qml.expval` and `qml.var` are supported."
        )

    # If there is more than one group of commuting observables, split tapes
    groups, group_coeffs = qml.grouping.group_observables(obs_list, range(len(obs_list)))
    if len(groups) > 1:
        # make one tape per commuting group
        tapes = []
        for group in groups:
            new_tape = tape.__class__(
                tape._ops, (obs_fn[type](o) for type, o in zip(return_types, group)), tape._prep
            )

            tapes.append(new_tape)

        def reorder_fn(res):
            """re-order the output to the original shape and order"""
            new_res = qml.math.concatenate(res)
            reorder_indxs = qml.math.concatenate(group_coeffs)

            # in order not to mess with the outputs I am just permuting them with a simple matrix multiplication
            permutation_matrix = np.zeros((len(new_res), len(new_res)))
            for column, indx in enumerate(reorder_indxs):
                permutation_matrix[indx, column] = 1
            return qml.math.dot(permutation_matrix, new_res)

        return tapes, reorder_fn

    # if the group is already commuting, no need to do anything
    return [tape], lambda res: res[0]
