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
from typing import Sequence, Callable
from functools import reduce

import pennylane as qml

from pennylane.transforms import transform


def null_postprocessing(results):
    """A postprocesing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def split_non_commuting(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""
    Splits a qnode measuring non-commuting observables into groups of commuting observables.

    Args:
        tape (QNode or QuantumTape or Callable): A circuit that contains a list of
            non-commuting observables to measure.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]: The transformed circuit as described in
        :func:`qml.transform <pennylane.transform>`.

    **Example**

    This transform allows us to transform a QNode that measures non-commuting observables to
    *multiple* circuit executions with qubit-wise commuting groups:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x,wires=0)
            return [qml.expval(qml.X(0)), qml.expval(qml.Z(0))]

    Instead of decorating the QNode, we can also create a new function that yields the same result
    in the following way:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x,wires=0)
            return [qml.expval(qml.X(0)), qml.expval(qml.Z(0))]

        circuit = qml.transforms.split_non_commuting(circuit)

    Internally, the QNode is split into groups of commuting observables when executed:

    >>> print(qml.draw(circuit)(0.5))
    0: ──RX(0.50)─┤  <X>
    \
    0: ──RX(0.50)─┤  <Z>

    Note that while internally multiple QNodes are created, the end result has the same ordering as
    the user provides in the return statement.
    Here is a more involved example where we can see the different ordering at the execution level
    but restoring the original ordering in the output:

    .. code-block:: python3

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit0(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return [qml.expval(qml.X(0)),
                    qml.expval(qml.Z(0)),
                    qml.expval(qml.Y(1)),
                    qml.expval(qml.Z(0) @ qml.Z(1)),
                    ]

    Drawing this QNode unveils the separate executions in the background

    >>> print(qml.draw(circuit0)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)──RX(0.79)─┤  <X>
    1: ─────────────────────┤  <Y>
    \
    0: ──RY(0.79)──RX(0.79)─┤  <Z> ╭<Z@Z>
    1: ─────────────────────┤      ╰<Z@Z>

    Yet, executing it returns the original ordering of the expectation values. The outputs
    correspond to
    :math:`(\langle \sigma_x^0 \rangle, \langle \sigma_z^0 \rangle, \langle \sigma_y^1 \rangle,
    \langle \sigma_z^0\sigma_z^1 \rangle)`.

    >>> circuit0([np.pi/4, np.pi/4])
    [0.7071067811865475, 0.49999999999999994, 0.0, 0.49999999999999994]


    .. details::
        :title: Usage Details

        Internally, this function works with tapes. We can create a tape with non-commuting
        observables:

        .. code-block:: python3

            measurements = [qml.expval(qml.Z(0)), qml.expval(qml.Y(0))]
            tape = qml.tape.QuantumTape(measurements=measurements)

            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        Now ``tapes`` is a list of two tapes, each for one of the non-commuting terms:

        >>> [t.observables for t in tapes]
        [[expval(Z(0))], [expval(Y(0))]]

        The processing function becomes important when creating the commuting groups as the order
        of the inputs has been modified:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.X(1)),
                qml.expval(qml.Z(0)),
                qml.expval(qml.X(0))
            ]
            tape = qml.tape.QuantumTape(measurements=measurements)

            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        In this example, the groupings are ``group_coeffs = [[0,2], [1,3]]`` and ``processing_fn``
        makes sure that the final output is of the same shape and ordering:

        >>> processing_fn([t.measurements for t in tapes])
        (expval(Z(0) @ Z(1)),
        expval(X(0) @ X(1)),
        expval(Z(0)),
        expval(X(0)))

        Measurements that accept both observables and ``wires`` so that e.g. ``qml.counts``,
        ``qml.probs`` and ``qml.sample`` can also be used. When initialized using only ``wires``,
        these measurements are interpreted as measuring with respect to the observable
        ``qml.Z(wires[0])@qml.Z(wires[1])@...@qml.Z(wires[len(wires)-1])``

        .. code-block:: python3

            measurements = [
                qml.expval(qml.X(0)),
                qml.probs(wires=[1]),
                qml.probs(wires=[0, 1])
            ]
            tape = qml.tape.QuantumTape(measurements=measurements)

            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        This results in two tapes, each with commuting measurements:

        >>> [t.measurements for t in tapes]
        [[expval(X(0)), probs(wires=[1])], [probs(wires=[0, 1])]]
    """

    # Construct a list of observables to group based on the measurements in the tape
    obs_list = []
    for obs in tape.observables:
        # observable provided for a measurement
        if isinstance(obs, qml.operation.Operator):
            obs_list.append(obs)
        # measurements using wires instead of observables
        else:
            # create the PauliZ tensor product observable when only wires are provided for the
            # measurements
            obs_wires = obs.wires if obs.wires else tape.wires
            pauliz_obs = qml.prod(*(qml.Z(wire) for wire in obs_wires))

            obs_list.append(pauliz_obs)

    # If there is more than one group of commuting observables, split tapes
    _, group_coeffs = qml.pauli.group_observables(obs_list, range(len(obs_list)))

    if len(group_coeffs) > 1:
        # make one tape per commuting group
        tapes = []
        for indices in group_coeffs:
            new_tape = tape.__class__(
                tape.operations, (tape.measurements[i] for i in indices), shots=tape.shots
            )

            tapes.append(new_tape)

        def reorder_fn(res):
            """re-order the output to the original shape and order"""
            # determine if shot vector is used
            if len(tapes[0].measurements) == 1:
                shot_vector_defined = isinstance(res[0], tuple)
            else:
                shot_vector_defined = isinstance(res[0][0], tuple)

            res = list(zip(*res)) if shot_vector_defined else [res]

            reorder_indxs = qml.math.concatenate(group_coeffs)

            res_ordered = []
            for shot_res in res:
                # flatten the results
                shot_res = reduce(
                    lambda x, y: x + list(y) if isinstance(y, (tuple, list)) else x + [y],
                    shot_res,
                    [],
                )

                # reorder the tape results to match the user-provided order
                shot_res = list(zip(range(len(shot_res)), shot_res))
                shot_res = sorted(shot_res, key=lambda r: reorder_indxs[r[0]])
                shot_res = [r[1] for r in shot_res]

                res_ordered.append(tuple(shot_res))

            return tuple(res_ordered) if shot_vector_defined else res_ordered[0]

        return tapes, reorder_fn

    # if the group is already commuting, no need to do anything
    return [tape], null_postprocessing
