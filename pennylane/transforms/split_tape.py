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
Contains the ``split_tape`` tape transform
"""
from collections import defaultdict
from dataclasses import dataclass

# pylint: disable=protected-access
from typing import Dict, List

import pennylane as qml
from pennylane.measurements import (
    ClassicalShadowMP,
    ExpectationMP,
    MeasurementProcess,
    ShadowExpvalMP,
)
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, SProd, Sum
from pennylane.pauli import group_observables, is_pauli_word
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


# pylint: disable=too-many-branches, too-many-statements
def split_tape(tape: QuantumTape, group=True):
    """Split the given tape into multiple tapes containing qubit-wise commuting measurements and/or
    measurements with non-overlapping wires.

    Args:
        tape (.QuantumTape): the original tape
        group (bool): Whether to compute disjoint groups of commuting Pauli-based measurements
            and/or groups of measurements with non-overlapping wires, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.PauliY(2) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(H)

    We can use the ``split_tape`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.split_tape(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    array(-0.5)

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

    With grouping, the Hamiltonian gets split into two groups of observables (here ``[qml.PauliZ(0)]`` and
    ``[qml.PauliX(1), qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.split_tape(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.PauliZ(0)]``, ``[qml.PauliX(1)]`` and ``[qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.split_tape(tape, group=False)
    >>> len(tapes)
    3

    Given a Sum operator,

    .. code-block:: python3

        S = qml.op_sum(qml.prod(qml.PauliY(2), qml.PauliZ(1)), qml.s_prod(0.5, qml.PauliZ(2)), qml.PauliZ(1))

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(S)
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))
            qml.expval(qml.PauliZ(2))

    We can use the ``split_tape`` transform to generate new tapes and a classical
    post-processing function to speed-up the computation of the expectation value of the `Sum`.

    >>> tapes, fn = qml.transforms.split_tape(tape, group=False)
    >>> [tape.measurements for tape in tapes]
    [expval(PauliY(wires=[2]) @ PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[2]))]
    [expval(PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[0]))]
    [expval(PauliX(wires=[1]))]

    Five tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last two contain the remaining observables. Note that the scalars of the scalar products
    have been removed. In the processing function, these values will be multiplied by the result obtained
    from executing the tapes.

    Additionally, the observable expval(PauliZ(wires=[2])) occurs twice in the original tape, but only once
    in the transformed tapes. When there are multiple identical measurements in the circuit, the measurement
    is performed once and the outcome is copied when obtaining the final result. This will also be resolved
    when the processing function is applied.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    (array(-0.5), array(0.), array(0.), array(-1.))

    Fewer tapes can be constructed by grouping observables acting on different wires. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        S = qml.op_sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(S)

    With grouping, the Sum gets split into two groups of observables (here
    ``[qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1))]`` and ``[qml.s_prod(3, qml.PauliX(0))]``):

    >>> tapes, fn = qml.transforms.split_tape(tape, group=True)
    >>> [tape.measurements for tape in tapes]
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1]))]
    [expval(PauliX(wires=[0]))]
    """
    m_grouping = _MGroup(tape=tape)

    tapes, processing_fn = m_grouping.get_tapes_and_processing_fn(group=group)

    return tapes, processing_fn


# pylint: disable=too-few-public-methods
class _MGroup:
    """Utils class used to group measurements.

    If the observables are pauli words, the groups contain observables that commute with each other.
    All the other observables are grouped into groups with non overlapping wires.

    Args:
        measurements (List[MeasurementProcess]): List of measurements to group together.

    Attributes:
        queue (Dict[int, MData]): Dictionary containing the information of all the measurements.
            It has the following structure:

            - keys (int): hash of the measurement
            - values (MData): object containing information about the measurement:
                - m (MeasurementProcess): the measurement class
                - data (dict):
                    - keys (int): position of the measurement in the return statement of the
                        original tape
                    - values (float): coefficient that will be multiplied to the result obtained
                        from the execution
        mdata_groups ()

    **Example:**
    Let's define the following QNode:

    .. code-block:: python

        import pennylane as qml
        from pennylane.transforms.split_tape import _MGroup

        dev = qml.device("default.qubit", wires=2)

        H = 3 * qml.PauliX(0) + qml.PauliZ(0) + qml.PauliX(0) + qml.PauliY(1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(H), qml.expval(qml.PauliX(0))

    Now we can use the :class:`_MeasurementsGrouping` class to group the measurements together:

    >>> circuit.construct((), {})  # construct tape
    >>> m_group = _MGroup(circuit.tape)
    >>> list(m_group.queue.values())  # we don't need to measurement hashes
    [_MGroup.MData(m=expval(PauliX(wires=[0])), data={0: 4, 1: 1}, group=None),
    _MGroup.MData(m=expval(PauliZ(wires=[0])), data={0: 1}, group=None),
    _MGroup.MData(m=expval(PauliY(wires=[1])), data={0: 1}, group=None)]

    The queue contains two items, one for each different measurement. Each item is an MData class
    that contains a ``data`` attribute with multiple pairs index-coefficient. For
    example, this attribute tells us that the ``expval(PauliX(0))`` measurement appears four times
    in the first return value and once in the second.

    The _MGroup class already groups equal measurements together during instantiation, reducing
    the total amount of executed measurements. We can obtain these measurements by using the
    ``get_measurements`` method, which returns a list of lists containing the grouped measurements.
    Setting ``group=False`` will skip the expensive grouping logic and return the measurements
    from the queue:

    >>> tapes = m_group.get_tapes(group=False)
    >>> [tape.measurements for tape in tapes]
    [[expval(PauliX(wires=[0]))],
    [expval(PauliZ(wires=[0]))],
    [expval(PauliY(wires=[1]))]]

    One must know that when computing expectation values of Hamiltonians, if the Hamiltonian
    previously computed its qubit-wise commuting groups, this information will be used even if
    ``group=False``:

    >>> H.compute_grouping()
    >>> m_group = _MGroup(circuit.tape)
    >>> tapes = m_group.get_tapes(group=False)
    >>> [tape.measurements for tape in tapes]
    [[expval(PauliX(wires=[0])), expval(PauliY(wires=[1]))],
    [expval(PauliZ(wires=[0]))]]

    ``PauliX(0)`` and ``PauliY(1)`` have been grouped together.

    Let's create a new QNode and group the measurements:

    .. code-block:: python

        H = 4 * qml.PauliX(0) + qml.PauliZ(0) + qml.PauliY(1)
        S = qml.op_sum(qml.s_prod(4, qml.PauliX(0)), qml.PauliZ(0), qml.PauliY(1))

        @qml.qnode(dev)
        def circuit():
            return qml.expval(H), qml.expval(qml.PauliX(1)), qml.expval(S), qml.expval(H)

    >>> circuit.construct((), {})  # construct tape
    >>> m_group = _MGroup(circuit.tape)
    >>> tapes = m_group.get_tapes(group=False)
    >>> [tape.measurements for tape in tapes]
    [[expval(PauliX(wires=[0]))],
    [expval(PauliZ(wires=[0]))],
    [expval(PauliY(wires=[1]))],
    [expval(PauliX(wires=[1]))]]
    >>> tapes = m_group.get_tapes(group=True)
    >>> [tape.measurements for tape in tapes]
    [[expval(PauliX(wires=[0])), expval(PauliY(wires=[1]))],
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1]))]]

    The original tape contains a total of 10 measurements. These have been reduced to 4 measurements
    distributed in two different tapes.
    """

    @dataclass
    class MData:
        """Dataclass containing all the needed information of a measurement."""

        m: MeasurementProcess
        data: dict  # {idx: coeff}
        group: int = None

    def __init__(self, tape: QuantumTape):
        # {hash: {location_idx: (coeff, measurement), ...}, ...}
        self.tape = tape
        self.queue: Dict[int, _MGroup.MData] = {}
        self.mdata_groups = None
        self._num_groups = 1
        self._generate_queue()

    def _generate_queue(self):
        for idx, m in enumerate(self.tape.measurements):
            self._add(measurement=m, idx=idx)

    def _add(self, measurement: MeasurementProcess, idx: int, coeff=1):
        """Add operator to the measurement queue.

        If the operator hash is already in the dictionary, the coefficient is increased instead.

        Args:
            summand (Operator): operator to add to the summands dictionary
            coeff (int, optional): Coefficient of the operator. Defaults to 1.
            op_hash (int, optional): Hash of the operator. Defaults to None.
        """
        obs = measurement.obs
        if isinstance(obs, Sum) and isinstance(measurement, ExpectationMP):
            for summand in obs.operands:
                coeff = 1
                if isinstance(summand, SProd):
                    coeff = summand.scalar
                    summand = summand.base
                self._add(qml.expval(summand), idx, coeff)
        elif isinstance(obs, Hamiltonian) and isinstance(measurement, ExpectationMP):
            if obs.grouping_indices is None:
                # If grouping_indices was not computed previously, add the individual hamiltonian
                # observables to the dictionary to group them later.
                for o, c in zip(obs.ops, obs.data):
                    self._add(qml.expval(o), idx, c)
            else:
                # Add the previously computed groups of qwc measurements
                for indices in obs.grouping_indices:
                    m_group = tuple(qml.expval(obs.ops[i]) for i in indices)
                    c_group = tuple(qml.math.stack([obs.data[i] for i in indices]))
                    for m, c in zip(m_group, c_group):
                        self._add_to_queue(m, idx, coeff * c, group=self._num_groups)
                    self._num_groups += 1

        elif isinstance(obs, SProd) and isinstance(measurement, ExpectationMP):
            self._add(qml.expval(obs.base), idx, coeff * obs.scalar)
        else:
            # Use ``coeff=None`` when the measurement is not an expectation value.
            # In the `_compute_result` method we skip using ``qml.math.dot`` when coeff is None,
            # this way we avoid an error when having non tensor objects.
            if not isinstance(measurement, ExpectationMP):
                coeff = None
            self._add_to_queue(measurement, idx, coeff)

    def _add_to_queue(self, measurement: MeasurementProcess, idx: int, coeff: float, group=None):
        m_hash = measurement.hash
        if m_hash not in self.queue:
            self.queue[m_hash] = self.MData(measurement, {idx: coeff}, group=group)
        else:
            mdata = self.queue[m_hash]
            mdata.data[idx] = mdata.data[idx] + coeff if idx in mdata.data else coeff
            if mdata.group is None:
                # update group information if the measurement didn't belong to any group
                mdata.group = group

    def get_tapes_and_processing_fn(self, group=False) -> List[List["_MGroup.MData"]]:
        """Splits the original tape into a list of tapes. Returns the list of tapes and a function
        to postprocess the results of the tapes' execution.

        If ``group=False``, the original tape is split if and only if it contains the expectation
        value of a ``Hamiltonian`` with its ``grouping_indices`` previously computed. If
        ``group=True``, all the measurements of the original tape are grouped into:

        * Qubit-wise commuting groups, if the observables are pauli words.
        * Groups with non-overlapping wires, if the observables are not pauli words.

        If the tape contains both pauli and non-pauli words, the two grouping techniques are applied.

        Returns:
            List[QuantumTape]: list of tapes
            Callable: postprocessing function
        """

        if group:
            self._group_measurements()
        else:
            grouped_data = defaultdict(lambda: [])
            non_grouped_data = []
            for mdata in self.queue.values():
                if mdata.group is not None:
                    grouped_data[mdata.group].append(mdata)
                else:
                    non_grouped_data.append([mdata])
            self.mdata_groups = list(grouped_data.values()) + non_grouped_data

        measurements = [[mdata.m for mdata in m_group] for m_group in self.mdata_groups]
        if (
            len(measurements) == 1
            and len(measurements[0]) == len(self.tape.measurements)
            and all(qml.equal(m1, m2) for m1, m2 in zip(measurements[0], self.tape.measurements))
        ):
            # no split is applied to the tape measurements
            return [self.tape], lambda res: res[0]

        tapes = [
            QuantumTape(ops=self.tape._ops, measurements=m, prep=self.tape._prep)
            for m in measurements
        ]

        return tapes, self.processing_fn

    def _group_measurements(self):
        # Separate measurements into pauli, non-pauli words and previously computed groups
        # of hamiltonian measurements
        hamiltonian_m: List[_MGroup.MData] = defaultdict(lambda: [])  # {group_idx: [measurements]}
        pauli_m: List[_MGroup.MData] = []
        non_pauli_m: List[_MGroup.MData] = []
        for mdata in self.queue.values():
            is_pauli = is_pauli_word(mdata.m.obs) or (mdata.m.obs is None and mdata.m.wires)
            is_shadow = isinstance(mdata.m, (ShadowExpvalMP, ClassicalShadowMP))
            if mdata.group is not None:
                hamiltonian_m[mdata.group].append(mdata)
            elif not is_shadow and is_pauli:
                # When m.obs is None the measurement acts on the basis states, and thus we can
                # assume that the observable is ``qml.PauliZ`` for all measurement wires.
                # Shadow measurements are not treated as pauli measurements because they cannot
                # be executed with other measurements.
                pauli_m.append(mdata)
            else:
                non_pauli_m.append(mdata)

        all_m = list(hamiltonian_m.values())

        # group pauli measurements
        if len(pauli_m) > 1:
            observables = [(mdata.m.obs or _pauli_z(mdata.m.wires)) for mdata in pauli_m]
            grouped_m = group_observables(observables=observables, coefficients=pauli_m)[1]
            all_m += grouped_m
        elif pauli_m:
            all_m += [pauli_m]

        # group remaining measurements into groups with non overlapping wires
        qwc_groups = [
            (Wires.all_wires([mdata.m.wires for mdata in group]), group) for group in all_m
        ]
        for mdata in non_pauli_m:
            if len(mdata.m.wires) == 0 or isinstance(mdata.m, (ClassicalShadowMP, ShadowExpvalMP)):
                # If the measurement doesn't have wires, we assume it acts on all wires and that
                # it won't commute with any other measurement
                # Shadow measurements cannot be executed with other measurements because the
                # `classical_shadow` implementation of the `DefaultQubit` class resets the state
                # of the device.
                qwc_groups.append(([], [mdata]))
            else:
                op_added = False

                for idx, (wires, group) in enumerate(qwc_groups):
                    # check overlapping wires
                    if len(wires) > 0 and all(wire not in mdata.m.wires for wire in wires):
                        qwc_groups[idx] = (wires + mdata.m.wires, group + [mdata])
                        op_added = True
                        break

                if not op_added:
                    qwc_groups.append((mdata.m.wires, [mdata]))

        self.mdata_groups = [group[1] for group in qwc_groups]

    def processing_fn(self, expanded_results):
        """Processing function used to post-process the results from the execution of the tapes
        generated by the ``get_tapes`` method.

        Args:
            expanded_results (Sequence[float]): execution results

        Returns:
            Sequence[float]: post-processed results
        """
        if qml.active_return():
            return self._processing_fn_new(expanded_results)
        results = {}  # {m_idx, result}
        for tape_res, m_group in zip(expanded_results, self.mdata_groups):
            batching = self.tape.batch_size is not None and self.tape.batch_size > 1
            shot_vector = len(m_group) != len(tape_res) and not batching
            if len(m_group) == 1 and shot_vector:
                # When using a shot vector and a single measurement we don't have the extra
                # dimension that we have with multiple measurements.
                _compute_result_and_add_to_dict(tape_res, m_group[0].data, results)
            elif shot_vector or batching:
                # When batching is used, the first dimension of tape_res corresponds to the
                # batching dimension.
                # When a shot vector is used, the first dimension of tape_res corresponds to the
                # shot vector dimension. The results need to be transposed. The transpose is done
                # right before the return statement.
                for i, mdata in enumerate(m_group):
                    res = [r[i] for r in tape_res] if len(m_group) > 1 else tape_res[0]
                    _compute_result_and_add_to_dict(res, mdata.data, results)
            else:
                for res, mdata in zip(tape_res, m_group):
                    _compute_result_and_add_to_dict(res, mdata.data, results)

        # sort results by idx
        results = tuple(results[key] for key in sorted(results))

        if any(qml.math.requires_grad(res) for res in results):
            # when computing gradients, we need to convert the tuple to a gradient box
            results = qml.math.stack(results)
        elif shot_vector:
            results = qml.math.transpose(results)
        else:
            # Convert results to array for the old return types
            results = qml.math.convert_like(results, results[0])

        return results[0] if len(results) == 1 else results

    def _processing_fn_new(self, expanded_results):
        """Processing function used to post-process the results from the execution of the tapes
        generated by the ``get_tapes`` method.

        Args:
            expanded_results (Sequence[float]): execution results

        Returns:
            Sequence[float]: post-processed results
        """
        results = {}  # {m_idx, result}
        for tape_res, m_group in zip(expanded_results, self.mdata_groups):
            if len(m_group) == 1 and qml.math.ndim(tape_res) == 0:
                # new return types return a scalar when returning one result without broadcasting
                tape_res = [tape_res]
            batching = self.tape.batch_size is not None and self.tape.batch_size > 1
            shot_vector = len(m_group) != len(tape_res) and not batching
            if shot_vector or batching:
                # When batching is used, the first dimension of tape_res corresponds to the
                # batching dimension.
                # When a shot vector is used, the first dimension of tape_res corresponds to the
                # shot vector dimension. The results need to be transposed. The transpose is done
                # right before the return statement.
                if len(m_group) == 1:
                    # When using a shot vector or batching and a single measurement we don't
                    # have the extra dimension that we have with multiple measurements.
                    _compute_result_and_add_to_dict(tape_res, m_group[0].data, results)
                else:
                    for i, mdata in enumerate(m_group):
                        res = [r[i] for r in tape_res] if len(m_group) > 1 else tape_res[0]
                        _compute_result_and_add_to_dict(res, mdata.data, results)
            else:
                for res, mdata in zip(tape_res, m_group):
                    _compute_result_and_add_to_dict(res, mdata.data, results)

        # sort results by idx
        results = tuple(results[key] for key in sorted(results))

        if shot_vector:
            # transpose for new return types
            results = tuple(zip(*results))

        return results[0] if len(results) == 1 else results


def _pauli_z(wires: Wires):
    """Generate ``PauliZ`` operator.

    Args:
        wires (Wires): wires that the operator acts on"""
    if len(wires) == 1:
        return qml.PauliZ(wires[0])
    return Tensor(*[qml.PauliZ(w) for w in wires])


def _compute_result_and_add_to_dict(res, data: dict, results: dict):
    for idx, coeff in data.items():
        r = qml.math.convert_like(qml.math.dot(res, coeff), res) if coeff is not None else res
        if idx in results:
            results[idx] += r
        else:
            results[idx] = r
