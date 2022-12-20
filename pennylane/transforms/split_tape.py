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
Contains the hamiltonian expand tape transform
"""
from dataclasses import dataclass

# pylint: disable=protected-access
from typing import Dict, List, Tuple

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, SProd, Sum
from pennylane.pauli import group_observables, is_pauli_word
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


# pylint: disable=too-many-branches, too-many-statements
def split_tape(tape: QuantumTape, group=True):
    """If the tape is measuring any Hamiltonian or Sum expectation, this method splits it into
    multiple tapes of Pauli expectations and provides a function to recombine the results.

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value
            of the Hamiltonian
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
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
    -0.5

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
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0]))]
    [expval(PauliY(wires=[2]) @ PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[2]))]
    [expval(PauliZ(wires=[1]))]
    [expval(PauliX(wires=[1]))]

    Five tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last two contain the remaining observables. Note that the scalars of the scalar products
    have been removed. These values will be multiplied by the result obtained from the execution.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    [0.0, -0.5, 0.0, -0.9999999999999996]

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
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1]))]
    [expval(PauliX(wires=[0]))]
    """
    m_grouping = _MeasurementsGrouping(measurements=tape.measurements)

    # Create the tapes, group observables if group==True
    mdata_groups = m_grouping.group() if group else m_grouping.measurements
    tapes = [
        QuantumTape(ops=tape._ops, measurements=[mdata.m for mdata in m_group], prep=tape._prep)
        for m_group in mdata_groups
    ]

    def processing_fn(expanded_results):
        results = []  # [(m_idx, result)]
        for tape_res, m_group in zip(expanded_results, mdata_groups):
            # tape_res contains multiple results
            if tape.batch_size is not None and tape.batch_size > 1:
                # when batching is used, the first dimension of tape_res corresponds to the
                # batching dimension
                for i, mdata in enumerate(m_group):
                    results.extend(_compute_result([r[i] for r in tape_res], mdata))
            else:
                for res, mdata in zip(tape_res, m_group):
                    results.extend(_compute_result(res, mdata.data))

        # sum results by idx
        res_dict = {}
        for idx, res in results:
            if idx in res_dict:
                res_dict[idx] += res
            else:
                res_dict[idx] = res

        # sort results by idx
        results = tuple(res_dict[key] for key in sorted(res_dict))

        return results[0] if len(results) == 1 else results

    return tapes, processing_fn


def _compute_result(result, data: dict):
    results = []
    for idx, coeff in data.items():
        if coeff is not None:
            tmp_res = qml.math.convert_like(qml.math.dot(coeff, result), result)
            results.append((idx, tmp_res))
        else:
            results.append((idx, result))
    return results


def _pauli_z(wires: Wires):
    """Generate ``PauliZ`` operator.

    Args:
        wires (Wires): wires that the operator acts on"""
    if len(wires) == 1:
        return qml.PauliZ(wires[0])
    return Tensor(*[qml.PauliZ(w) for w in wires])


class _MeasurementsGrouping:
    """Utils class used to group measurements.

    If the observables are pauli words, the groups contain observables that commute with each other.
    All the other observables are grouped into groups with non overlapping wires.
    """

    @dataclass
    class MData:
        """Dataclass containing all the needed information of a measurement."""

        m: MeasurementProcess
        data: dict  # {idx: coeff}

    def __init__(self, measurements: List[MeasurementProcess]):
        # {hash: {location_idx: (coeff, measurement), ...}, ...}
        self._measurements = measurements
        self.queue: Dict[int, _MeasurementsGrouping.MData] = {}
        """Dictionary containing the information of all the measurements. Its keys are the hashes
        of the measurements, while its values are dictionaries containing the index of the location
        of the measurement inside the tape and a tuple with the coefficient and the measurement.

        **Example:**
        Let's define the following QNode:

        .. code-block:: python

            @qml.qnode(dev):
            def circuit():
                return qml.expval(qml.PauliX(0) + qml.PauliZ(0) + qml.PauliX(1)), qml.expval(qml.PauliX(0))

        Now we can use the :class:`_MeasurementsGrouping` class to group the measurements together:
        >>> circuit.construct()  # construct tape
        >>> m_group = _MeasurementsGrouping(circuit.tape.measurements)
        >>> m_group.group()
        >>> m_group.queue

        """
        # [[(coeff1, measurement1), (coeff2, measurement2), ...], ...]
        self.groups: List[List[Tuple[float, MeasurementProcess]]] = []
        self._generate_queue()

    def _generate_queue(self):
        for idx, m in enumerate(self._measurements):
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
                # Add the previously computed groups of "qwc" measurements
                for indices in obs.grouping_indices:
                    m_group = tuple(qml.expval(obs.ops[i]) for i in indices)
                    c_group = tuple(qml.math.stack([obs.data[i] for i in indices]))
                    self._add_group_to_queue(m_group, idx, qml.math.dot(coeff, c_group))

        elif isinstance(obs, SProd) and isinstance(measurement, ExpectationMP):
            self._add(qml.expval(obs.base), idx, coeff * obs.scalar)
        else:
            self._add_to_queue(measurement, idx, coeff)

    def _add_to_queue(self, measurement: MeasurementProcess, idx: int, coeff: float):
        m_hash = measurement.hash
        if m_hash not in self.queue:
            self.queue[m_hash] = self.MData(measurement, {idx: coeff})
        else:
            mdata = self.queue[m_hash]
            # Use qml.math.add because coeff might be a tuple of values (for Hamiltonian)
            mdata.data[idx] = qml.math.add(mdata.data[idx], coeff) if idx in mdata.data else coeff

    def _add_group_to_queue(
        self, m_group: Tuple[MeasurementProcess], idx: int, c_group: List[float]
    ):
        hash_g = hash(m_group)
        if hash_g not in self.queue:
            self.queue[hash_g] = [self.MData(m, {idx: c}) for m, c in zip(m_group, c_group)]
        else:
            mdata_list = self.queue[hash_g]
            for mdata, c in zip(mdata_list, c_group):
                mdata.data[idx] = qml.math.add(mdata.data[idx], c) if idx in mdata.data else c

    @property
    def measurements(self):
        """Get summands list.

        All summands with a coefficient less than cutoff are ignored.

        Args:
            cutoff (float, optional): Cutoff value. Defaults to 1.0e-12.
        """
        return [
            mdata if isinstance(mdata, (list, tuple)) else [mdata] for mdata in self.queue.values()
        ]

    def group(self) -> List[List["_MeasurementsGrouping.MData"]]:
        """Returns a list of lists containing groups of qubit-wise commuting pauli-based
        measurements and groups of non-overlapping wires non-pauli-based measurements."""
        # Separate measurements into pauli and non-pauli words
        hamiltonian_m: List[_MeasurementsGrouping.MData] = []
        pauli_m: List[_MeasurementsGrouping.MData] = []
        non_pauli_m: List[_MeasurementsGrouping.MData] = []
        for mdata in self.queue.values():
            m = mdata.m
            if isinstance(m, list):
                hamiltonian_m.append(mdata)
            elif is_pauli_word(m.obs) or (m.obs is None and m.wires):
                # When m.obs is None the measurement acts on the basis states, and thus we can
                # assume that the observable is ``qml.PauliZ`` for all measurement wires.
                pauli_m.append(mdata)
            else:
                non_pauli_m.append(mdata)

        all_m = hamiltonian_m
        # group pauli measurements
        if len(pauli_m) > 1:
            all_m = hamiltonian_m + pauli_m
            observables = [(mdata.m.obs or _pauli_z(mdata.m.wires)) for mdata in all_m]
            all_m = group_observables(observables=observables, coefficients=all_m)[1]
        elif pauli_m:
            all_m = hamiltonian_m + [pauli_m]

        # group remaining measurements into groups with non overlapping wires
        qwc_groups = [
            (Wires.all_wires([mdata.m.wires for mdata in group]), group) for group in all_m
        ]
        for mdata in non_pauli_m:
            if len(mdata.m.wires) == 0:
                # If the measurement doesn't have wires, we assume it acts on all wires and that
                # it won't commute with any other measurement
                qwc_groups.append((mdata.m.wires, [mdata]))
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

        return [group[1] for group in qwc_groups]
