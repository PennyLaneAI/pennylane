# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Contains the tape transform that splits a tape into tapes measuring commuting observables.
"""

# pylint: disable=too-many-boolean-expressions

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial, wraps
from typing import Literal

import numpy as np
from scipy.stats import multinomial

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMP
from pennylane.ops import Prod, SProd, Sum
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn, Result, ResultBatch, TensorLike


@dataclass
class SingleTermMP:
    """A dataclass to represent a single-term observable in a list of measurement processes.

    Args:
        indices (list[int]): indices of the single-term observable in the list of measurement processes
        coeffs (list[complex]): coefficients of the single-term observable in the list of measurement processes
        group_idx (int): index of the commuting group the single-term observable belongs to
        idx_in_group (int): index of the single-term observable in his own commuting group
    """

    indices: list[int]
    coeffs: list[complex]
    group_idx: int
    idx_in_group: int


ShotDistFunction = Callable[
    # total_shots, coeffs_per_group, seed -> shots_per_tape
    [int, list[list[TensorLike]], np.random.Generator | int | None],
    Sequence[int],
]


def _uniform_deterministic_sampling(total_shots, coeffs_per_group, _):
    """Uniform deterministic splitting of the total number of shots for each commuting group."""
    num_groups = len(coeffs_per_group)
    shots, remainder = divmod(total_shots, num_groups)
    shots_per_group = np.full(num_groups, shots)
    shots_per_group[:remainder] += 1
    return shots_per_group.astype(int)


def _weighted_deterministic_sampling(total_shots, coeffs_per_group, _):
    """Weighted deterministic splitting of the total number of shots for each commuting group.
    For each group, the weight is proportional to the L1 norm of the group's coefficients.
    """
    norm_per_group = [np.linalg.norm(coeffs, ord=1) for coeffs in coeffs_per_group]
    prob_shots = np.array(norm_per_group) / np.sum(norm_per_group)
    shots_per_group = np.floor(total_shots * prob_shots)
    remainder = int(total_shots - np.sum(shots_per_group))
    shots_per_group[:remainder] += 1
    return shots_per_group.astype(int)


def _weighted_random_sampling(total_shots, coeffs_per_group, seed):
    """Weighted random sampling of the number of shots for each commuting group.
    For each group, the weight is proportional to the L1 norm of the group's coefficients.
    """
    norm_per_group = [np.linalg.norm(coeffs, ord=1) for coeffs in coeffs_per_group]
    prob_shots = np.array(norm_per_group) / np.sum(norm_per_group)
    distribution = multinomial(n=total_shots, p=prob_shots, seed=seed)
    shots_per_group = distribution.rvs()[0]
    return shots_per_group.astype(int)


shot_dist_str2fn = {
    "uniform": _uniform_deterministic_sampling,
    "weighted": _weighted_deterministic_sampling,
    "weighted_random": _weighted_random_sampling,
}


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def shot_vector_support(initial_postprocessing: PostprocessingFn) -> PostprocessingFn:
    """Convert a postprocessing function to one with shot vector support."""

    @wraps(initial_postprocessing)
    def shot_vector_postprocessing(results):
        return tuple(initial_postprocessing(r) for r in zip(*results, strict=True))

    return shot_vector_postprocessing


@transform
def split_non_commuting(
    tape: QuantumScript,
    grouping_strategy: Literal["default", "wires", "qwc"] | None = "default",
    shot_dist: ShotDistFunction | Literal["uniform", "weighted", "weighted_random"] | None = None,
    seed: np.random.Generator | int | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Splits a circuit into tapes measuring groups of commuting observables.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to be split.
        grouping_strategy (str): The strategy to use for computing disjoint groups of
            commuting observables, can be ``"default"``, ``"wires"``, ``"qwc"``,
            or ``None`` to disable grouping.
        shot_dist (str or Callable or None): The strategy to use for shot distribution
            over the disjoint groups of commuting observables. Values can be ``"uniform"``
            (evenly distributes the number of ``shots`` across all groups of commuting terms),
            ``"weighted"`` (distributes the number of ``shots`` according to weights proportional
            to the L1 norm of the coefficients in each group), ``"weighted_random"`` (same
            as ``"weighted"``, but the numbers of ``shots`` are sampled from a multinomial distribution)
            or a custom callable. ``None`` will disable any shot distribution strategy.
            See Usage Details for more information.
        seed (Generator or int or None): A seed-like parameter used only when the shot distribution
            strategy involves a non-deterministic sampling process (e.g. ``"weighted_random"``).

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:
        the transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        TypeError: if ``shot_dist`` is not a str or Callable or None.
        ValueError: if ``shot_dist`` is a str but not an available strategy.

    .. note::
        This transform splits expectation values of sums into separate terms, and also distributes the terms into
        multiple executions if there are terms that do not commute with one another. For state-based simulators
        that are able to handle non-commuting measurements in a single execution, but don't natively support sums
        of observables, consider :func:`split_to_single_terms <pennylane.transforms.split_to_single_terms>` instead.

    **Examples:**

    This transform allows us to transform a QNode measuring multiple observables into multiple
    circuit executions, each measuring a group of commuting observables.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    Instead of decorating the QNode, we can also create a new function that yields the same
    result in the following way:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

        circuit = qml.transforms.split_non_commuting(circuit)

    Internally, the QNode is split into multiple circuits when executed:

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤  <X> ╭<X@Z>
    1: ──RX(0.79)─┤      ╰<X@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<Z@Z>  <Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>

    Note that the observable ``Y(1)`` occurs twice in the original QNode, but only once in the
    transformed circuits. When there are multiple expectation value measurements that rely on
    the same observable, this observable is measured only once, and the result is copied to each
    original measurement.

    While internally multiple tapes are created, the end result has the same ordering as the user
    provides in the return statement. Executing the above QNode returns the original ordering of
    the expectation values.

    >>> circuit([np.pi/4, np.pi/4])
    [0.7071067811865475,
     -0.7071067811865475,
     0.49999999999999994,
     0.8535533905932737]

    There are two algorithms used to compute disjoint groups of commuting observables: ``"qwc"``
    grouping uses :func:`~pennylane.pauli.group_observables` which computes groups of qubit-wise
    commuting observables, producing the fewest number of circuit executions, but can be expensive
    to compute for large multi-term Hamiltonians, while ``"wires"`` grouping simply ensures
    that no circuit contains two measurements with overlapping wires, disregarding commutativity
    between the observables being measured.

    The ``grouping_strategy`` keyword argument can be used to specify the grouping strategy. By
    default, qwc grouping is used whenever possible, except when the circuit contains multiple
    measurements that includes an expectation value of a ``qml.Hamiltonian``, in which case wires
    grouping is used in case the Hamiltonian is very large, to save on classical runtime. To force
    qwc grouping in all cases, set ``grouping_strategy="qwc"``. Similarly, to force wires grouping,
    set ``grouping_strategy="wires"``:

    .. code-block:: python3

        @functools.partial(qml.transforms.split_non_commuting, grouping_strategy="wires")
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    In this case, four circuits are created as follows:

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤  <X>
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<Z@Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<X@Z>
    1: ──RX(0.79)─┤ ╰<X@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤  <Z>
    1: ──RX(0.79)─┤

    Alternatively, to disable grouping completely, set ``grouping_strategy=None``:

    .. code-block:: python3

        @functools.partial(qml.transforms.split_non_commuting, grouping_strategy=None)
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    In this case, each observable is measured in a separate circuit execution.

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤  <X>
    1: ──RX(0.79)─┤
    <BLANKLINE>
    0: ──RY(0.79)─┤
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<Z@Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<X@Z>
    1: ──RX(0.79)─┤ ╰<X@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤  <Z>
    1: ──RX(0.79)─┤

    Note that there is an exception to the above rules: if the circuit only contains a single
    expectation value measurement of a ``Hamiltonian`` or ``Sum`` with pre-computed grouping
    indices, the grouping information will be used regardless of the requested ``grouping_strategy``

    .. details::
        :title: Usage Details

        **Shot distribution**

        With finite-shot measurements, the default behaviour of ``split_non_commuting`` is to perform one
        execution with the total number of ``shots`` for each group of commuting terms. With the
        ``shot_dist`` argument, this behaviour can be changed. For example,
        ``shot_dist = "weighted"`` will partition the number of shots performed for
        each commuting group according to the L1 norm of each group's coefficients:

        .. code-block:: python3

            import pennylane as qml
            from pennylane.transforms import split_non_commuting
            from functools import partial

            ham = qml.Hamiltonian(
                coeffs=[10, 0.1, 20, 100, 0.2],
                observables=[
                    qml.X(0) @ qml.Y(1),
                    qml.Z(0) @ qml.Z(2),
                    qml.Y(1),
                    qml.X(1) @ qml.X(2),
                    qml.Z(0) @ qml.Z(1) @ qml.Z(2)
                ]
            )

            dev = qml.device("default.qubit")

            @partial(split_non_commuting, shot_dist="weighted")
            @qml.qnode(dev, shots=10000)
            def circuit():
                return qml.expval(ham)

            with qml.Tracker(dev) as tracker:
                circuit()

        >>> print(tracker.history["shots"])
        [2303, 23, 7674]

        The ``shot_dist`` strategy can be also defined by a custom function. For example:

        .. code-block:: python3

            import numpy as np

            def my_shot_dist(total_shots, coeffs_per_group, seed):
                max_per_group = [np.max(np.abs(coeffs)) for coeffs in coeffs_per_group]
                prob_shots = np.array(max_per_group) / np.sum(max_per_group)
                return np.round(total_shots * prob_shots)

            @partial(split_non_commuting, shot_dist=my_shot_dist)
            @qml.qnode(dev, shots=10000)
            def circuit():
                return qml.expval(ham)

            with qml.Tracker(dev) as tracker:
                circuit()

        >>> print(tracker.history["shots"])
        [1664, 17, 8319]

        **Internal details**

        Internally, this function works with tapes. We can create a tape with multiple
        measurements of non-commuting observables:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.X(1)),
                qml.expval(qml.Z(0)),
                qml.expval(qml.X(0))
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        Now ``tapes`` is a list of two tapes, each contains a group of commuting observables:

        >>> [t.measurements for t in tapes]
        [[expval(Z(0) @ Z(1)), expval(Z(0))], [expval(X(0) @ X(1)), expval(X(0))]]

        The processing function becomes important as the order of the inputs has been modified.

        >>> dev = qml.device("default.qubit", wires=2)
        >>> result_batch = [dev.execute(t) for t in tapes]
        >>> result_batch
        [(1.0, 1.0), (0.0, 0.0)]

        The processing function can be used to reorganize the results:

        >>> processing_fn(result_batch)
        (1.0, 0.0, 1.0, 0.0)

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
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        This results in two tapes, each with commuting measurements:

        >>> [t.measurements for t in tapes]
        [[expval(X(0)), probs(wires=[1])], [probs(wires=[0, 1])]]

    """
    if len(tape.measurements) == 0:
        return [tape], null_postprocessing

    if shot_dist is None:
        shot_dist_fn = None
    elif isinstance(shot_dist, Callable):
        shot_dist_fn = shot_dist
    elif isinstance(shot_dist, str):
        try:
            shot_dist_fn = shot_dist_str2fn[shot_dist]
        except KeyError as e:
            raise ValueError(
                f"Unknown shot_dist='{shot_dist}'. Available options are {list(shot_dist_str2fn.keys())}."
            ) from e
    else:
        raise TypeError(f"shot_dist must be a callable or str or None, not {type(shot_dist)}.")

    # Special case for a single measurement of a Sum, in which case
    # the grouping information can be computed and cached in the observable.
    if (
        len(tape.measurements) == 1
        and isinstance(tape.measurements[0], ExpectationMP)
        and isinstance(tape.measurements[0].obs, Sum)
        and (
            (
                grouping_strategy in ("default", "qwc")
                and all(qml.pauli.is_pauli_word(o) for o in tape.measurements[0].obs.terms()[1])
            )
            or tape.measurements[0].obs.grouping_indices is not None
        )
    ):
        return _split_ham_with_grouping(tape, shot_dist_fn=shot_dist_fn, seed=seed)

    if shot_dist is not None:
        warnings.warn(
            f"shot_dist='{shot_dist}' is not supported for multiple measurements.", UserWarning
        )

    single_term_obs_mps, offsets = _split_all_multi_term_obs_mps(tape)

    if grouping_strategy is None:
        measurements = list(single_term_obs_mps.keys())
        tapes = [tape.copy(measurements=[m]) for m in measurements]
        fn = partial(
            _processing_fn_no_grouping,
            single_term_obs_mps=single_term_obs_mps,
            offsets=offsets,
            batch_size=tape.batch_size,
        )
        if tape.shots.has_partitioned_shots:
            fn = shot_vector_support(fn)
        return tapes, fn

    if grouping_strategy == "wires" or any(
        m.obs is not None and not qml.pauli.is_pauli_word(m.obs) for m in single_term_obs_mps
    ):
        # TODO: here we fall back to wire-based grouping if any of the observables in the tape
        #       is not a pauli word. As a result, adding a single measurement to a circuit could
        #       significantly increase the number of circuit executions. We should be able to
        #       separate the logic for pauli-word observables and non-pauli-word observables,
        #       putting non-pauli-word observables in separate wire-based groups, but using qwc
        #       based grouping for the rest of the observables. [sc-79686]
        return _split_using_wires_grouping(tape, single_term_obs_mps, offsets)

    return _split_using_qwc_grouping(tape, single_term_obs_mps, offsets)


def _split_ham_with_grouping(
    tape: qml.tape.QuantumScript, shot_dist_fn: ShotDistFunction, seed: int
):
    """Split a tape measuring a single Sum and group commuting observables.
    It also assigns to each new tape the correct number of shots according to the
    shot distribution function defining the strategy for shot allocation.
    """
    # pylint:disable=too-many-branches

    obs = tape.measurements[0].obs
    if obs.grouping_indices is None:
        obs.compute_grouping()

    coeff_list, obs_list = obs.terms()

    single_term_obs_mps = {}  # dictionary mapping ExpectationMP to SingleTermMP
    mps_groups = []  # list of groups of commuting measurement processes
    coeffs_per_group = []  # list of coefficients for each group
    offset = 0  # constant offset of the Sum arising from Identity terms

    # obs.grouping_indices is a list of lists, where each list contains the indices of obs that belong in each group
    for group_idx, obs_indices in enumerate(obs.grouping_indices):
        mps_group = []
        coeffs_group = []
        idx_in_group = 0

        for idx in obs_indices:
            obs = obs_list[idx]
            coeff = coeff_list[idx]

            if isinstance(obs, qml.Identity):
                offset += coeff
            else:
                mp = qml.expval(obs)
                if mp in single_term_obs_mps:
                    single_term_obs_mps[mp].coeffs[0] += coeff
                else:
                    single_term_obs_mps[mp] = SingleTermMP(
                        indices=[0],
                        coeffs=[coeff],
                        group_idx=group_idx,
                        idx_in_group=idx_in_group,
                    )
                    mps_group.append(mp)
                    coeffs_group.append(coeff)
                    idx_in_group += 1

        if mps_group:
            mps_groups.append(mps_group)
            coeffs_per_group.append(coeffs_group)

    if tape.shots.total_shots is not None and shot_dist_fn is not None:
        shots_per_group = shot_dist_fn(tape.shots.total_shots, coeffs_per_group, seed)
        tapes = []
        for mps, shots in zip(mps_groups, shots_per_group, strict=True):
            if int(shots) != 0:
                tapes.append(tape.copy(measurements=mps, shots=int(shots)))
            else:
                for mp in mps:
                    del single_term_obs_mps[mp]
    else:
        tapes = [tape.copy(measurements=mps) for mps in mps_groups]

    group_sizes = [len(mps) for mps in mps_groups]

    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps,
        offsets=[offset],
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )

    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)

    return tapes, fn


def _split_using_qwc_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]],
    offsets: list[TensorLike],
):
    """Split tapes using group_observables in the Pauli module.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (dict[MeasurementProcess, tuple[list[int], list[TensorLike]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (list[TensorLike]): Offsets associated with each original measurement in the tape.

    """

    # The legacy device does not support state measurements combined with any other
    # measurement, so each state measurement must be in its own tape.
    state_measurements = [m for m in single_term_obs_mps if isinstance(m, StateMP)]

    measurements = [m for m in single_term_obs_mps if not isinstance(m, StateMP)]
    obs_list = [_mp_to_obs(m, tape) for m in measurements]
    index_groups = []
    if len(obs_list) > 0:
        index_groups = qml.pauli.compute_partition_indices(obs_list)

    # A dictionary for measurements of each unique single-term observable, mapped to the
    # indices of the original measurements it belongs to, its coefficients, the index of
    # the group it belongs to, and the index of the measurement in the group.
    single_term_obs_mps_grouped = {}

    mp_groups = [[] for _ in index_groups]
    group_sizes = []
    for group_idx, obs_indices in enumerate(index_groups):
        group_size = 0
        for obs_idx in obs_indices:
            mp = measurements[obs_idx]
            mp_groups[group_idx].append(mp)
            single_term_obs_mps_grouped[mp] = SingleTermMP(
                indices=single_term_obs_mps[mp][0],
                coeffs=single_term_obs_mps[mp][1],
                group_idx=group_idx,
                idx_in_group=group_size,
            )
            group_size += 1
        group_sizes.append(group_size)

    for state_mp in state_measurements:
        mp_groups.append([state_mp])
        single_term_obs_mps_grouped[state_mp] = SingleTermMP(
            indices=single_term_obs_mps[state_mp][0],
            coeffs=single_term_obs_mps[state_mp][1],
            group_idx=len(mp_groups) - 1,
            idx_in_group=0,
        )
        group_sizes.append(1)
    tapes = [tape.copy(measurements=mps) for mps in mp_groups]
    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )
    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)
    return tapes, fn


def _split_using_wires_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]],
    offsets: list[float | TensorLike],
):
    """Split tapes by grouping observables based on overlapping wires.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (list[float | TensorLike]): Offsets associated with each original measurement in the tape.

    """

    mp_groups = []
    wires_for_each_group = []
    group_sizes = []

    # A dictionary for measurements of each unique single-term observable, mapped to the
    # indices of the original measurements it belongs to, its coefficient, the index of
    # the group it belongs to, and the index of the measurement in the group.
    single_term_obs_mps_grouped = {}
    num_groups = 0

    for smp, (mp_indices, coeffs) in single_term_obs_mps.items():

        if len(smp.wires) == 0:  # measurement acting on all wires
            mp_groups.append([smp])
            wires_for_each_group.append(tape.wires)
            group_sizes.append(1)
            single_term_obs_mps_grouped[smp] = SingleTermMP(
                indices=mp_indices,
                coeffs=coeffs,
                group_idx=num_groups,
                idx_in_group=0,
            )
            num_groups += 1
            continue

        group_idx = 0
        added_to_existing_group = False
        while not added_to_existing_group and group_idx < num_groups:
            wires = wires_for_each_group[group_idx]
            if len(wires) != 0 and len(qml.wires.Wires.shared_wires([wires, smp.wires])) == 0:
                mp_groups[group_idx].append(smp)
                wires_for_each_group[group_idx] += smp.wires
                single_term_obs_mps_grouped[smp] = SingleTermMP(
                    indices=mp_indices,
                    coeffs=coeffs,
                    group_idx=group_idx,
                    idx_in_group=group_sizes[group_idx],
                )
                group_sizes[group_idx] += 1
                added_to_existing_group = True
            group_idx += 1

        if not added_to_existing_group:
            mp_groups.append([smp])
            wires_for_each_group.append(smp.wires)
            group_sizes.append(1)
            single_term_obs_mps_grouped[smp] = SingleTermMP(
                indices=mp_indices,
                coeffs=coeffs,
                group_idx=num_groups,
                idx_in_group=0,
            )
            num_groups += 1

    tapes = [tape.copy(measurements=mps) for mps in mp_groups]
    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )
    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)
    return tapes, fn


def _split_all_multi_term_obs_mps(tape: qml.tape.QuantumScript):
    """Splits all multi-term observables in a tape to measurements of single-term observables.

    Args:
        tape (~qml.tape.QuantumScript): The tape with measurements to split.

    Returns:
        single_term_obs_mps (dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]]): A dictionary
            for measurements of each unique single-term observable, mapped to the
            indices of the original measurements it belongs to, and its coefficients.
        offsets (list[float | TensorLike]): Offsets associated with each original measurement in the tape.

    """

    # The dictionary for measurements of each unique single-term observable, mapped the indices
    # of the original measurements it belongs to, and its coefficients.
    single_term_obs_mps = {}

    # Offsets associated with each original measurement in the tape (from Identity)
    offsets = []

    for mp_idx, mp in enumerate(tape.measurements):
        obs = mp.obs
        offset = 0
        if isinstance(mp, ExpectationMP) and isinstance(obs, (Sum, Prod, SProd)):
            # Break the observable into terms, and construct an ExpectationMP with each term.
            for c, o in zip(*obs.terms(), strict=True):
                # If the observable is an identity, track it with a constant offset
                if isinstance(o, qml.Identity):
                    offset += c
                # If the single-term measurement already exists, it can be reused by all original
                # measurements. In this case, add the existing single-term measurement to the list
                # corresponding to this original measurement.
                elif (sm := qml.expval(o)) in single_term_obs_mps:
                    single_term_obs_mps[sm][0].append(mp_idx)
                    single_term_obs_mps[sm][1].append(c)
                # Otherwise, add this new measurement to the list of single-term measurements.
                else:
                    single_term_obs_mps[sm] = ([mp_idx], [c])
        elif isinstance(obs, qml.Identity):
            offset += 1
        else:
            if isinstance(obs, (SProd, Prod)):
                obs = obs.simplify()
            if isinstance(obs, Sum):
                raise RuntimeError(
                    f"Cannot split up terms in sums for MeasurementProcess {type(mp)}"
                )
            # For all other measurement types, simply add them to the list of measurements.
            if mp not in single_term_obs_mps:
                single_term_obs_mps[mp] = ([mp_idx], [1])
            else:
                single_term_obs_mps[mp][0].append(mp_idx)
                single_term_obs_mps[mp][1].append(1)

        offsets.append(offset)

    return single_term_obs_mps, offsets


def _processing_fn_no_grouping(
    res: ResultBatch,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]],
    offsets: list[float | TensorLike],
    batch_size: int | None,
):
    """Postprocessing function for the split_non_commuting transform without grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps] [,batch_size])
        single_term_obs_mps (dict[MeasurementProcess, tuple[list[int], list[float | TensorLike]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (list[float | TensorLike]): Offsets associated with each original measurement in the tape.
        shots (Shots): The shots settings of the original tape.

    """

    res_batch_for_each_mp = [[] for _ in offsets]
    coeffs_for_each_mp = [[] for _ in offsets]

    for smp_idx, (_, (mp_indices, coeffs)) in enumerate(single_term_obs_mps.items()):
        for mp_idx, coeff in zip(mp_indices, coeffs, strict=True):
            res_batch_for_each_mp[mp_idx].append(res[smp_idx])
            coeffs_for_each_mp[mp_idx].append(coeff)

    result_shape = (batch_size,) if batch_size and batch_size > 1 else ()
    # Sum up the results for each original measurement

    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset, result_shape)
        for _sub_res, coeffs, offset in zip(
            res_batch_for_each_mp, coeffs_for_each_mp, offsets, strict=True
        )
    ]
    # res_for_each_mp should have shape (n_mps, [,n_shots] [,batch_size])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    return tuple(res_for_each_mp)


def _processing_fn_with_grouping(
    res: ResultBatch,
    single_term_obs_mps: dict[MeasurementProcess, SingleTermMP],
    offsets: list[TensorLike],
    group_sizes: list[int],
    batch_size: int,
):
    """Postprocessing function for the split_non_commuting transform with grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps_in_group] [,batch_size])
        single_term_obs_mps (dict[MeasurementProcess, SingleTermMP]):
            A dictionary of measurements of each unique single-term observable,
            mapped to the corresponding single-term observable objects.
        offsets (list[float | TensorLike]): Offsets associated with each original measurement in the tape.
        group_sizes (list[int]): The number of tapes in each group.
        shots (Shots): The shots setting of the original tape.

    Returns:
        The results combined into a single result for each original measurement.

    """

    res_batch_for_each_mp = [[] for _ in offsets]  # ([n_mps] [,n_shots] [,batch_size])
    coeffs_for_each_mp = [[] for _ in offsets]

    for term in single_term_obs_mps.values():

        res_group = res[term.group_idx]  # ([n_shots] [,n_mps] [,batch_size])
        group_size = group_sizes[term.group_idx]

        # If there is only one term in the group, the n_mps dimension would have
        # been squeezed out, use the entire result directly.
        sub_res = res_group if group_size == 1 else res_group[term.idx_in_group]

        # Add this result to the result batch for the corresponding original measurement
        for mp_idx, coeff in zip(term.indices, term.coeffs, strict=True):
            res_batch_for_each_mp[mp_idx].append(sub_res)
            coeffs_for_each_mp[mp_idx].append(coeff)

    result_shape = (batch_size,) if batch_size and batch_size > 1 else ()

    # Sum up the results for each original measurement
    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset, result_shape)
        for _sub_res, coeffs, offset in zip(
            res_batch_for_each_mp, coeffs_for_each_mp, offsets, strict=True
        )
    ]

    # res_for_each_mp should have shape (n_mps, [,n_shots] [,batch_size])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    return tuple(res_for_each_mp)


def _sum_terms(
    res: ResultBatch,
    coeffs: list[float | TensorLike],
    offset: float | TensorLike,
    shape: tuple,
) -> Result:
    """Sum results from measurements of multiple terms in a multi-term observable."""
    if (
        coeffs
        and not qml.math.is_abstract(coeffs[0])
        and not qml.math.is_abstract(offset)
        and coeffs == [1]
        and offset == 0
    ):
        return res[0]

    # The shape of res at this point is (n_terms, [,n_shots] [,batch_size])
    dot_products = []
    for c, r in zip(coeffs, res, strict=True):
        if qml.math.get_interface(r) == "autograd":
            r = qml.math.array(r)
        if isinstance(r, (list, tuple)):
            r = qml.math.stack(r)
        dot_products.append(qml.math.dot(c, qml.math.squeeze(r)))
    if len(dot_products) == 0:
        return qml.math.ones(shape) * offset
    summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
    if qml.math.get_interface(offset) == "autograd" and qml.math.requires_grad(summed_dot_products):
        offset = qml.math.array(offset)
    return summed_dot_products + offset


def _mp_to_obs(mp: MeasurementProcess, tape: qml.tape.QuantumScript) -> qml.operation.Operator:
    """Extract the observable from a measurement process.

    If the measurement process has an observable, return it. Otherwise, return a dummy
    observable that is a tensor product of Z gates on every wire.

    """

    if mp.obs is not None:
        return mp.obs

    obs_wires = mp.wires if mp.wires else tape.wires
    return qml.prod(*(qml.Z(wire) for wire in obs_wires))
