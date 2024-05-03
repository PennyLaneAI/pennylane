# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Class CutStrategy, for executing (large) circuits on available (comparably smaller) devices.
"""

import warnings
from collections.abc import Sequence as SequenceType
from dataclasses import InitVar, dataclass
from typing import Any, ClassVar, Dict, List, Sequence, Union

from networkx import MultiDiGraph

import pennylane as qml
from pennylane.ops.meta import WireCut


@dataclass()
class CutStrategy:
    """
    A circuit-cutting distribution policy for executing (large) circuits on available (comparably
    smaller) devices.

    .. note::

        This class is part of a work-in-progress feature to support automatic cut placement in the
        circuit cutting workflow. Currently only manual placement of cuts is supported,
        check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        devices (Union[qml.Device, Sequence[qml.Device]]): Single, or Sequence of, device(s).
            Optional only when ``max_free_wires`` is provided.
        max_free_wires (int): Number of wires for the largest available device. Optional only when
            ``devices`` is provided where it defaults to the maximum number of wires among
            ``devices``.
        min_free_wires (int): Number of wires for the smallest available device, or, equivalently,
            the smallest max fragment-wire-size that the partitioning is allowed to explore.
            When provided, this parameter will be used to derive an upper-bound to the range of
            explored number of fragments.  Optional, defaults to 2 which corresponds to attempting
            the most granular partitioning of max 2-wire fragments.
        num_fragments_probed (Union[int, Sequence[int]]): Single, or 2-Sequence of, number(s)
            specifying the potential (range of) number of fragments for the partitioner to attempt.
            Optional, defaults to probing all valid strategies derivable from the circuit and
            devices. When provided, has precedence over all other arguments affecting partitioning
            exploration, such as ``max_free_wires``, ``min_free_wires``, or ``exhaustive``.
        max_free_gates (int): Maximum allowed circuit depth for the deepest available device.
            Optional, defaults to unlimited depth.
        min_free_gates (int): Maximum allowed circuit depth for the shallowest available device.
            Optional, defaults to ``max_free_gates``.
        imbalance_tolerance (float): The global maximum allowed imbalance for all partition trials.
            Optional, defaults to unlimited imbalance. Used only if there's a known hard balancing
            constraint on the partitioning problem.
        trials_per_probe (int): Number of repeated partitioning trials for a random automatic
            cutting method to attempt per set of partitioning parameters. For a deterministic
            cutting method, this can be set to 1. Defaults to 4.

    **Example**

    The following cut strategy specifies that a circuit should be cut into between
    ``2`` to ``5`` fragments, with each fragment having at most ``6`` wires and
    at least ``4`` wires:

    >>> cut_strategy = qml.qcut.CutStrategy(
    ...     max_free_wires=6,
    ...     min_free_wires=4,
    ...     num_fragments_probed=(2, 5),
    ... )

    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes

    #: Initialization argument only, used to derive ``max_free_wires`` and ``min_free_wires``.
    devices: InitVar[Union[qml.Device, Sequence[qml.Device]]] = None

    #: Number of wires for the largest available device.
    max_free_wires: int = None
    #: Number of wires for the smallest available device.
    min_free_wires: int = None
    #: The potential (range of) number of fragments for the partitioner to attempt.
    num_fragments_probed: Union[int, Sequence[int]] = None
    #: Maximum allowed circuit depth for the deepest available device.
    max_free_gates: int = None
    #: Maximum allowed circuit depth for the shallowest available device.
    min_free_gates: int = None
    #: The global maximum allowed imbalance for all partition trials.
    imbalance_tolerance: float = None
    #: Number of trials to repeat for per set of partition parameters probed.
    trials_per_probe: int = 4

    #: Class attribute, threshold for warning about too many fragments.
    HIGH_NUM_FRAGMENTS: ClassVar[int] = 20
    #: Class attribute, threshold for warning about too many partition attempts.
    HIGH_PARTITION_ATTEMPTS: ClassVar[int] = 20

    def __post_init__(
        self,
        devices,
    ):
        """Deriving cutting constraints from given devices and parameters."""

        self.max_free_wires = self.max_free_wires
        if isinstance(self.num_fragments_probed, int):
            self.num_fragments_probed = [self.num_fragments_probed]
        if isinstance(self.num_fragments_probed, (list, tuple)):
            self.num_fragments_probed = sorted(self.num_fragments_probed)
            self.k_lower = self.num_fragments_probed[0]
            self.k_upper = self.num_fragments_probed[-1]
            if self.k_lower <= 0:
                raise ValueError("`num_fragments_probed` must be positive int(s)")
        else:
            self.k_lower, self.k_upper = None, None

        if devices is None and self.max_free_wires is None:
            raise ValueError("One of arguments `devices` and max_free_wires` must be provided.")

        if isinstance(devices, (qml.Device, qml.devices.Device)):
            devices = (devices,)

        if devices is not None:
            if not isinstance(devices, SequenceType) or any(
                (not isinstance(d, (qml.Device, qml.devices.Device)) for d in devices)
            ):
                raise ValueError(
                    "Argument `devices` must be a list or tuple containing elements of type "
                    "`qml.Device` or `qml.devices.Device`"
                )

            device_wire_sizes = [len(d.wires) for d in devices]

            self.max_free_wires = self.max_free_wires or max(device_wire_sizes)
            self.min_free_wires = self.min_free_wires or min(device_wire_sizes)

        if (self.imbalance_tolerance is not None) and not (
            isinstance(self.imbalance_tolerance, (float, int)) and self.imbalance_tolerance >= 0
        ):
            raise ValueError(
                "The overall `imbalance_tolerance` is expected to be a non-negative number, "
                f"got {type(self.imbalance_tolerance)} with value {self.imbalance_tolerance}."
            )

        self.min_free_wires = self.min_free_wires or 1

    def get_cut_kwargs(
        self,
        tape_dag: MultiDiGraph,
        max_wires_by_fragment: Sequence[int] = None,
        max_gates_by_fragment: Sequence[int] = None,
        exhaustive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Derive the complete set of arguments, based on a given circuit, for passing to a graph
        partitioner.

        Args:
            tape_dag (nx.MultiDiGraph): Graph representing a tape, typically the output of
                :func:`tape_to_graph`.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                for a circuit with N gates, N - 1 attempts will be made with ``num_fragments``
                ranging from [2, N], i.e. from bi-partitioning to complete partitioning where each
                fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal kwargs being passed to a graph
            partitioner method.

        **Example**

        Deriving kwargs for a given circuit and feeding them to a custom partitioner, along with
        extra parameters specified using ``extra_kwargs``:

        >>> cut_strategy = qcut.CutStrategy(devices=dev)
        >>> cut_kwargs = cut_strategy.get_cut_kwargs(tape_dag)
        >>> cut_trials = [
        ...     my_partition_fn(tape_dag, **kwargs, **extra_kwargs) for kwargs in cut_kwargs
        ... ]

        """
        wire_depths = {}
        for g in tape_dag.nodes:
            if not isinstance(g.obj, WireCut):
                for w in g.obj.wires:
                    wire_depths[w] = wire_depths.get(w, 0) + 1 / len(g.obj.wires)
        self._validate_input(max_wires_by_fragment, max_gates_by_fragment)

        probed_cuts = self._infer_probed_cuts(
            wire_depths=wire_depths,
            max_wires_by_fragment=max_wires_by_fragment,
            max_gates_by_fragment=max_gates_by_fragment,
            exhaustive=exhaustive,
        )

        return probed_cuts

    @staticmethod
    def _infer_imbalance(k, wire_depths, free_wires, free_gates, imbalance_tolerance=None) -> float:
        """Helper function for determining best imbalance limit."""
        num_wires = len(wire_depths)
        num_gates = sum(wire_depths.values())

        avg_fragment_wires = (num_wires - 1) // k + 1
        avg_fragment_gates = (num_gates - 1) // k + 1
        if free_wires < avg_fragment_wires:
            raise ValueError(
                "`free_wires` should be no less than the average number of wires per fragment. "
                f"Got {free_wires} >= {avg_fragment_wires} ."
            )
        if free_gates < avg_fragment_gates:
            raise ValueError(
                "`free_gates` should be no less than the average number of gates per fragment. "
                f"Got {free_gates} >= {avg_fragment_gates} ."
            )
        if free_gates > num_gates - k:
            # Case where gate depth not limited (`-k` since each fragments has to have >= 1 gates):
            free_gates = num_gates
            # A small adjustment is added to the imbalance factor to prevents small ks from resulting
            # in extremely unbalanced fragments. It will heuristically force the smallest fragment size
            # to be >= 3 if the average fragment size is greater than 5. In other words, tiny fragments
            # are only allowed when average fragmeng size is small in the first place.
            balancing_adjustment = 2 if avg_fragment_gates > 5 else 0
            free_gates = free_gates - (k - 1 + balancing_adjustment)

        depth_imbalance = max(wire_depths.values()) * num_wires / num_gates - 1
        max_imbalance = free_gates / avg_fragment_gates - 1
        imbalance = min(depth_imbalance, max_imbalance)
        if imbalance_tolerance is not None:
            imbalance = min(imbalance, imbalance_tolerance)

        return imbalance

    @staticmethod
    def _validate_input(
        max_wires_by_fragment,
        max_gates_by_fragment,
    ):
        """Helper parameter checker."""
        if max_wires_by_fragment is not None:
            if not isinstance(max_wires_by_fragment, (list, tuple)):
                raise ValueError(
                    "`max_wires_by_fragment` is expected to be a list or tuple, but got "
                    f"{type(max_gates_by_fragment)}."
                )
            if any(not (isinstance(i, int) and i > 0) for i in max_wires_by_fragment):
                raise ValueError(
                    "`max_wires_by_fragment` is expected to contain positive integers only."
                )
        if max_gates_by_fragment is not None:
            if not isinstance(max_gates_by_fragment, (list, tuple)):
                raise ValueError(
                    "`max_gates_by_fragment` is expected to be a list or tuple, but got "
                    f"{type(max_gates_by_fragment)}."
                )
            if any(not (isinstance(i, int) and i > 0) for i in max_gates_by_fragment):
                raise ValueError(
                    "`max_gates_by_fragment` is expected to contain positive integers only."
                )
        if max_wires_by_fragment is not None and max_gates_by_fragment is not None:
            if len(max_wires_by_fragment) != len(max_gates_by_fragment):
                raise ValueError(
                    "The lengths of `max_wires_by_fragment` and `max_gates_by_fragment` should be "
                    f"equal, but got {len(max_wires_by_fragment)} and {len(max_gates_by_fragment)}."
                )

    def _infer_probed_cuts(
        self,
        wire_depths,
        max_wires_by_fragment=None,
        max_gates_by_fragment=None,
        exhaustive=True,
    ) -> List[Dict[str, Any]]:
        """
        Helper function for deriving the minimal set of best default partitioning constraints
        for the graph partitioner.

        Args:
            num_tape_wires (int): Number of wires in the circuit tape to be partitioned.
            num_tape_gates (int): Number of gates in the circuit tape to be partitioned.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                ``num_tape_gates - 1`` attempts will be made with ``num_fragments`` ranging from
                [2, ``num_tape_gates``], i.e. from bi-partitioning to complete partitioning where
                each fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal set of kwargs being passed to a graph
                partitioner method.
        """

        num_tape_wires = len(wire_depths)
        num_tape_gates = int(sum(wire_depths.values()))

        # Assumes unlimited width/depth if not supplied.
        max_free_wires = self.max_free_wires or num_tape_wires
        max_free_gates = self.max_free_gates or num_tape_gates

        # Assumes same number of wires/gates across all devices if min_free_* not provided.
        min_free_wires = self.min_free_wires or max_free_wires
        min_free_gates = self.min_free_gates or max_free_gates

        # The lower bound of k corresponds to executing each fragment on the largest available
        # device.
        k_lb = 1 + max(
            (num_tape_wires - 1) // max_free_wires,  # wire limited
            (num_tape_gates - 1) // max_free_gates,  # gate limited
        )
        # The upper bound of k corresponds to executing each fragment on the smallest available
        # device.
        k_ub = 1 + max(
            (num_tape_wires - 1) // min_free_wires,  # wire limited
            (num_tape_gates - 1) // min_free_gates,  # gate limited
        )

        if exhaustive:
            k_lb = max(2, k_lb)
            k_ub = num_tape_gates

        # The global imbalance tolerance, if not given, defaults to a very loose upper bound:
        imbalance_tolerance = k_ub if self.imbalance_tolerance is None else self.imbalance_tolerance

        probed_cuts = []

        if max_gates_by_fragment is None and max_wires_by_fragment is None:
            # k_lower, when supplied by a user, can be higher than k_lb if the the desired k is known:
            k_lower = self.k_lower if self.k_lower is not None else k_lb
            # k_upper, when supplied by a user, can be higher than k_ub to encourage exploration:
            k_upper = self.k_upper if self.k_upper is not None else k_ub

            if k_lower < k_lb:
                warnings.warn(
                    f"The provided `k_lower={k_lower}` is less than the lowest allowed value, "
                    f"will override and set `k_lower={k_lb}`."
                )
                k_lower = k_lb

            if k_lower > self.HIGH_NUM_FRAGMENTS:
                warnings.warn(
                    f"The attempted number of fragments seems high with lower bound at {k_lower}."
                )

            # Prepare the list of ks to explore:
            ks = list(range(k_lower, k_upper + 1))

            if len(ks) > self.HIGH_PARTITION_ATTEMPTS:
                warnings.warn(f"The numer of partition attempts seems high ({len(ks)}).")
        else:
            # When the by-fragment wire and/or gate limits are supplied, derive k and imbalance and
            # return a single partition config.
            ks = [len(max_wires_by_fragment or max_gates_by_fragment)]

        for k in ks:
            imbalance = self._infer_imbalance(
                k,
                wire_depths,
                max_free_wires if max_wires_by_fragment is None else max(max_wires_by_fragment),
                max_free_gates if max_gates_by_fragment is None else max(max_gates_by_fragment),
                imbalance_tolerance,
            )
            cut_kwargs = {
                "num_fragments": k,
                "imbalance": imbalance,
            }
            if max_wires_by_fragment is not None:
                cut_kwargs["max_wires_by_fragment"] = max_wires_by_fragment
            if max_gates_by_fragment is not None:
                cut_kwargs["max_gates_by_fragment"] = max_gates_by_fragment

            probed_cuts.append(cut_kwargs)

        return probed_cuts
