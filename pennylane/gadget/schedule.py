# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This module contains the scheduling of each phase's check measurements into
collision-free layers of check-qubit interactions.

Scheduling a round is an edge colouring of the bipartite graph between checks and qubits:
each colour is one layer, in which no check and no qubit takes part in two interactions.
By König's line colouring theorem, the minimum number of layers equals
``max(largest check weight, largest qubit degree)``. The schedules here are computed with
alternating-path recolouring, which reaches this bound; one auxiliary qubit per check is
assumed, and no hardware connectivity or gate noise is modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ir import GadgetProgram, Phase, Rounds


@dataclass(frozen=True)
class Schedule:
    """A collision-free schedule for one round of a phase.

    Args:
        phase (str): name of the phase
        layers (tuple[tuple[tuple[int, int]]]): the ``(check, qubit)`` interactions in each
            layer
        bound (int): the minimum possible number of layers,
            ``max(check weight, qubit degree)``
    """

    phase: str
    layers: tuple[tuple[tuple[int, int], ...], ...]
    bound: int

    @property
    def depth(self) -> int:
        """Number of layers per round."""
        return len(self.layers)

    @property
    def optimal(self) -> bool:
        """Whether the depth equals the minimum possible number of layers."""
        return self.depth == self.bound

    @property
    def n_interactions(self) -> int:
        """Number of check-qubit interactions per round."""
        return sum(len(layer) for layer in self.layers)

    def summary(self) -> str:
        """A one-line description of the schedule.

        Returns:
            str: the description
        """
        mark = "optimal" if self.optimal else f"ABOVE the bound of {self.bound}"
        return (
            f"phase {self.phase}: depth {self.depth} ({mark}), "
            f"{self.n_interactions} interactions per round"
        )


def schedule_phase(phase: Phase) -> Schedule:
    """Schedule one round of a phase's check measurements.

    Args:
        phase (~.Phase): the phase

    Returns:
        ~.Schedule: a schedule whose depth equals its bound

    **Example**

    >>> from pennylane import gadget
    >>> from pennylane.gadget.library import steane_code
    >>> schedule = gadget.schedule_phase(gadget.Phase.from_code("steane", steane_code()))
    >>> print(schedule.summary())
    phase steane: depth 6 (optimal), 24 interactions per round
    """
    checks = phase.checks
    edges: list[tuple[int, int]] = []
    for i in range(checks.shape[0]):
        for q in np.nonzero(checks[i])[0]:
            edges.append((i, int(q)))
    bound = max(phase.max_check_weight, phase.max_qubit_degree)
    at: dict[tuple[str, int], dict[int, tuple[str, int]]] = {}
    colour: dict[tuple[int, int], int] = {}

    def node_of(kind: str, i: int) -> tuple[str, int]:
        key = (kind, i)
        at.setdefault(key, {})
        return key

    def free_colour(node) -> int:
        used = at[node]
        c = 0
        while c in used:
            c += 1
        return c

    def canon(x, y) -> tuple[int, int]:
        return (x[1], y[1]) if x[0] == "c" else (y[1], x[1])

    for check, qubit in edges:
        u = node_of("c", check)
        v = node_of("q", qubit)
        a = free_colour(u)
        b = free_colour(v)
        if a == b:
            at[u][a] = v
            at[v][a] = u
            colour[(check, qubit)] = a
            continue
        # Swap colours a and b along the alternating path starting at v, which frees a at v.
        path: list[tuple[tuple[str, int], tuple[str, int], int]] = []
        node, c = v, a
        while c in at[node]:
            nxt = at[node][c]
            path.append((node, nxt, c))
            node = nxt
            c = b if c == a else a
        for x, y, c in path:
            at[x].pop(c, None)
            at[y].pop(c, None)
        for x, y, c in path:
            nc = b if c == a else a
            at[x][nc] = y
            at[y][nc] = x
            colour[canon(x, y)] = nc
        at[u][a] = v
        at[v][a] = u
        colour[(check, qubit)] = a

    n_layers = max(colour.values(), default=-1) + 1
    layers: list[list[tuple[int, int]]] = [[] for _ in range(n_layers)]
    for edge, c in colour.items():
        layers[c].append(edge)
    return Schedule(
        phase=phase.name,
        layers=tuple(tuple(sorted(layer)) for layer in layers),
        bound=bound,
    )


@dataclass(frozen=True)
class GadgetSchedule:
    """Schedules for every phase a gadget measures, and the gadget's total depth.

    Args:
        gadget (str): name of the gadget
        per_phase (tuple[~.Schedule]): one schedule per measured phase, in order of first use
        rounds_per_phase (tuple[tuple[str, int]]): the phase and round count of each round
            window, in program order
    """

    gadget: str
    per_phase: tuple[Schedule, ...]
    rounds_per_phase: tuple[tuple[str, int], ...]

    @property
    def total_layers(self) -> int:
        """Number of interaction layers over every round of the gadget."""
        depth = {s.phase: s.depth for s in self.per_phase}
        return sum(depth[name] * count for name, count in self.rounds_per_phase)

    @property
    def max_depth(self) -> int:
        """Largest number of layers in one round."""
        return max((s.depth for s in self.per_phase), default=0)

    def summary(self) -> str:
        """A multi-line description of the schedules.

        Returns:
            str: the description
        """
        lines = [f"schedule for {self.gadget}"]
        lines += [f"  {s.summary()}" for s in self.per_phase]
        lines.append(
            f"  total {self.total_layers} interaction layers; deepest round "
            f"{self.max_depth} layers"
        )
        return "\n".join(lines)


def schedule_gadget(program: GadgetProgram) -> GadgetSchedule:
    """Schedule every phase a gadget measures.

    Args:
        program (~.GadgetProgram): the traced gadget

    Returns:
        ~.GadgetSchedule: the schedules

    **Example**

    >>> from pennylane import gadget
    >>> from pennylane.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3)
    >>> print(gadget.schedule_gadget(measure_zz.program).summary())
    schedule for measure_zz
      phase base: depth 2 (optimal), 8 interactions per round
      phase merged: depth 2 (optimal), 10 interactions per round
      total 10 interaction layers; deepest round 2 layers
    """
    used = [op.phase for op in program.ops if isinstance(op, Rounds)]
    per_phase = tuple(schedule_phase(program.phase(name)) for name in dict.fromkeys(used))
    rounds = tuple((op.phase, op.count) for op in program.ops if isinstance(op, Rounds))
    return GadgetSchedule(gadget=program.name, per_phase=per_phase, rounds_per_phase=rounds)


__all__ = ["Schedule", "GadgetSchedule", "schedule_phase", "schedule_gadget"]
