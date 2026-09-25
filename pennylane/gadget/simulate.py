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
This module contains the Stim-based phenomenological simulation of a gadget, used by
:func:`~pennylane.gadget.verify`. It requires `Stim <https://github.com/quantumlib/Stim>`__.

The circuit measures each check as a whole Pauli product, applies X and Z errors to every
live qubit before each round and flips each measurement with the same probability. No
syndrome-extraction circuit is assumed, so the resulting distance is phenomenological, not
circuit-level. Only the gadget's declared outcomes are included as observables: an error
that corrupts a logical qubit the gadget preserves, without changing an outcome, is not
detected by this check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import DetectorLayout
from .ir import Deform, Detach, Frame, GadgetProgram, Observe, RecordExpr, RecordTerm, Rounds

try:  # pragma: no cover - import guard
    import stim
except ImportError as exc:  # pragma: no cover
    raise ImportError("gadget.simulate requires stim") from exc


class SimulationUnsupported(RuntimeError):
    """Raised when a gadget cannot be expressed in the phenomenological model."""


@dataclass
class SimulationResult:
    """The result of :func:`check`.

    Args:
        deterministic (bool): whether every detector is deterministic without noise
        distance (int or None): Smallest number of faults that flips an outcome without
            triggering a detector, or ``None`` if not determined.
        detail (str): how the result was obtained, or why no distance was found
        circuit (stim.Circuit or None): the noisy circuit
        dem (stim.DetectorErrorModel or None): its detector error model
    """

    deterministic: bool
    distance: int | None
    detail: str
    circuit: "stim.Circuit | None" = None
    dem: "stim.DetectorErrorModel | None" = None


def build_circuit(
    program: GadgetProgram,
    layout: DetectorLayout,
    p: float = 1e-3,
) -> tuple["stim.Circuit", dict[RecordTerm, int], dict[int, int], int]:
    """Build the phenomenological Stim circuit of a gadget.

    The circuit starts with a noiseless round of the entry phase on a fresh codeword; its
    outcomes stand in for the entry syndrome. The detectors and observables of ``layout``
    are appended at the end.

    Args:
        program (~.GadgetProgram): the traced gadget
        layout (~.DetectorLayout): detectors and observables to include
        p (float): probability of each X error, Z error and measurement flip

    Returns:
        tuple[stim.Circuit, dict[RecordTerm, int], dict[int, int], int]: the circuit, the
        measurement index of each record and of each entry-syndrome bit, and the total
        number of measurements

    Raises:
        SimulationUnsupported: if a check is empty, or a detector refers to a record or
        entry-syndrome bit the circuit does not measure

    **Example**

    >>> from pennylane import gadget
    >>> from pennylane.gadget.simulate import build_circuit
    >>> from pennylane.gadget.library import steane_memory
    >>> _, _, memory = steane_memory(rounds=3)
    >>> layout = gadget.derive_detectors(memory.program)
    >>> circuit, _, _, n_meas = build_circuit(memory.program, layout)
    >>> n_meas, circuit.num_detectors
    (24, 18)
    """
    circuit = stim.Circuit()
    entry = program.phase(_first_phase(program))
    rec: dict[RecordTerm, int] = {}
    entry_meas: dict[int, int] = {}
    count = 0

    # Ideal entry round: prepare a codeword of the entry phase and read its checks. These
    # outcomes are the entry syndrome the first detectors are compared with.
    active = np.nonzero(entry.active)[0].tolist()
    circuit.append("R", active)
    for i, axis in enumerate(entry.check_axes):
        _append_mpp(circuit, entry.checks[i], axis, 0.0)
        entry_meas[i] = count
        count += 1

    current = entry
    for op in program.ops:
        if isinstance(op, Deform):
            target = program.phase(op.to_phase)
            for qubit, basis in op.init:
                circuit.append("RX" if basis == "x" else "R", [qubit])
            current = target
            continue

        if isinstance(op, Rounds):
            phase = program.phase(op.phase)
            block = program.record(op.record)
            live = np.nonzero(phase.active)[0].tolist()
            for r in range(op.count):
                if p:
                    circuit.append("X_ERROR", live, p)
                    circuit.append("Z_ERROR", live, p)
                for c in range(block.width):
                    _append_mpp(circuit, phase.checks[c], phase.check_axes[c], p)
                    rec[RecordTerm(block.name, r, c)] = count
                    count += 1
            current = phase
            continue

        if isinstance(op, Detach):
            block = program.record(op.record)
            if p:
                live = np.nonzero(current.active)[0].tolist()
                circuit.append("X_ERROR", live, p)
                circuit.append("Z_ERROR", live, p)
            for i, (qubit, basis) in enumerate(op.measure_out):
                circuit.append("MX" if basis == "x" else "M", [qubit], p)
                rec[RecordTerm(block.name, 0, i)] = count
                count += 1
            current = program.phase(op.to_phase)
            continue

        if isinstance(op, (Observe, Frame)):
            continue

        raise SimulationUnsupported(f"no phenomenological model for op {type(op).__name__}")

    for det in layout.detectors:
        circuit.append("DETECTOR", _targets(det.expr, rec, entry_meas, count))
    for obs in layout.observables:
        circuit.append("OBSERVABLE_INCLUDE", _targets(obs.expr, rec, entry_meas, count), obs.index)
    return circuit, rec, entry_meas, count


def _append_mpp(circuit: "stim.Circuit", row: np.ndarray, axis: str, p: float) -> None:
    support = np.nonzero(row)[0].tolist()
    if not support:
        raise SimulationUnsupported("a check with empty support cannot be measured")
    make = stim.target_x if axis == "x" else stim.target_z
    targets = []
    for i, q in enumerate(support):
        if i:
            targets.append(stim.target_combiner())
        targets.append(make(int(q)))
    circuit.append("MPP", targets, p)


def _targets(
    expr: RecordExpr,
    rec: dict[RecordTerm, int],
    entry_meas: dict[int, int],
    total: int,
) -> list:
    out = []
    for term in sorted(expr.terms, key=lambda t: (t.block, t.round, t.check)):
        if term not in rec:
            raise SimulationUnsupported(f"record {term} was never measured")
        out.append(stim.target_rec(rec[term] - total))
    for i in sorted(expr.entry):
        if i not in entry_meas:
            raise SimulationUnsupported(f"entry-syndrome bit {i} has no entry measurement")
        out.append(stim.target_rec(entry_meas[i] - total))
    if not out:
        raise SimulationUnsupported("an empty parity cannot be a detector or observable")
    return out


def check(
    program: GadgetProgram,
    layout: DetectorLayout,
    p: float = 1e-3,
    search_degree: int = 4,
) -> SimulationResult:
    """Check that the detectors are deterministic and find the outcome fault distance.

    Determinism is checked by building the noiseless detector error model, which Stim
    rejects if any detector is non-deterministic. The distance is the length of the
    shortest undetectable logical error found by
    ``stim.Circuit.search_for_undetectable_logical_errors``.

    Args:
        program (~.GadgetProgram): the traced gadget
        layout (~.DetectorLayout): detectors and observables to check
        p (float): error probability of the noisy model
        search_degree (int): Bound on the size of the detection-event sets the search
            explores. Errors beyond the bound are not considered, so the distance is an
            upper bound.

    Returns:
        ~.SimulationResult: the result

    **Example**

    >>> from pennylane import gadget
    >>> from pennylane.gadget.simulate import check
    >>> from pennylane.gadget.library import rep_code_zz_merge
    >>> _, _, measure_zz = rep_code_zz_merge(d=3, merged_rounds=2)
    >>> result = check(measure_zz.program, gadget.derive_detectors(measure_zz.program))
    >>> result.deterministic, result.distance
    (True, 2)
    """
    ideal, _, _, _ = build_circuit(program, layout, p=0.0)
    try:
        ideal.detector_error_model(allow_gauge_detectors=False)
    except Exception as exc:
        return SimulationResult(
            deterministic=False, distance=None, detail=str(exc).strip().splitlines()[0]
        )

    noisy, _, _, _ = build_circuit(program, layout, p=p)
    dem = noisy.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
    if not layout.observables:
        return SimulationResult(
            deterministic=True,
            distance=None,
            detail="no observables declared, so there is no outcome to corrupt",
            circuit=noisy,
            dem=dem,
        )
    try:
        errors = noisy.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=search_degree,
            dont_explore_edges_with_degree_above=search_degree,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    except Exception as exc:
        return SimulationResult(
            deterministic=True,
            distance=None,
            detail=(
                "stim found no undetectable logical error within the search bound "
                f"(degree <= {search_degree}): {str(exc).strip().splitlines()[0]}"
            ),
            circuit=noisy,
            dem=dem,
        )
    return SimulationResult(
        deterministic=True,
        distance=len(errors),
        detail=(
            f"outcome observables only, p={p}, search degree bound {search_degree}, "
            f"{dem.num_errors} error mechanisms in the DEM"
        ),
        circuit=noisy,
        dem=dem,
    )


def _first_phase(program: GadgetProgram) -> str:
    for op in program.ops:
        if isinstance(op, Rounds):
            return op.phase
        if isinstance(op, (Deform, Detach)):
            break
    return program.phases[0].name


__all__ = ["SimulationResult", "SimulationUnsupported", "build_circuit", "check"]
