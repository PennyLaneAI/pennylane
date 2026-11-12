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

"""Built-in decomposition rules for templates.state_preparations.partial_unary."""

from collections import Counter, defaultdict

import numpy as np

import pennylane as qp
from pennylane import allocate, math
from pennylane.templates.state_preparations.partial_unary import (
    PartialUnaryStatePreparation,
    PUIsometryFinder,
)
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires


def _reduced_binary_row_basis(values, max_rank):
    """Return a reduced row basis of packed binary vectors, or ``None`` above ``max_rank``."""
    basis = {}
    for value in values:
        reduced = int(value)
        for pivot in sorted(basis, reverse=True):
            if reduced & (1 << pivot):
                reduced ^= basis[pivot]
        if not reduced:
            continue

        pivot = reduced.bit_length() - 1
        for other_pivot, row in tuple(basis.items()):
            if row & (1 << pivot):
                basis[other_pivot] = row ^ reduced
        basis[pivot] = reduced

        if len(basis) > max_rank:
            return None

    return basis


def _is_affine_subspace(basis_states, n_subspace):
    """Return whether ``basis_states`` lie in an affine subspace of the given width."""
    anchor = int(basis_states[0])
    translated = (int(state) ^ anchor for state in basis_states)
    return _reduced_binary_row_basis(translated, n_subspace) is not None


def _clear_nonpivot_columns(translated, rows, pivots, n_qubits):
    """Clear non-pivot columns and return the corresponding CNOT gates."""
    pivot_set = set(pivots)
    circuit = []
    for target_bit in range(n_qubits - 1, -1, -1):
        if target_bit in pivot_set:
            continue
        for row, control_bit in zip(rows, pivots, strict=True):
            if row & (1 << target_bit):
                circuit.append(("CNOT", n_qubits - 1 - control_bit, n_qubits - 1 - target_bit))
                target_mask = 1 << target_bit
                for i, value in enumerate(translated):
                    if value & (1 << control_bit):
                        translated[i] = value ^ target_mask
    return circuit


def _move_pivots_to_leading_wires(translated, pivots, n_qubits):
    """Move pivot columns to the leading wires and return the corresponding SWAP gates."""
    locations = list(pivots)
    circuit = []
    for i, desired_bit in enumerate(range(n_qubits - 1, n_qubits - 1 - len(pivots), -1)):
        current_bit = locations[i]
        if current_bit == desired_bit:
            continue

        circuit.append(("SWAP", [n_qubits - 1 - current_bit, n_qubits - 1 - desired_bit]))
        current_mask, desired_mask = 1 << current_bit, 1 << desired_bit
        for j, value in enumerate(translated):
            if bool(value & current_mask) != bool(value & desired_mask):
                translated[j] = value ^ current_mask ^ desired_mask

        locations[i] = desired_bit
    return circuit


def _find_affine_subspace_isometry(basis_states, n_qubits, n_subspace):
    """Find a Clifford-only isometry for support in a minimal affine binary subspace.

    Returns ``None`` if the translated support has binary rank larger than ``n_subspace``.
    Otherwise, returns a circuit of X, CNOT and SWAP gates together with the induced mapping
    into the subspace register.
    """
    states = list(basis_states)
    n_remainder = n_qubits - n_subspace
    remainder_mask = (1 << n_remainder) - 1

    if all((state & remainder_mask) == 0 for state in states):
        return [], {i: state >> n_remainder for i, state in enumerate(states)}

    anchor = states[0]
    translated = [state ^ anchor for state in states]

    # Row operations only change the chosen basis of the affine span the column operations
    # recorded below are the physical Clifford circuit.
    basis = _reduced_binary_row_basis(translated, n_subspace)
    if basis is None:
        return None

    pivots = sorted(basis, reverse=True)
    rows = [basis[pivot] for pivot in pivots]
    circuit = []

    # Translate the affine support to a linear subspace.
    for bit in range(n_qubits - 1, -1, -1):
        if anchor & (1 << bit):
            wire = n_qubits - 1 - bit
            circuit.append(("X", wire))

    # In reduced row-echelon form, each non-pivot column is a combination of pivot columns.
    # Clear it with CNOTs controlled by those pivots.
    circuit.extend(_clear_nonpivot_columns(translated, rows, pivots, n_qubits))

    # Move the pivot columns to the leading wires, updating pivot locations after each swap.
    circuit.extend(_move_pivots_to_leading_wires(translated, pivots, n_qubits))

    if any(value & remainder_mask for value in translated):  # pragma: no cover
        raise AssertionError("Affine preconditioning failed to map support into the subspace.")

    bijection = {i: value >> n_remainder for i, value in enumerate(translated)}
    return circuit, bijection


def _partial_unary_state_prep_resources(coefficients, wires, indices, work_wires):
    """Compute the resources for the partial unary iteration state preparation technique.
    These resource counts are numerically obtained heuristics, extended to guarantee all
    resource reps that may appear are included at least once."""
    # pylint: disable=unused-argument
    num_entries = len(indices)
    num_wires = 1 if isinstance(wires, int) else len(wires)
    if num_entries == 1:
        return {qp.MultiX(Bool[num_wires], Wire[num_wires]): 1}

    n_subspace = max(math.ceil_log2(num_entries), 1)
    is_affine = _is_affine_subspace(indices, n_subspace)
    resources = defaultdict(int)
    # QROM needs n_subspace - 1 work wires, while Toffoli needs one zeroed work wire.
    needed_work_wires = max(n_subspace - 1, 1)
    effective_num_wires = num_wires + max(len(work_wires) - needed_work_wires, 0)
    state_type, _ = math.get_abstract_and_cast_dtype(coefficients)
    resources[
        qp.MultiplexerStatePreparation(state_type[2**n_subspace], wires=Wire[n_subspace])
    ] += 1

    if is_affine:
        resources[qp.X] += effective_num_wires
        resources[qp.CNOT] += n_subspace * max(effective_num_wires - n_subspace, 1)
        resources[qp.SWAP] += effective_num_wires
        return resources

    R = effective_num_wires - n_subspace
    # Cap by num_entries: the isometry finder (PUIsometryFinder) can never actually produce a
    # batch larger than the number of states being mapped, regardless of how many remainder
    # wires are available.
    main_pui_batch_size = min(1 << math.floor_log2(max(R, 1)), num_entries)

    qrom_reps = {
        p: qp.QROM(
            bitstrings=Bool[p, p],
            control_wires=Wire[n_subspace],
            target_wires=Wire[p],
            work_wires=Wire[n_subspace - 1],
            clean=True,
        )
        for p in range(1, main_pui_batch_size + 1)
    }

    resources[qrom_reps[main_pui_batch_size]] += max(num_entries // main_pui_batch_size, 1)
    for p in range(1, main_pui_batch_size):
        resources[qrom_reps[p]] += 1

    ctrl_basis_rep = qp.ctrl(
        qp.MultiX(Bool[effective_num_wires - 1], Wire[effective_num_wires - 1]), Wire[1]
    )
    resources[ctrl_basis_rep] += num_entries

    embed_rep = qp.MultiX(Bool[n_subspace], Wire[n_subspace])
    resources[embed_rep] += 2 * (num_entries // main_pui_batch_size + 1)

    resources[qp.SWAP] += effective_num_wires

    num_toffolis = int(effective_num_wires / 10) + 1
    mcx_rep = qp.MultiControlledX(Wire[3], work_wires=Wire[1], work_wire_type="zeroed")
    resources[mcx_rep] += num_toffolis
    resources[qp.MultiX(Bool[1], Wire[1])] += 2 * num_toffolis

    return resources


def _select_isometry(indices, num_wires, n_subspace):
    """Select the Clifford-only affine isometry when available, or use the generic finder."""
    affine_isometry = _find_affine_subspace_isometry(indices, num_wires, n_subspace)
    if affine_isometry is not None:
        circuit, bijection = affine_isometry
        return circuit, [], bijection, True

    iso_finder = PUIsometryFinder(np.array(indices), num_wires)
    circuit, fanout_bits, bijection = iso_finder.find_isometry()
    return circuit, fanout_bits, bijection, False


def _apply_affine_isometry(circuit, wires):
    """Apply the inverse of a Clifford-only affine isometry."""
    for op_type, *data in reversed(circuit):
        if op_type == "X":
            qp.X(wires[data[0]])
        elif op_type == "CNOT":
            qp.CNOT([wires[data[0]], wires[data[1]]])
        else:
            qp.SWAP([wires[idx] for idx in data[0]])


def _branch_probs(circuit):
    """Compute the relative frequencies of the operation types in the isometry circuit of
    ``PartialUnaryStatePreparation``, see the ``PUIsometryFinder.circuit`` structure:
     - type ``0``: QROM/partial unary iterator
     - type ``1``: Fanout
     - type ``2``: SWAP
     - type ``3``: Toffoli
    Also compute the relative frequencies of the batch sizes in the partial unary iterators."""
    type_counts = Counter(_type for _type, *_ in circuit)
    batch_counts = Counter(data[1] - data[0] for _type, *data in circuit if _type == 0)
    type_probs = [type_counts[_type] / len(circuit) for _type in range(4)]
    # Sort by batch size so cond branch order (and thus hint order) is deterministic.
    batch_probs = {b: batch_counts[b] / type_counts[0] for b in sorted(batch_counts)}
    return type_probs, batch_probs


def _partial_unary_dense_state(coefficients, bijection, n_subspace):
    """Place coefficients in the dense subspace using a gather supported by Catalyst."""
    # TODO: Revisit using qp.math.scatter instead when Catalyst supports it without sorted/unique index guarantees [sc-135792]
    num_entries = len(bijection)
    # Reverse the static mapping so each dense position selects its coefficient.
    # E.g., destinations [2, 0, 1] need coefficient indices [1, 2, 0].
    inverse = {bijection[i]: i for i in range(num_entries)}

    # Unoccupied positions select the appended zero at index num_entries.
    padded_coefficients = math.concatenate([coefficients, math.zeros_like(coefficients[:1])])
    gather_indices = [inverse.get(i, num_entries) for i in range(2**n_subspace)]

    # E.g., [a, b, c, 0] gathered at [1, 2, 0, 3] gives [b, c, a, 0].
    return math.take(padded_coefficients, gather_indices, axis=0)


def _partial_unary_state_prep_core(coefficients, wires, indices, work_wires):
    """Compute the decomposition of the partial unary iteration state preparation technique.
    This core method is used by the two rules below, which only differ by the work
    wire management."""
    num_entries = len(indices)
    if num_entries == 1:
        qp.MultiX(math.int_to_binary(indices[0], len(wires)), wires)
        return

    n_subspace = max(math.ceil_log2(num_entries), 1)
    needed_work_wires = max(n_subspace - 1, 1)  # Need n_subspace-1 for QROM, 1 for Toffoli
    wires = Wires(wires)
    if len(work_wires) > needed_work_wires:
        # There is an easy way to use more work wires: Pretend they are part of the system wires,
        # which gives the isometry circuit more space to move states around. Adding the excess
        # work wires to the beginning of the `wires` register allows us to keep `indices`
        # unchanged. It is not obvious whether this approach or a different wire ordering will
        # be cheaper in terms of quantum resources used in the isometry circuit.
        wires = Wires(work_wires[needed_work_wires:]) + wires

    circuit, fanout_bits, bijection, is_affine = _select_isometry(indices, len(wires), n_subspace)

    subspace_wires = Wires(wires[:n_subspace])
    nonsubspace_wires = Wires(wires[n_subspace:])

    # Step 1: Dense state preparation
    dense_state = _partial_unary_dense_state(coefficients, bijection, n_subspace)
    _, cast_dtype = math.get_abstract_and_cast_dtype(coefficients)
    qp.MultiplexerStatePreparation(math.cast(dense_state, cast_dtype), subspace_wires)

    if not circuit:
        return

    if is_affine:
        _apply_affine_isometry(circuit, wires)
        return

    # The keys of batch_probs are the occurring batch sizes
    type_probs, batch_probs = _branch_probs(circuit)

    if qp.compiler.active() or qp.capture.enabled():
        fanout_bits = qp.math.array(fanout_bits, like="jax")
        circuit = qp.math.array(circuit, like="jax")
        wires = qp.math.array(wires, like="jax")

        def del_cwire(control):
            """Return wires with the wire at position ``control`` deleted (qjit-compatible)"""
            return qp.math.delete(wires, control, assume_unique_indices=True)

    else:

        def del_cwire(control):
            """Return wires with the wire at position ``control`` deleted."""
            return wires[:control] + wires[control + 1 :]

    # Step 2: Apply the inverse of the isometry circuit
    @qp.for_loop(len(circuit) - 1, -1, -1)
    def main_loop(i):
        """This main loop exclusively reads ``circuit[i]`` and then calls a conditional function,
        with the branches corresponding to the different data put into ``circuit`` by
        ``PUIsometryFinder.find_isometry``. The first entry in ``circuit[i]``, ``_type``, encodes
        the operation to be applied:

        - 0: Partial unary iterator (realized via shifted ``QROM``)
        - 1: Fanout (realized via controlled ``MultiX``)
        - 2: SWAP of two remainder qubits (realized via a simple ``SWAP``)
        - 3: Toffoli (realized via ``MultiControlledX`` in order to use work wires for elbow decomp)

        Note that the entries in ``circuit`` have been padded with zeros to all have length 5,
        the ``_type`` and four data integers, in order to enable casting to an ``ndarray``.
        Each branch of the conditional then reads out only the entries it needs, and discards
        the padded zeros. The data type of the array depends on the number of qubits :math:`n`.
        """
        # pylint: disable=cell-var-from-loop

        _type, *data = circuit[i]

        @qp.cond(_type == 0)
        @qp.hint({"branch-prob": type_probs[0]})
        def branches():
            """The first branch calls a partial unary iterator, in form of a ``QROM`` of a given
            size. In order to avoid dynamic shapes, we define an inner conditional function
            ``qrom_branches`` and register ``iso_finder.m`` many branches to it, one for each
            possible batch size. This is not optimal in terms of coding practice and resource
            accounting, but it works in a stable manner and is sufficiently efficient.
            """
            k_start, k = data[:2]
            b = k - k_start
            _work_wires = work_wires[: n_subspace - 1]

            first_case_b, *other_case_bs = batch_probs

            @qp.cond(b == first_case_b)
            @qp.hint({"branch-prob": batch_probs[first_case_b]})
            def qrom_branches():
                target_wires = nonsubspace_wires[:first_case_b]
                qp.QROM(np.eye(first_case_b, dtype=int), subspace_wires, target_wires, _work_wires)

            for case_b in other_case_bs:
                # Register an additional branch to ``qrom_branches`` for each occurring batch size.
                # Passing a default argument makes sure that we don't use an outdated closure
                # variable `case_b` when calling the function.
                @qrom_branches.else_if(b == case_b)
                @qp.hint({"branch-prob": batch_probs[case_b]})
                def _(case_b=case_b):
                    target_wires = nonsubspace_wires[:case_b]
                    qp.QROM(np.eye(case_b, dtype=int), subspace_wires, target_wires, _work_wires)

            # Realize PUI via QROM, shifted by k_start
            k_start_bits = math.int_to_binary(k_start, n_subspace)
            qp.MultiX(k_start_bits, subspace_wires)
            qrom_branches()
            qp.MultiX(k_start_bits, subspace_wires)

        @branches.else_if(_type == 1)
        @qp.hint({"branch-prob": type_probs[1]})
        def fanout():
            """The second branch calls a fan-out, in form of a controlled ``MultiX``,
            controlled by one of the (first ``iso_finder.m``) remainder qubits and targeting
            all other qubits.
            """
            control, fanout_bit_pointer = data[:2]
            target_wires = del_cwire(control)
            qp.ctrl(qp.MultiX(fanout_bits[fanout_bit_pointer], target_wires), wires[control])

        @branches.else_if(_type == 2)
        @qp.hint({"branch-prob": type_probs[2]})
        def swap():
            """The third branch is a simple SWAP gate."""
            _wires = [wires[idx] for idx in data[:2]]
            qp.SWAP(_wires)

        @branches.else_if(_type == 3)
        @qp.hint({"branch-prob": type_probs[3]})
        def toffoli():
            """The fourth branch is a simple Toffoli gate. We manually realize the control values
            in order to avoid dynamic control values in MultiControlledX. We don't use ``Toffoli``
            directly because it does not have ``work_wires`` and we would like to use the cheaper
            elbow-based decomposition.
            """
            _wires = [wires[idx] for idx in data[:3]]
            cval = 1 - math.array(data[3:4], like=math.get_interface(data[3]))
            qp.MultiX(cval, _wires[1:2])
            qp.MultiControlledX(_wires, work_wires=work_wires[0], work_wire_type="zeroed")
            qp.MultiX(cval, _wires[1:2])

        branches()

    main_loop()  # pylint: disable=no-value-for-parameter


def _partial_unary_state_prep_work_wires(coefficients, wires, indices, work_wires):
    # pylint: disable=unused-argument
    work_wires = [] if work_wires is None else list(work_wires)
    num_entries = len(indices)
    is_affine = _is_affine_subspace(indices, max(math.ceil_log2(num_entries), 1))
    needed_work_wires = (
        0 if (is_affine or num_entries == 1) else max(math.ceil_log2(num_entries) - 1, 1)
    )
    need_to_allocate = max(needed_work_wires - len(work_wires), 0)
    return {"zeroed": need_to_allocate}


@qp.register_resources(
    _partial_unary_state_prep_resources,
    work_wires=_partial_unary_state_prep_work_wires,
    exact=False,
)
def _partial_unary_state_prep(coefficients, wires, indices, work_wires):
    """Compute the decomposition of the partial unary iteration state preparation technique.

    Dynamically allocates any work wires still missing after the ones passed to
    ``PartialUnaryStatePreparation``. Allocating zero wires records no operators.
    """
    work_wires = [] if work_wires is None else list(work_wires)
    need_to_allocate = _partial_unary_state_prep_work_wires(
        coefficients, wires, indices, work_wires
    )["zeroed"]
    with allocate(need_to_allocate, state="zero", restored=True) as extra_work_wires:
        work_wires = work_wires + list(extra_work_wires)
        _partial_unary_state_prep_core(coefficients, wires, indices, work_wires)


qp.add_decomps(PartialUnaryStatePreparation, _partial_unary_state_prep)
