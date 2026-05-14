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
"""Functionality to compare non-unitary circuits by comparing individual branches (for different measurement outcomes)"""

# pylint: disable=not-callable, too-many-arguments, too-many-branches, too-many-statements, too-many-nested-blocks

import numpy as np

import pennylane as qp


def _get_branch_outputs(qnode_fn, in_state, n_tot, kw):
    """
    Run a circuit and return unique branch output states (ordered by branch index).

    With tree-traversal MCM, snapshots return one state per shot. States from the
    same measurement branch are identical. We deduplicate to get one state per branch.

    Returns:
        list[np.ndarray]: unique output states, one per branch, in order of first appearance
    """
    out_states = qp.snapshots(qnode_fn)(in_state, **kw)["state"]

    # If it's a single state (no measurements or single branch), normalize to list
    if isinstance(out_states, np.ndarray):
        if out_states.ndim == 1:
            return [out_states]
        # 2D array: each row is a state from a different shot
        out_states = list(out_states)

    # If it's already a list of length 2^n_tot, it's the full state vector not branches
    if len(out_states) == 2 ** n_tot and isinstance(
        out_states[0], (int, float, complex, np.number)
    ):
        return [np.array(out_states)]

    # Deduplicate: preserve order of first appearance (= branch index order with tree-traversal)
    unique_states = []
    for state in out_states:
        state = np.array(state)
        is_new = True
        for existing in unique_states:
            if np.allclose(state, existing, atol=1e-10):
                is_new = False
                break
        if is_new:
            unique_states.append(state)

    return unique_states


def compare_circuits(
    circuit1,
    circuit2,
    wires,
    aux_wires1=None,
    aux_wires2=None,
    aux_state_io1=None,
    aux_state_io2=None,
    verbose=False,
    **kwargs,
):
    r"""
    Check (non-unitary) circuit identities by comparing two circuits that may
    contain mid-circuit measurements.

    Both circuits are expected to act as the same unitary on the ``wires`` subspace.
    Comparison is done branch-by-branch: for each measurement branch, the effective
    action on the main wires must match between the two circuits (up to a consistent
    global phase across all input states).

    Args:
        circuit1 (callable): first undecorated ``qfunc``
        circuit2 (callable): second undecorated ``qfunc``
        wires: the main (logical) wires shared by both circuits
        aux_wires1 (optional): auxiliary wires for ``circuit1``
        aux_wires2 (optional): auxiliary wires for ``circuit2``
        aux_state_io1 (list, optional): ``[input_state, output_state]`` on aux wires of circuit1
        aux_state_io2 (list, optional): ``[input_state, output_state]`` on aux wires of circuit2
        verbose (bool): Whether or not to display the circuit identities that are being checked
        kwargs: additional kwargs forwarded to both circuits
            (use ``kwargs1``/``kwargs2`` keys for circuit-specific args)

    Returns:
        tuple[bool, str]: (success, message) where message indicates "exact" or "up to global phase"
    """
    kwargs1 = kwargs.pop("kwargs1", {})
    kwargs2 = kwargs.pop("kwargs2", {})
    kwargs1 = {**kwargs, **kwargs1}
    kwargs2 = {**kwargs, **kwargs2}

    wires = qp.wires.Wires(wires)
    n = len(wires)

    # Handle defaults for aux_wires
    if aux_wires1 is None:
        aux_wires1 = qp.wires.Wires([])
    else:
        aux_wires1 = qp.wires.Wires(aux_wires1)
    if aux_wires2 is None:
        aux_wires2 = qp.wires.Wires([])
    else:
        aux_wires2 = qp.wires.Wires(aux_wires2)

    n_aux1 = len(aux_wires1)
    n_aux2 = len(aux_wires2)
    n_tot1 = n + n_aux1
    n_tot2 = n + n_aux2

    # Handle defaults for aux_state_io (|0...0> in and out)
    if aux_state_io1 is None:
        if n_aux1 > 0:
            zero_state = np.zeros(2**n_aux1)
            zero_state[0] = 1.0
            aux_state_io1 = [zero_state, zero_state]
        else:
            aux_state_io1 = [np.array([1.0]), np.array([1.0])]
    if aux_state_io2 is None:
        if n_aux2 > 0:
            zero_state = np.zeros(2**n_aux2)
            zero_state[0] = 1.0
            aux_state_io2 = [zero_state, zero_state]
        else:
            aux_state_io2 = [np.array([1.0]), np.array([1.0])]

    if n_aux1 > 0:
        assert int(np.log2(len(aux_state_io1[0]))) == n_aux1
    if n_aux2 > 0:
        assert int(np.log2(len(aux_state_io2[0]))) == n_aux2

    all_wires1 = wires + aux_wires1 if n_aux1 > 0 else wires
    all_wires2 = wires + aux_wires2 if n_aux2 > 0 else wires

    if verbose:
        print("Circuit 1:\n")
        print(qp.draw(circuit1)(**kwargs1))
        print("=?= Circuit 2:\n")
        print(qp.draw(circuit2)(**kwargs2))

    @qp.set_shots(10)
    @qp.qnode(qp.device("default.qubit"), mcm_method="tree-traversal")
    def qnode1(in_state, **kw):
        qp.StatePrep(in_state, wires=all_wires1)
        circuit1(**kw)
        qp.Snapshot(measurement=qp.state(), tag="state")
        return qp.expval(qp.Z(wires[0]))

    @qp.set_shots(10)
    @qp.qnode(qp.device("default.qubit"), mcm_method="tree-traversal")
    def qnode2(in_state, **kw):
        qp.StatePrep(in_state, wires=all_wires2)
        circuit2(**kw)
        qp.Snapshot(measurement=qp.state(), tag="state")
        return qp.expval(qp.Z(wires[0]))

    in_states = np.eye(2**n)

    if n_aux1 > 0:
        in_states1 = np.array([np.kron(s, aux_state_io1[0]) for s in in_states])
    else:
        in_states1 = in_states

    if n_aux2 > 0:
        in_states2 = np.array([np.kron(s, aux_state_io2[0]) for s in in_states])
    else:
        in_states2 = in_states

    success = True
    is_exact = True  # Track whether match is exact or only up to global phase

    # Track the reference phase per branch to enforce consistency across all inputs
    branch_phases = {}  # branch_idx -> reference phase (complex number)

    # For each input basis state, get branch outputs and compare branch-by-branch
    for i, (in1, in2) in enumerate(zip(in_states1, in_states2)):
        branches1 = _get_branch_outputs(qnode1, in1, n_tot1, kwargs1)
        branches2 = _get_branch_outputs(qnode2, in2, n_tot2, kwargs2)

        if len(branches1) != len(branches2):
            print(
                f"Branch count mismatch for input |{np.binary_repr(i, n)}>: "
                f"circuit1 has {len(branches1)} branches, circuit2 has {len(branches2)}"
            )
            success = False
            continue

        # Compare each branch
        for b_idx, (state1, state2) in enumerate(zip(branches1, branches2)):
            # Extract main-wire state by projecting out aux
            if n_aux1 > 0:
                reshaped1 = state1.reshape(2**n, 2**n_aux1)
                main1 = reshaped1 @ np.conj(aux_state_io1[1])
            else:
                main1 = state1

            if n_aux2 > 0:
                reshaped2 = state2.reshape(2**n, 2**n_aux2)
                main2 = reshaped2 @ np.conj(aux_state_io2[1])
            else:
                main2 = state2

            # Skip if both states are zero (e.g. post-selected away)
            if np.linalg.norm(main1) < 1e-10 and np.linalg.norm(main2) < 1e-10:
                continue

            # Check exact match first
            if np.allclose(main1, main2, atol=1e-8):
                # Exact match — if we already have a phase for this branch, it must be 1
                if b_idx in branch_phases:
                    if not np.isclose(branch_phases[b_idx], 1.0, atol=1e-8):
                        print(
                            f"Inconsistent phase for branch {b_idx}: "
                            f"input |{np.binary_repr(i, n)}> gives exact match (phase=0), "
                            f"but previous inputs gave phase "
                            f"{np.angle(branch_phases[b_idx])/np.pi:.4f}\u03c0"
                        )
                        success = False
                else:
                    branch_phases[b_idx] = 1.0 + 0.0j
                continue

            # Not exact — check if match up to global phase
            idx = np.argmax(np.abs(main1))
            if np.abs(main1[idx]) > 1e-10 and np.abs(main2[idx]) > 1e-10:
                phase = main1[idx] / main2[idx]
                if np.allclose(main1, phase * main2, atol=1e-8):
                    # Valid phase relationship — enforce consistency
                    if b_idx in branch_phases:
                        if not np.isclose(phase, branch_phases[b_idx], atol=1e-8):
                            print(
                                f"Inconsistent phase for branch {b_idx}: "
                                f"input |{np.binary_repr(i, n)}> gives phase "
                                f"{np.angle(phase)/np.pi:.4f}\u03c0, "
                                f"but previous inputs gave phase "
                                f"{np.angle(branch_phases[b_idx])/np.pi:.4f}\u03c0"
                            )
                            success = False
                        else:
                            is_exact = False
                    else:
                        branch_phases[b_idx] = phase
                        is_exact = False
                else:
                    print(f"Mismatch for input |{np.binary_repr(i, n)}>, branch {b_idx}:")
                    print(f"  circuit1 main state: {main1}")
                    print(f"  circuit2 main state: {main2}")
                    success = False
            else:
                print(
                    f"Mismatch for input |{np.binary_repr(i, n)}>, branch {b_idx}: "
                    f"zero entries don't align"
                )
                success = False

    if success:
        if is_exact:
            return True, "exact"

        # Report the phase(s) — if all branches have the same phase, report one value
        unique_phases = list(
            {
                np.round(np.angle(p) / np.pi, 6)
                for p in branch_phases.values()
                if not np.isclose(p, 1.0)
            }
        )
        if len(unique_phases) == 1:
            return True, f"up to global phase {unique_phases[0]:.4f}\u03c0"

        phase_strs = ", ".join(
            f"branch {k}: {np.angle(v)/np.pi:.4f}\u03c0"
            for k, v in sorted(branch_phases.items())
            if not np.isclose(v, 1.0)
        )
        return True, f"up to global phase (per branch: {phase_strs})"

    return False, "mismatch"
