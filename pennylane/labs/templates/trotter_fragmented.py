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
"""Contains the Trotter templates for fragmented Hamiltonians."""

from collections import defaultdict

import numpy as np

import pennylane as qp
from pennylane.decomposition.decomposition_rule import add_decomps, register_resources
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.resource.resource import Resources
from pennylane.tape.qscript import QuantumScript
from pennylane.wires import Wires
from pennylane.queuing import QueuingManager, apply

# pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False

class TrotterCGF(Operation):
    r"""Second-order Suzuki-Trotter product for a Christiansen Greedy Factorized Hamiltonian.

    Args:
        hamiltonian (dict): A CGF Hamiltonian dictionary with keys:
            * ``"nuc_constant"`` (float): Zero-energy constant.
            * ``"core_tensors"`` (array[float]): Shape ``(L+1, M, M, N, N)`` --
              diagonal interaction tensors per fragment, where M is the number of modes, N is the number of modals per mode,
              and L is the number of two-body fragments.
            * ``"leaf_tensors"`` (array[float]): Shape ``(L+1, M, N, N)`` --
              per-mode basis-rotation unitaries per fragment, where M is the number of modes, N is the number of modals per mode,
              and L is the number of two-body fragments.
        time (float): Total evolution time :math:`t`.
        num_steps (int): Number of second-order Trotter steps.
        wires (Wires): System wires, expects ``M*N`` wires in mode-major
            order (wire ``l*N + p`` corresponds to modal ``p`` of mode ``l``).
        control_wires (Wires | None): Optional control wire(s) for controlled
            time evolution (e.g. Hadamard test).

    """

    resource_keys = {"num_fragments", "num_modes", "num_modals", "num_steps", "control_wires"}

    def __init__(self, hamiltonian, time, num_steps, wires=None, control_wires=None, id=None):
        self._hyperparameters = {
            "hamiltonian": hamiltonian,
            "num_steps": num_steps,
            "control_wires": control_wires,
        }

        Z = hamiltonian["core_tensors"]
        U = hamiltonian["leaf_tensors"]

        if not (Z.ndim == 5 and U.ndim == 4):
            raise TypeError(
                "TrotterCGF expects core_tensors with ndim=5 and leaf_tensors with ndim=4. "
                f"Got core_tensors.ndim={Z.ndim}, leaf_tensors.ndim={U.ndim}."
            )

        all_wires = qp.wires.Wires(wires)
        if control_wires is not None:
            all_wires = all_wires + qp.wires.Wires(control_wires)

        super().__init__(time, wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        hamiltonian = self._hyperparameters["hamiltonian"]
        num_fragments = hamiltonian["core_tensors"].shape[0] - 1
        num_modes = hamiltonian["core_tensors"].shape[1]
        num_modals = hamiltonian["core_tensors"].shape[3]
        return {
            "num_fragments": num_fragments,
            "num_modes": num_modes,
            "num_modals": num_modals,
            "num_steps": self._hyperparameters["num_steps"],
            "control_wires": self._hyperparameters["control_wires"],
        }

    def resources(self) -> Resources:
        with QueuingManager.stop_recording():
            decomp = self.compute_decomposition(*self.parameters, wires=self.wires, **self._hyperparameters)

        num_wires = len(self.wires)
        num_gates = len(decomp)

        depth = QuantumScript(decomp).graph.get_depth()

        gate_types = defaultdict(int)
        gate_sizes = defaultdict(int)

        for op in decomp:
            gate_types[op.name] += 1
            gate_sizes[op.name] += len(op.wires)

        return Resources(
            num_wires,
            num_gates,
            gate_types,
            gate_sizes,
            depth)

    @staticmethod
    def compute_decomposition(*args, wires, **kwargs):
        time = args[0]
        num_steps = kwargs["num_steps"]
        hamiltonian = kwargs["hamiltonian"]
        control_wires = kwargs.get("control_wires", None)

        with qp.tape.QuantumTape() as tape:
            trotter_factorized(
                evolution_time=time,
                num_trotter_steps=num_steps,
                hamiltonian=hamiltonian,
                wires=wires,
                control_wires=control_wires,
            )

        return tape.operations

def _trotter_cgf_resources(num_steps, num_fragments, num_modes, num_modals, control_wires):
    """Compute resources for TrotterCGF given the resource parameters."""
    num_basis_rotations = num_modes * ((2 * num_fragments + 1) * num_steps + 1)
    num_zz_gates = num_fragments * num_modes * (num_modes - 1) * num_modals**2 * num_steps
    num_rz_gates = num_modes * num_modals * num_steps
    if control_wires is not None:
        num_rz_gates += 1
        num_cnot_gates = 2 * num_steps * (num_fragments * num_modes * (num_modes - 1) * num_modals**2 + num_modes * num_modals)
        resources = {resource_rep(qp.BasisRotation, dim=num_modals, is_real=True): num_basis_rotations, resource_rep(qp.IsingZZ): num_zz_gates, resource_rep(qp.RZ): num_rz_gates, resource_rep(qp.CNOT): num_cnot_gates}
    else:
        num_cnot_gates = 0
        resources = {resource_rep(qp.BasisRotation, dim=num_modals, is_real=True): num_basis_rotations, resource_rep(qp.IsingZZ): num_zz_gates, resource_rep(qp.RZ): num_rz_gates}

    return resources

@register_resources(_trotter_cgf_resources)
def _trotter_cgf_decomposition(*args, wires, **kwargs):
    time = args[-1]
    n = kwargs["num_steps"]
    hamiltonian = kwargs["hamiltonian"]
    control_wires = kwargs.get("control_wires", None)
    if control_wires is not None:
        system_wires = qp.wires.Wires([w for w in wires if w not in qp.wires.Wires(control_wires)])
    else:
        system_wires = wires

    trotter_factorized(
        evolution_time=time,
        num_trotter_steps=n,
        hamiltonian=hamiltonian,
        wires=system_wires,
        control_wires=control_wires,
    )


add_decomps(TrotterCGF, _trotter_cgf_decomposition)



def trotter_factorized(evolution_time, num_trotter_steps, hamiltonian, wires, control_wires=None):
    r"""Second-order Trotter time evolution for a factorized Hamiltonian.

    This template works for both electronic CDF and vibrational CGF Hamiltonians.

    Args:
        evolution_time (float): Total evolution time ``t``.
        num_trotter_steps (int): Number of second-order Trotter steps.
        hamiltonian (dict): A Hamiltonian in the form of a dictionary with keys ``nuc_constant``, ``core_tensors``, and ``leaf_tensors``.
            * CDF shapes: ``core_tensors: (L+1, N, N)`` (diagonal per fragment),
              ``leaf_tensors: (L+1, N, N)``, where N is the number of orbitals, and L is the number of two-body fragments.
            * CGF shapes: ``core_tensors: (L+1, M, M, N, N)``,
              ``leaf_tensors:  (L+1, M, N, N)``, where M is the number of modes, N is the number of modals per mode, and L is the number of two-body fragments.
        wires (Wires): The system wires. CDF expects ``2N`` wires (alpha / beta interleaved).
            CGF expects ``M*N`` wires arranged mode-major: wire ``l*N + p``
            corresponds to modal ``p`` of mode ``l`` (unary/SBE layout).
        control_wires (Wires | None): Optional control wires.
    """

    Z = hamiltonian["core_tensors"]
    U = hamiltonian["leaf_tensors"]

    if Z.ndim == 3 and U.ndim == 3:
        frag_scheme = "cdf"
    elif Z.ndim == 5 and U.ndim == 4:
        frag_scheme = "cgf"
    else:
        raise ValueError(
            "Could not auto-detect Hamiltonian type. "
            f"Got core_tensors.ndim={Z.ndim}, leaf_tensors.ndim={U.ndim}. "
        )

    if num_trotter_steps > 0:
        second_order_time_step = evolution_time / num_trotter_steps
    else:
        second_order_time_step = 0.0

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx, hamiltonian):
        _trotter_step(
            step_idx, second_order_time_step, hamiltonian, wires, control_wires, frag_scheme
        )
        return hamiltonian

    trotter_steps(hamiltonian)  # pylint: disable=no-value-for-parameter

    if num_trotter_steps > 0:
        U_tensor = hamiltonian["leaf_tensors"]
        very_last_U = _transpose_leaf(U_tensor[1], frag_scheme)
        _apply_system_basis_rotation(very_last_U, wires, frag_scheme)

    # Global phase from the Hamiltonian zero-energy shift is only relevant
    # when the evolution is controlled (Hadamard test).
    if control_wires is not None:
        energy_shift = _energy_shift(hamiltonian, frag_scheme)
        phi = (energy_shift * evolution_time) % (4 * np.pi)
        # exp(-i phi) would be PhaseShift(-phi) controlled. The double-phase
        # trick turns it into RZ(-phi) (differ only by an unobservable global
        # phase, but we keep it exact for bookkeeping).
        qp.RZ(-phi, control_wires)


def _trotter_step(step_idx, second_order_time_step, hamiltonian, wires, control_wires, frag_scheme):
    """Single second-order Trotter step for either CDF or CGF Hamiltonian."""

    if not has_jax:
        raise ImportError(
            "jax is required for trotter_factorized. Install it with: pip install jax jaxlib"
        )

    if qp.compiler.active():
        wires = qp.math.array(wires, like="jax")
        if control_wires is not None:
            control_wires = qp.math.array(control_wires, like="jax")

    U_tensor = hamiltonian["leaf_tensors"]
    Z_tensor = hamiltonian["core_tensors"]

    # U_tensor contains rotations for one-body term and L two-body terms
    # we denote L as ``num_two_body_fragments``.
    num_two_body_fragments = U_tensor.shape[0] - 1

    first_order_time_step = second_order_time_step / 2

    def two_body_fragments(fragment_idx, prev_fragment_idx):
        U = jax.lax.cond(
            prev_fragment_idx < 0,
            lambda U_tensor, fragment_idx, prev_fragment_idx: U_tensor[fragment_idx],
            lambda U_tensor, fragment_idx, prev_fragment_idx: _merge_leaves(
                U_tensor[prev_fragment_idx], U_tensor[fragment_idx], frag_scheme
            ),
            U_tensor,
            fragment_idx,
            prev_fragment_idx,
        )
        Z = Z_tensor[fragment_idx]
        _apply_system_basis_rotation(U, wires, frag_scheme)
        _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, frag_scheme)
        return fragment_idx

    def one_body_fragment():
        U_one = U_tensor[0]
        U = _merge_leaves(U_tensor[num_two_body_fragments], U_one, frag_scheme)
        _apply_system_basis_rotation(U, wires, frag_scheme)
        _apply_one_body_diagonal(
            Z_tensor[0], wires, first_order_time_step, control_wires, frag_scheme
        )

    prev_fragment_idx_forward = qp.math.sign(2 * step_idx - 1)
    qp.for_loop(1, num_two_body_fragments + 1)(two_body_fragments)(prev_fragment_idx_forward)

    one_body_fragment()

    prev_fragment_idx_backward = 0
    qp.for_loop(num_two_body_fragments, 0, -1)(two_body_fragments)(prev_fragment_idx_backward)


def _apply_system_basis_rotation(U, wires, frag_scheme):
    """Apply a fragment's basis rotation on the whole system."""
    if frag_scheme == "cdf":
        if qp.math.is_abstract(U):
            qp.BasisRotation(unitary_matrix=U, wires=wires[::2])
            qp.BasisRotation(unitary_matrix=U, wires=wires[1::2])
        elif not np.allclose(U, np.eye(len(U))):
            qp.BasisRotation(unitary_matrix=U, wires=wires[::2])
            qp.BasisRotation(unitary_matrix=U, wires=wires[1::2])
    else:
        num_modes, n_states, _ = U.shape
        # The fragment's leaf O stores the "rotate-from-bare-to-diagonal-basis"
        # direction.  The circuit needs to rotate the qubits so that the
        # diagonal layer acts in the correct basis: qml.BasisRotation(U) with
        # U = O^T implements the single-particle map O^T (moving from bare
        # modal states to the diagonal basis).  Hence we pass the transpose.
        for l in range(num_modes):
            U_l = qp.math.swapaxes(U[l], -2, -1)
            mode_wires = wires[l * n_states : (l + 1) * n_states]
            if qp.math.is_abstract(U_l):
                qp.BasisRotation(unitary_matrix=U_l, wires=mode_wires)
            elif not np.allclose(U_l, np.eye(n_states)):
                qp.BasisRotation(unitary_matrix=U_l, wires=mode_wires)


def _merge_leaves(U_prev, U_curr, frag_scheme):
    """Fragment-rotation merge rule: ``U_prev^dagger @ U_curr``.

    CDF: a single ``(N, N)`` matmul.
    CGF: per-mode matmul, vectorized via ``einsum``.
    """
    if frag_scheme == "cdf":
        return U_prev.T @ U_curr
    return qp.math.einsum("lji,ljk->lik", U_prev, U_curr)


def _transpose_leaf(U, frag_scheme):
    """Transpose (i.e. adjoint for real orthogonal rotations) of a leaf."""
    if frag_scheme == "cdf":
        return U.T
    # CGF: batch-transpose over the leading mode axis.
    return qp.math.swapaxes(U, -2, -1)


def _apply_two_body_diagonal(Z, wires, first_order_time_step, control_wires, frag_scheme):
    if frag_scheme == "cdf":
        num_cas = Z.shape[0]

        @qp.for_loop(2 * num_cas - 1)
        def zz_rotations(wire_idx0):

            @qp.for_loop(wire_idx0 + 1, 2 * num_cas)
            def _zz_rotations(wire_idx1):
                # Prefactor breakdown:
                #   1/8  from (A29)
                #   2    from exp(-i H t) -> IsingZZ(phi) = exp(-i phi Z Z / 2)
                #   -1/2 from the double-phase trick
                #   2    from symmetrization (k<->l for sigma=tau,
                #                             (k,sigma)<->(l,tau) for sigma!=tau)
                #   => -1/4
                angle = -0.25 * Z[wire_idx0 // 2, wire_idx1 // 2] * first_order_time_step
                qp.IsingZZ(angle, [wires[wire_idx0], wires[wire_idx1]])

            # TODO: support multiple control wires (not needed for Hadamard tests)
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx0]])
            _zz_rotations()
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx0]])

        zz_rotations()
    else:
        num_modes = Z.shape[0]
        n_states = Z.shape[2]

        ising_coeff = 0.5

        for l in range(1, num_modes):
            for m in range(l):  # strict lower triangle: l > m
                Z_lm = Z[l, m]

                @qp.for_loop(n_states)
                def _p_loop(p, Z_lm=Z_lm, l=l, m=m):
                    wire_lp = wires[l * n_states + p]

                    @qp.for_loop(n_states)
                    def _q_loop(q, Z_lm=Z_lm, p=p, wire_lp=wire_lp, l=l, m=m):
                        wire_mq = wires[m * n_states + q]
                        lam = Z_lm[p, q]
                        angle = ising_coeff * lam * first_order_time_step
                        if control_wires is not None:
                            qp.CNOT([control_wires[0], wire_lp])
                        qp.IsingZZ(angle, [wire_lp, wire_mq])
                        if control_wires is not None:
                            qp.CNOT([control_wires[0], wire_lp])

                    _q_loop()

                _p_loop()


def _apply_one_body_diagonal(Z_one_body, wires, first_order_time_step, control_wires, frag_scheme):
    if frag_scheme == "cdf":
        num_cas = Z_one_body.shape[0]

        @qp.for_loop(2 * num_cas)
        def z_rotations(wire_idx):
            # Prefactor breakdown:
            #   -1/2 from (A29)
            #   2    from exp(-i H t) -> RZ(phi) = exp(-i phi Z / 2)
            #   -1/2 from the double-phase trick
            #   2    from merging forward+backward occurrences of the 2nd-order
            #        Trotter formula
            #   => 1
            angle = Z_one_body[wire_idx // 2, wire_idx // 2] * first_order_time_step

            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx]])
            qp.RZ(angle, wires[wire_idx])
            if control_wires is not None:
                qp.CNOT([control_wires[0], wires[wire_idx]])

        z_rotations()
    else:
        num_modes = Z_one_body.shape[0]
        n_states = Z_one_body.shape[2]

        @qp.for_loop(num_modes)
        def mode_loop(l):

            @qp.for_loop(n_states)
            def modal_loop(p):
                wire_lp = wires[l * n_states + p]
                # One-body prefactor derivation:
                #   n^l_p = (I - Z_{l,p}) / 2  ->  Z-piece has coefficient -eps/2
                #   target operator: exp(-i eps n t) has Z-piece exp(+i eps t / 2 Z)
                #     which equals RZ(-eps t).
                #   This fragment is visited ONCE per Trotter step, at duration
                #     first_order_time_step = dt_trotter / 2, so to accumulate a
                #     total RZ angle of -eps * dt_trotter we need angle-per-step
                #     = -eps * dt_trotter = -2 * eps * first_order_time_step.
                # So alpha_oneB = -2.
                angle = -2.0 * Z_one_body[l, l, p, p] * first_order_time_step

                if control_wires is not None:
                    qp.CNOT([control_wires[0], wire_lp])
                qp.RZ(angle, wire_lp)
                if control_wires is not None:
                    qp.CNOT([control_wires[0], wire_lp])

            modal_loop()

        mode_loop()


# Zero-energy shift (global phase on controlled evolution)


def _energy_shift(hamiltonian, frag_scheme):
    """Compute the zero-of-energy shift that must be applied as a controlled
    ``RZ`` during a Hadamard test."""
    nuc_constant = hamiltonian.get("nuc_constant", 0.0)
    Z_tensor = hamiltonian["core_tensors"]

    if frag_scheme == "cdf":
        # Eq. (A29) first line: nuc + sum_k Z^(0)_{k,k}
        #   - (sum_{l,k,l'} Z^(l)_{k,l'}) / 2
        #   + (sum_{l,k} Z^(l)_{k,k}) / 4
        phase_from_mod_one_body = qp.math.trace(Z_tensor[0])
        phase_from_two_body = (
            -qp.math.sum(Z_tensor[1:]) / 2
            + qp.math.sum(qp.math.trace(Z_tensor[1:], axis1=1, axis2=2)) / 4
        )
        return nuc_constant + phase_from_mod_one_body + phase_from_two_body

    # One-body diagonal: nested traces over axes (-2, -1) twice pick out the
    # (l == m, p == q) entries and sum ε^l_p.
    one_body_diag = qp.math.trace(
        qp.math.trace(Z_tensor[0], axis1=-2, axis2=-1),
        axis1=-2,
        axis2=-1,
    )  # scalar: Σ_{l, p} ε^l_p
    phase_from_one_body = one_body_diag / 2

    return nuc_constant + phase_from_one_body
