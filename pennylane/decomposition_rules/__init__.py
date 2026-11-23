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

r"""Built-in decomposition rules, organized like their operator and template modules.

The modules directly beneath this package mirror :mod:`pennylane.ops`, and the
``templates`` branch mirrors :mod:`pennylane.templates`.
The drawing-specific ``LabelledOp`` rule remains in :mod:`pennylane.drawer.label`.
Rules are loaded automatically when PennyLane is imported.
Use :func:`pennylane.list_decomps` to retrieve the registered rules for an operator.
Unambiguous rules, helpers, and factories can also be imported from this package.
Symbolic implementations with the same name remain in their respective modules.

.. currentmodule:: pennylane.decomposition_rules

.. autosummary::

    qubit
    op_math
    identity
    meta
    common
    templates
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import (
        null_decomp,
    )
    from .identity import (
        _controlled_g_phase_standard,
    )
    from .op_math.adjoint import (
        make_adjoint_decomp,
    )
    from .op_math.adjoint2 import (
        _make_adjoint_decomp,
    )
    from .op_math.controlled import (
        make_controlled_decomp,
    )
    from .op_math.controlled2 import (
        _make_controlled_decomp,
        ctrl_many_zeroed_work_wires,
    )
    from .op_math.controlled_ops import (
        _toffoli_elbow,
    )
    from .op_math.controlled_ops import (
        _mcx_many_work_wires,
        _mcx_no_work_wires,
        _mcx_one_work_wire,
        _mcx_two_work_wires,
    )
    from .op_math.symbolicop import (
        self_adjoint_legacy,
    )
    from .op_math.symbolicop2 import (
        self_adjoint,
    )
    from .qubit.non_parametric_ops import (
        _PPRBasis,
        _controlled_hadamard_standard,
        _controlled_z_standard,
        _controlled_z_to_ccz,
        _ctrl_x_to_mcx,
    )
    from .qubit.parametric_ops_multi_qubit import (
        _IsingXYBasis,
        _IsingXYRotations,
        _MultiRZLadder,
        _PauliRotBasis,
    )
    from .qubit.parametric_ops_single_qubit import (
        _YToZBasis,
        _ZToYBasis,
        _controlled_phase_shift_standard,
        _controlled_rot_standard,
        _controlled_rx_standard,
    )
    from .qubit.qchem_ops import (
        _SingleExcitationBasis,
        _SingleExcitationCRZBasis,
        _SingleExcitationRZRotations,
        _SingleExcitationRotations,
    )
    from .templates.state_preparations.mottonen import (
        mottonen_decomp,
    )
    from .templates.state_preparations.partial_unary import (
        _find_affine_subspace_isometry,
        _is_affine_subspace,
        _partial_unary_dense_state,
        _partial_unary_state_prep_core,
        _partial_unary_state_prep_standard_resources,
    )
    from .templates.state_preparations.phase_gradient import (
        _phase_gradient_state_prep_to_clifford_t,
        _phase_gradient_state_prep_to_ppr,
    )
    from .templates.state_preparations.state_prep_mps import (
        _mps_prep_standard,
    )
    from .templates.state_preparations.sum_of_slaters import (
        _sos_state_prep_standard,
        _sos_state_prep_with_wires,
    )
    from .templates.subroutines.alias_sampling import (
        _build_alias_tables,
    )
    from .templates.subroutines.alias_sampling_thc import (
        _build_qrom_data,
        _compute_contiguous_register,
        _cswap_pair,
        _lcu_signs,
        _symmetrize,
    )
    from .templates.subroutines.arithmetic.adder import (
        _adder_arithmetic,
    )
    from .templates.subroutines.arithmetic.incrementer import (
        _adjoint_incrementer_bit_flips,
        _decompose_mcxs,
    )
    from .templates.subroutines.arithmetic.left_classical_comparator import (
        _left_classical_comparator_standard,
    )
    from .templates.subroutines.arithmetic.out_adder import (
        _OutAdderTarget,
    )
    from .templates.subroutines.arithmetic.out_multiplier import (
        _OutMultiplierHadamards,
        _add_plus_one,
        _caddsub_multiplication,
        _out_multiplier_with_cache_resources,
        _out_multiplier_with_caddsub,
        _out_multiplier_with_qft,
    )
    from .templates.subroutines.arithmetic.out_square import (
        _out_square_with_adder_zeroed,
        _out_square_with_caddsub,
    )
    from .templates.subroutines.arithmetic.phase_adder import (
        _PhaseAdderBasis,
        _add_k_fourier,
    )
    from .templates.subroutines.arithmetic.semi_adder import (
        _controlled_semi_adder_standard,
        _semi_adder_standard,
        _semi_adder_standard_resources,
    )
    from .templates.subroutines.arithmetic.signed_out_multiplier import (
        _signed_out_multiplier_borrowed_resources,
        _signed_out_multiplier_zeroed_resources,
        _twos_complement_helper,
    )
    from .templates.subroutines.arithmetic.signed_out_square import (
        _signed_out_square_from_unsigned_resources,
    )
    from .templates.subroutines.arithmetic.temporary_and import (
        _TemporaryANDBasis,
        _adjoint_temporary_and_measurement,
        _adjoint_temporary_and_to_ppm,
        _adjoint_temporary_and_to_toffoli,
    )
    from .templates.subroutines.ffft import (
        _permute_and_apply_parallel,
    )
    from .templates.subroutines.multix import (
        _controlled_multix_ladder_standard,
        _controlled_multix_ladder_standard_work_wires,
        _multix_ladder_fanout,
        _multix_standard,
    )
    from .templates.subroutines.qchem.basis_rotation import (
        _basis_rotation_real,
        _basis_rotation_real_resources,
    )
    from .templates.subroutines.qrom import (
        _calculate_select_swap_sizes,
        _count_tempAND_in_measurement_qrom,
        _qrom_measurement,
        _qrom_measurement_condition,
        _qrom_measurement_resources,
        _qrom_select_swap,
        _qrom_unary_iteration,
    )
    from .templates.subroutines.qubitization_thc import (
        _THCPrepare,
    )
    from .templates.subroutines.select import (
        _partial_select,
        _select_multi_control_work_wire,
        _select_unary,
    )
    from .templates.subroutines.select_pauli_rot import (
        _SelectPauliRotYBasis,
        _UniformRZ,
        _select_pauli_rot_standard,
    )
    from .templates.subroutines.select_thc import (
        _SelectTHCBasis,
        _THCIndexSwaps,
        _apply_loaded_rotation,
        _build_qrom_givens_data,
        _cascade_angles,
        _select_half,
    )
    from .templates.subroutines.time_evolution.trotter_cdf import (
        _apply_two_body_diagonal_with_work_wires,
        _cdf_resource_counts,
        _controlled_trotter_cdf_standard,
        _trotter_cdf_multi_control,
        _trotter_cdf_standard,
    )
    from .templates.subroutines.time_evolution.trotter_cgf import (
        _cgf_resource_counts,
        _controlled_trotter_cgf_standard,
        _trotter_cgf_standard,
    )
    from .templates.subroutines.time_evolution.trotter_vibronic import (
        _diagonalize_probs,
        _diagonalize_vibronic_circuit,
        _half_signed_out_multiplier,
        _num_applied_terms,
        _validate_registers,
    )
    from .op_math.controlled_ops import _CRZPhaseGradientCompute, make_crz_to_phase_gradient_decomp
    from .templates.state_preparations.phase_gradient import (
        _jones_num_rounds,
        _jones_register_sizes,
        _jones_seed,
        _jones_success_probabilities,
        make_phase_gradient_distillation_decomp,
    )
    from .qubit.parametric_ops_single_qubit import (
        make_rz_to_phase_gradient_decomp,
        phase_gradient_precision_stats,
        validate_phase_gradient_wires,
    )
    from .templates.subroutines.select_pauli_rot import (
        _SelectPauliRotPhaseGradientCompute,
        _select_pauli_rot_phase_gradient,
        make_selectpaulirot_to_phase_gradient_decomp,
    )

__all__ = [
    "_CRZPhaseGradientCompute",
    "_IsingXYBasis",
    "_IsingXYRotations",
    "_MultiRZLadder",
    "_OutAdderTarget",
    "_OutMultiplierHadamards",
    "_PPRBasis",
    "_PauliRotBasis",
    "_PhaseAdderBasis",
    "_SelectPauliRotPhaseGradientCompute",
    "_SelectPauliRotYBasis",
    "_SelectTHCBasis",
    "_SingleExcitationBasis",
    "_SingleExcitationCRZBasis",
    "_SingleExcitationRZRotations",
    "_SingleExcitationRotations",
    "_THCIndexSwaps",
    "_THCPrepare",
    "_TemporaryANDBasis",
    "_UniformRZ",
    "_YToZBasis",
    "_ZToYBasis",
    "_add_k_fourier",
    "_add_plus_one",
    "_adder_arithmetic",
    "_adjoint_incrementer_bit_flips",
    "_adjoint_temporary_and_measurement",
    "_adjoint_temporary_and_to_ppm",
    "_adjoint_temporary_and_to_toffoli",
    "_apply_loaded_rotation",
    "_apply_two_body_diagonal_with_work_wires",
    "_basis_rotation_real",
    "_basis_rotation_real_resources",
    "_build_alias_tables",
    "_build_qrom_data",
    "_build_qrom_givens_data",
    "_caddsub_multiplication",
    "_calculate_select_swap_sizes",
    "_cascade_angles",
    "_cdf_resource_counts",
    "_cgf_resource_counts",
    "_compute_contiguous_register",
    "_controlled_g_phase_standard",
    "_controlled_hadamard_standard",
    "_controlled_multix_ladder_standard",
    "_controlled_multix_ladder_standard_work_wires",
    "_controlled_phase_shift_standard",
    "_controlled_rot_standard",
    "_controlled_rx_standard",
    "_controlled_semi_adder_standard",
    "_controlled_trotter_cdf_standard",
    "_controlled_trotter_cgf_standard",
    "_controlled_z_standard",
    "_controlled_z_to_ccz",
    "_count_tempAND_in_measurement_qrom",
    "_cswap_pair",
    "_ctrl_x_to_mcx",
    "_decompose_mcxs",
    "_diagonalize_probs",
    "_diagonalize_vibronic_circuit",
    "_find_affine_subspace_isometry",
    "_half_signed_out_multiplier",
    "_is_affine_subspace",
    "_jones_num_rounds",
    "_jones_register_sizes",
    "_jones_seed",
    "_jones_success_probabilities",
    "_lcu_signs",
    "_left_classical_comparator_standard",
    "_make_adjoint_decomp",
    "_make_controlled_decomp",
    "_mcx_many_work_wires",
    "_mcx_no_work_wires",
    "_mcx_one_work_wire",
    "_mcx_two_work_wires",
    "_mps_prep_standard",
    "_multix_ladder_fanout",
    "_multix_standard",
    "_num_applied_terms",
    "_out_multiplier_with_cache_resources",
    "_out_multiplier_with_caddsub",
    "_out_multiplier_with_qft",
    "_out_square_with_adder_zeroed",
    "_out_square_with_caddsub",
    "_partial_select",
    "_partial_unary_dense_state",
    "_partial_unary_state_prep_core",
    "_partial_unary_state_prep_standard_resources",
    "_permute_and_apply_parallel",
    "_phase_gradient_state_prep_to_clifford_t",
    "_phase_gradient_state_prep_to_ppr",
    "_qrom_measurement",
    "_qrom_measurement_condition",
    "_qrom_measurement_resources",
    "_qrom_select_swap",
    "_qrom_unary_iteration",
    "_select_half",
    "_select_multi_control_work_wire",
    "_select_pauli_rot_phase_gradient",
    "_select_pauli_rot_standard",
    "_select_unary",
    "_semi_adder_standard",
    "_semi_adder_standard_resources",
    "_signed_out_multiplier_borrowed_resources",
    "_signed_out_multiplier_zeroed_resources",
    "_signed_out_square_from_unsigned_resources",
    "_sos_state_prep_standard",
    "_sos_state_prep_with_wires",
    "_symmetrize",
    "_toffoli_elbow",
    "_trotter_cdf_multi_control",
    "_trotter_cdf_standard",
    "_trotter_cgf_standard",
    "_twos_complement_helper",
    "_validate_registers",
    "ctrl_many_zeroed_work_wires",
    "make_adjoint_decomp",
    "make_controlled_decomp",
    "make_crz_to_phase_gradient_decomp",
    "make_phase_gradient_distillation_decomp",
    "make_rz_to_phase_gradient_decomp",
    "make_selectpaulirot_to_phase_gradient_decomp",
    "mottonen_decomp",
    "null_decomp",
    "phase_gradient_precision_stats",
    "self_adjoint",
    "self_adjoint_legacy",
    "validate_phase_gradient_wires",
]

_BUILTIN_MODULES = (
    "pennylane.decomposition_rules.identity",
    "pennylane.decomposition_rules.op_math.change_op_basis2",
    "pennylane.decomposition_rules.meta",
    "pennylane.decomposition_rules.qubit.arithmetic_ops",
    "pennylane.decomposition_rules.qubit.matrix_ops",
    "pennylane.decomposition_rules.qubit.non_parametric_ops",
    "pennylane.decomposition_rules.qubit.parametric_ops_single_qubit",
    "pennylane.decomposition_rules.qubit.parametric_ops_multi_qubit",
    "pennylane.decomposition_rules.qubit.qchem_ops",
    "pennylane.decomposition_rules.qubit.special_unitary",
    "pennylane.decomposition_rules.templates.state_preparations.arbitrary_state_preparation",
    "pennylane.decomposition_rules.templates.state_preparations.cosine_window",
    "pennylane.decomposition_rules.templates.state_preparations.mottonen",
    "pennylane.decomposition_rules.templates.state_preparations.superposition",
    "pennylane.decomposition_rules.templates.state_preparations.state_prep_mps",
    "pennylane.decomposition_rules.templates.state_preparations.multiplexer_state_prep",
    "pennylane.decomposition_rules.templates.state_preparations.sum_of_slaters",
    "pennylane.decomposition_rules.templates.state_preparations.partial_unary",
    "pennylane.decomposition_rules.templates.state_preparations.phase_gradient",
    "pennylane.decomposition_rules.qubit.state_preparation",
    "pennylane.decomposition_rules.op_math.controlled_ops",
    "pennylane.decomposition_rules.op_math.exp",
    "pennylane.decomposition_rules.op_math.evolution",
    "pennylane.decomposition_rules.op_math.prod2",
    "pennylane.decomposition_rules.op_math.prod",
    "pennylane.decomposition_rules.op_math.change_op_basis",
    "pennylane.decomposition_rules.templates.core",
    "pennylane.decomposition_rules.templates.embeddings.amplitude",
    "pennylane.decomposition_rules.templates.embeddings.angle",
    "pennylane.decomposition_rules.templates.embeddings.iqp",
    "pennylane.decomposition_rules.templates.embeddings.qaoaembedding",
    "pennylane.decomposition_rules.templates.subroutines.arbitrary_unitary",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.approx_time_evolution",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.commuting_evolution",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.trotter",
    "pennylane.decomposition_rules.templates.subroutines.qft",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.phase_adder",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.adder",
    "pennylane.decomposition_rules.templates.subroutines.controlled_sequence",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.multiplier",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.mod_exp",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.out_adder",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.temporary_and",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.semi_adder",
    "pennylane.decomposition_rules.templates.subroutines.multix",
    "pennylane.decomposition_rules.templates.subroutines.qrom",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.incrementer",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.out_multiplier",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.signed_out_multiplier",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.out_poly",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.out_square",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.signed_out_square",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.left_classical_comparator",
    "pennylane.decomposition_rules.templates.subroutines.arithmetic.left_quantum_comparator",
    "pennylane.decomposition_rules.templates.subroutines.qchem.all_singles_doubles",
    "pennylane.decomposition_rules.templates.subroutines.qchem.fermionic_double_excitation",
    "pennylane.decomposition_rules.templates.subroutines.qchem.fermionic_single_excitation",
    "pennylane.decomposition_rules.templates.subroutines.qchem.kupccgsd",
    "pennylane.decomposition_rules.templates.subroutines.qchem.uccsd",
    "pennylane.decomposition_rules.templates.subroutines.qchem.basis_rotation",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.trotter_cdf",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.trotter_cgf",
    "pennylane.decomposition_rules.templates.subroutines.aqft",
    "pennylane.decomposition_rules.templates.subroutines.time_evolution.trotter_vibronic",
    "pennylane.decomposition_rules.templates.subroutines.permute",
    "pennylane.decomposition_rules.templates.subroutines.qpe",
    "pennylane.decomposition_rules.templates.subroutines.qmc",
    "pennylane.decomposition_rules.templates.subroutines.grover",
    "pennylane.decomposition_rules.templates.subroutines.hilbert_schmidt",
    "pennylane.decomposition_rules.templates.subroutines.flip_sign",
    "pennylane.decomposition_rules.templates.subroutines.fable",
    "pennylane.decomposition_rules.templates.subroutines.ffft",
    "pennylane.decomposition_rules.templates.subroutines.select",
    "pennylane.decomposition_rules.templates.subroutines.prepselprep",
    "pennylane.decomposition_rules.templates.subroutines.reflection",
    "pennylane.decomposition_rules.templates.subroutines.qubitization",
    "pennylane.decomposition_rules.templates.subroutines.amplitude_amplification",
    "pennylane.decomposition_rules.templates.subroutines.qram",
    "pennylane.decomposition_rules.templates.subroutines.iqp",
    "pennylane.decomposition_rules.templates.subroutines.alias_sampling",
    "pennylane.decomposition_rules.templates.subroutines.one_body_block_encoding",
    "pennylane.decomposition_rules.templates.subroutines.alias_sampling_thc",
    "pennylane.decomposition_rules.templates.subroutines.select_thc",
    "pennylane.decomposition_rules.templates.subroutines.superposition_thc",
    "pennylane.decomposition_rules.templates.subroutines.qubitization_thc",
    "pennylane.decomposition_rules.templates.subroutines.gqsp",
    "pennylane.decomposition_rules.templates.subroutines.select_pauli_rot",
    "pennylane.decomposition_rules.templates.subroutines.qsvt",
    "pennylane.decomposition_rules.templates.layers.basic_entangler",
    "pennylane.decomposition_rules.templates.layers.gate_fabric",
    "pennylane.decomposition_rules.templates.layers.particle_conserving_u1",
    "pennylane.decomposition_rules.templates.layers.particle_conserving_u2",
    "pennylane.decomposition_rules.templates.layers.simplified_two_design",
    "pennylane.decomposition_rules.templates.layers.strongly_entangling",
    "pennylane.decomposition_rules.op_math.adjoint2",
    "pennylane.decomposition_rules.op_math.controlled2",
    "pennylane.decomposition_rules.op_math.pow2",
    "pennylane.decomposition_rules.op_math.adjoint",
    "pennylane.decomposition_rules.op_math.pow",
    "pennylane.decomposition_rules.op_math.symbolicop",
    "pennylane.decomposition_rules.op_math.symbolicop2",
    "pennylane.decomposition_rules.op_math.controlled",
    "pennylane.decomposition_rules.common",
)


def _load_builtin_rules():
    """Load the built-in registrations after the operator and template definitions."""
    modules = []
    for module_name in _BUILTIN_MODULES:
        if module_name == "pennylane.decomposition_rules.op_math.adjoint2":
            # Phase-gradient helpers were historically registered after the template rules.
            for helper_module in (
                "pennylane.decomposition_rules.templates.subroutines.select_pauli_rot",
                "pennylane.decomposition_rules.op_math.controlled_ops",
            ):
                import_module(helper_module).register_phase_gradient_compute()
        modules.append(import_module(module_name))
    for module in modules:
        module_name = module.__name__
        for name in __all__:
            value = vars(module).get(name)
            if getattr(value, "__module__", None) == module_name:
                globals()[name] = value
