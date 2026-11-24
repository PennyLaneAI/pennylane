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
Phase-gradient rules and shared helpers are grouped in :mod:`pennylane.decomposition_rules.phase_gradient`.
The drawing-specific ``LabelledOp`` rule remains in :mod:`pennylane.drawer.label`.
Rules are loaded automatically when PennyLane is imported.
Use :func:`pennylane.list_decomps` to retrieve the registered rules for an operator.
Unambiguous rules, helpers, and factories can also be imported from this package.
Operator2 symbolic rules and factories are exported from ``op_math``.
Legacy symbolic rules remain in their individual modules.

.. currentmodule:: pennylane.decomposition_rules

.. autosummary::

    qubit
    op_math
    identity
    meta
    common
    templates
    phase_gradient
"""

from .identity import controlled_g_phase_standard
from .op_math import change_op_basis2
from . import meta
from .qubit import arithmetic_ops
from .qubit import matrix_ops
from .qubit.non_parametric_ops import (
    _PPRBasis,
    controlled_hadamard_standard,
    controlled_z_standard,
    controlled_z_to_ccz,
    ctrl_x_to_mcx,
)
from .qubit.parametric_ops_single_qubit import (
    _YToZBasis,
    _ZToYBasis,
    controlled_phase_shift_standard,
    controlled_rot_standard,
    controlled_rx_standard,
)
from .qubit.parametric_ops_multi_qubit import (
    _IsingXYBasis,
    _IsingXYRotations,
    _MultiRZLadder,
    _PauliRotBasis,
)
from .qubit.qchem_ops import (
    _SingleExcitationBasis,
    _SingleExcitationCRZBasis,
    _SingleExcitationRZRotations,
    _SingleExcitationRotations,
)
from .qubit import special_unitary
from .templates.state_preparations import arbitrary_state_preparation
from .templates.state_preparations import cosine_window
from .templates.state_preparations.mottonen import mottonen_decomp
from .templates.state_preparations import superposition
from .templates.state_preparations.state_prep_mps import mps_prep_standard
from .templates.state_preparations import multiplexer_state_prep
from .templates.state_preparations.sum_of_slaters import (
    sos_state_prep_standard,
    _sos_state_prep_with_wires,
)
from .templates.state_preparations.partial_unary import (
    _find_affine_subspace_isometry,
    _is_affine_subspace,
    _partial_unary_dense_state,
    _partial_unary_state_prep_core,
    _partial_unary_state_prep_standard_resources,
)
from .qubit import state_preparation
from .op_math.controlled_ops import (
    ctrl_decomp_bisect,
    ctrl_decomp_zyz,
    toffoli_elbow,
    mcx_many_work_wires,
    mcx_no_work_wires,
    mcx_one_work_wire,
    mcx_two_work_wires,
)
from .op_math import exp
from .op_math import evolution
from .op_math import prod2
from .op_math import prod
from .op_math import change_op_basis
from .templates import core
from .templates.embeddings import amplitude
from .templates.embeddings import angle
from .templates.embeddings import iqp
from .templates.embeddings import qaoaembedding
from .templates.subroutines import arbitrary_unitary
from .templates.subroutines.time_evolution import approx_time_evolution
from .templates.subroutines.time_evolution import commuting_evolution
from .templates.subroutines.time_evolution import trotter
from .templates.subroutines import qft
from .templates.subroutines.arithmetic.phase_adder import _PhaseAdderBasis, _add_k_fourier
from .templates.subroutines.arithmetic.adder import adder_arithmetic
from .templates.subroutines import controlled_sequence
from .templates.subroutines.arithmetic import multiplier
from .templates.subroutines.arithmetic import mod_exp
from .templates.subroutines.arithmetic.out_adder import _OutAdderTarget
from .templates.subroutines.arithmetic.temporary_and import (
    _TemporaryANDBasis,
    adjoint_temporary_and_measurement,
    adjoint_temporary_and_to_ppm,
    adjoint_temporary_and_to_toffoli,
)
from .templates.subroutines.arithmetic.semi_adder import (
    controlled_semi_adder_standard,
    semi_adder_standard,
    _semi_adder_standard_resources,
)
from .templates.subroutines.multix import (
    controlled_multix_ladder_standard,
    controlled_multix_lattice_surgery_ppm,
    _controlled_multix_ladder_standard_work_wires,
    _multix_ladder_fanout,
    multix_standard,
)
from .templates.subroutines.qrom import (
    _count_tempAND_in_measurement_qrom,
    qrom_measurement,
    _qrom_measurement_condition,
    _qrom_measurement_resources,
    qrom_unary_iteration,
)
from .templates.subroutines.qrom_load import (
    _calculate_select_swap_sizes,
    adjoint_qrom_load_measurement,
    qrom_load_select_swap,
    qrom_load_to_qrom,
)
from .templates.subroutines.select_swap import adjoint_select_swap_measurement, select_swap
from .templates.subroutines.arithmetic.incrementer import (
    adjoint_incrementer_bit_flips,
    _decompose_mcxs,
)
from .templates.subroutines.arithmetic.out_multiplier import (
    _OutMultiplierHadamards,
    _add_plus_one,
    _caddsub_multiplication,
    _out_multiplier_with_cache_resources,
    out_multiplier_with_caddsub,
    out_multiplier_with_qft,
)
from .templates.subroutines.arithmetic.signed_out_multiplier import (
    _signed_out_multiplier_borrowed_resources,
    _signed_out_multiplier_zeroed_resources,
    _twos_complement_helper,
)
from .templates.subroutines.arithmetic import out_poly
from .templates.subroutines.arithmetic.out_square import (
    out_square_with_adder_zeroed,
    out_square_with_caddsub,
)
from .templates.subroutines.arithmetic.signed_out_square import (
    _signed_out_square_from_unsigned_resources,
)
from .templates.subroutines.arithmetic.left_classical_comparator import (
    left_classical_comparator_standard,
)
from .templates.subroutines.arithmetic import left_quantum_comparator
from .templates.subroutines.qchem import all_singles_doubles
from .templates.subroutines.qchem import fermionic_double_excitation
from .templates.subroutines.qchem import fermionic_single_excitation
from .templates.subroutines.qchem import kupccgsd
from .templates.subroutines.qchem import uccsd
from .templates.subroutines.qchem.basis_rotation import (
    basis_rotation_real,
    _basis_rotation_real_resources,
)
from .templates.subroutines.time_evolution.trotter_cdf import (
    _apply_two_body_diagonal_with_work_wires,
    _cdf_resource_counts,
    controlled_trotter_cdf_standard,
    trotter_cdf_multi_control,
    trotter_cdf_standard,
)
from .templates.subroutines.time_evolution.trotter_cgf import (
    _cgf_resource_counts,
    controlled_trotter_cgf_standard,
    trotter_cgf_standard,
)
from .templates.subroutines import aqft
from .templates.subroutines.time_evolution.trotter_vibronic import (
    _diagonalize_probs,
    _diagonalize_vibronic_circuit,
    _half_signed_out_multiplier,
    _num_applied_terms,
    _validate_registers,
)
from .templates.subroutines import permute
from .templates.subroutines import qpe
from .templates.subroutines import qmc
from .templates.subroutines import grover
from .templates.subroutines import hilbert_schmidt
from .templates.subroutines import flip_sign
from .templates.subroutines import fable
from .templates.subroutines.ffft import _permute_and_apply_parallel
from .templates.subroutines.select import (
    _partial_select,
    select_multi_control_work_wire,
    select_unary,
)
from .templates.subroutines import prepselprep
from .templates.subroutines import reflection
from .templates.subroutines import qubitization
from .templates.subroutines import amplitude_amplification
from .templates.subroutines import qram
from .templates.subroutines import iqp
from .templates.subroutines.alias_sampling import _build_alias_tables
from .templates.subroutines import one_body_block_encoding
from .templates.subroutines.alias_sampling_thc import (
    _build_qrom_data,
    _compute_contiguous_register,
    _cswap_pair,
    _lcu_signs,
    _symmetrize,
)
from .templates.subroutines.select_thc import (
    _SelectTHCBasis,
    _THCIndexSwaps,
    _apply_loaded_rotation,
    _build_qrom_givens_data,
    _cascade_angles,
    _select_half,
)
from .templates.subroutines import superposition_thc
from .templates.subroutines.qubitization_thc import _THCPrepare
from .templates.subroutines import gqsp
from .templates.subroutines.select_pauli_rot import (
    _SelectPauliRotYBasis,
    _UniformRZ,
    select_pauli_rot_standard,
)
from .templates.subroutines import qsvt
from .templates.layers import basic_entangler
from .templates.layers import gate_fabric
from .templates.layers import particle_conserving_u1
from .templates.layers import particle_conserving_u2
from .templates.layers import simplified_two_design
from .templates.layers import strongly_entangling
from .phase_gradient import (
    phase_gradient_state_prep_to_clifford_t,
    phase_gradient_state_prep_to_ppr,
    make_phase_gradient_distillation_decomp,
    _CRZPhaseGradientCompute,
    make_crz_to_phase_gradient_decomp,
    make_rz_to_phase_gradient_decomp,
    phase_gradient_precision_stats,
    validate_phase_gradient_wires,
    _SelectPauliRotPhaseGradientCompute,
    make_selectpaulirot_to_phase_gradient_decomp,
)
from .phase_gradient.state_preparation import (
    _jones_num_rounds,
    _jones_register_sizes,
    _jones_seed,
    _jones_success_probabilities,
)
from .phase_gradient.select_pauli_rot import _select_pauli_rot_phase_gradient
from .op_math import pow
from .common import null_decomp

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
    "adder_arithmetic",
    "adjoint_incrementer_bit_flips",
    "adjoint_temporary_and_measurement",
    "adjoint_temporary_and_to_ppm",
    "adjoint_temporary_and_to_toffoli",
    "_apply_loaded_rotation",
    "_apply_two_body_diagonal_with_work_wires",
    "basis_rotation_real",
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
    "controlled_g_phase_standard",
    "controlled_hadamard_standard",
    "controlled_multix_ladder_standard",
    "controlled_multix_lattice_surgery_ppm",
    "_controlled_multix_ladder_standard_work_wires",
    "controlled_phase_shift_standard",
    "controlled_rot_standard",
    "controlled_rx_standard",
    "controlled_semi_adder_standard",
    "controlled_trotter_cdf_standard",
    "controlled_trotter_cgf_standard",
    "controlled_z_standard",
    "controlled_z_to_ccz",
    "_count_tempAND_in_measurement_qrom",
    "_cswap_pair",
    "ctrl_decomp_bisect",
    "ctrl_decomp_zyz",
    "ctrl_x_to_mcx",
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
    "left_classical_comparator_standard",
    "mcx_many_work_wires",
    "mcx_no_work_wires",
    "mcx_one_work_wire",
    "mcx_two_work_wires",
    "mps_prep_standard",
    "_multix_ladder_fanout",
    "multix_standard",
    "_num_applied_terms",
    "_out_multiplier_with_cache_resources",
    "out_multiplier_with_caddsub",
    "out_multiplier_with_qft",
    "out_square_with_adder_zeroed",
    "out_square_with_caddsub",
    "_partial_select",
    "_partial_unary_dense_state",
    "_partial_unary_state_prep_core",
    "_partial_unary_state_prep_standard_resources",
    "_permute_and_apply_parallel",
    "phase_gradient_state_prep_to_clifford_t",
    "phase_gradient_state_prep_to_ppr",
    "qrom_measurement",
    "_qrom_measurement_condition",
    "_qrom_measurement_resources",
    "qrom_unary_iteration",
    "adjoint_qrom_load_measurement",
    "qrom_load_select_swap",
    "qrom_load_to_qrom",
    "adjoint_select_swap_measurement",
    "select_swap",
    "_select_half",
    "select_multi_control_work_wire",
    "_select_pauli_rot_phase_gradient",
    "select_pauli_rot_standard",
    "select_unary",
    "semi_adder_standard",
    "_semi_adder_standard_resources",
    "_signed_out_multiplier_borrowed_resources",
    "_signed_out_multiplier_zeroed_resources",
    "_signed_out_square_from_unsigned_resources",
    "sos_state_prep_standard",
    "_sos_state_prep_with_wires",
    "_symmetrize",
    "toffoli_elbow",
    "trotter_cdf_multi_control",
    "trotter_cdf_standard",
    "trotter_cgf_standard",
    "_twos_complement_helper",
    "_validate_registers",
    "make_crz_to_phase_gradient_decomp",
    "make_phase_gradient_distillation_decomp",
    "make_rz_to_phase_gradient_decomp",
    "make_selectpaulirot_to_phase_gradient_decomp",
    "mottonen_decomp",
    "null_decomp",
    "phase_gradient_precision_stats",
    "validate_phase_gradient_wires",
]
