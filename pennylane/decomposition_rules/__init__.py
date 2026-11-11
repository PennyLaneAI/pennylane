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

The modules directly beneath this package mirror :mod:`pennylane.ops`. The
``templates``, ``transforms``, and ``drawer`` branches mirror their corresponding
PennyLane packages. Rules are loaded automatically when PennyLane is imported.
Use :func:`pennylane.list_decomps` to retrieve the registered rules for an operator.

.. currentmodule:: pennylane.decomposition_rules

.. autosummary::

    qubit
    op_math
    identity
    meta
    templates
    transforms
    drawer
"""

from importlib import import_module

_BUILTIN_MODULES = (
    "pennylane.decomposition_rules.identity",
    "pennylane.decomposition_rules.op_math.change_op_basis2",
    "pennylane.decomposition_rules.meta",
    "pennylane.decomposition_rules.qubit.arithmetic_ops",
    "pennylane.decomposition_rules.qubit.matrix_ops",
    "pennylane.decomposition_rules.qubit.non_parametric_ops",
    "pennylane.decomposition_rules.op_math.prod2",
    "pennylane.decomposition_rules.op_math.prod",
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
    "pennylane.decomposition_rules.drawer.label",
    "pennylane.decomposition_rules.op_math.controlled2",
    "pennylane.decomposition_rules.op_math.adjoint2",
    "pennylane.decomposition_rules.op_math.pow2",
    "pennylane.decomposition_rules.op_math.decompositions.controlled_decompositions",
    "pennylane.decomposition_rules.op_math.decompositions.unitary_decompositions",
    "pennylane.decomposition_rules.transforms.decompositions.crz_phase_gradient",
    "pennylane.decomposition_rules.transforms.decompositions.phase_gradient_state_prep_decomp",
    "pennylane.decomposition_rules.transforms.decompositions.rz_phase_gradient",
    "pennylane.decomposition_rules.transforms.decompositions.select_pauli_rot_phase_gradient",
)


def _load_builtin_rules():
    """Load the built-in registrations after the operator and template definitions."""
    for module in _BUILTIN_MODULES:
        import_module(module)
