# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage contains QNode, quantum function, device, and tape transforms.


.. currentmodule:: pennylane

Transforms
----------

Transforms that act on QNodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept QNodes, and return new transformed functions
that compute the desired quantity.

.. autosummary::
    :toctree: api

    ~transforms.classical_jacobian
    ~batch_params
    ~batch_input
    ~batch_partial
    ~metric_tensor
    ~adjoint_metric_tensor
    ~specs
    ~transforms.split_non_commuting

Transforms that act on quantum functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum functions (Python functions
containing quantum operations) that are used to construct QNodes.

.. autosummary::
    :toctree: api

    ~transforms.cond
    ~defer_measurements
    ~apply_controlled_Q
    ~quantum_monte_carlo
    ~transforms.insert

Transforms for circuit compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This set of transforms accept quantum functions, and perform basic circuit compilation tasks.

.. autosummary::
    :toctree: api

    ~compile
    ~transforms.cancel_inverses
    ~transforms.commute_controlled
    ~transforms.merge_rotations
    ~transforms.single_qubit_fusion
    ~transforms.unitary_to_rot
    ~transforms.merge_amplitude_embedding
    ~transforms.remove_barrier
    ~transforms.undo_swaps
    ~transforms.pattern_matching_optimization
    ~transforms.transpile

There are also utility functions and decompositions available that assist with
both transforms, and decompositions within the larger PennyLane codebase.

.. autosummary::
    :toctree: api

    ~transforms.zyz_decomposition
    ~transforms.two_qubit_decomposition
    ~transforms.set_decomposition
    ~transforms.simplify
    ~transforms.pattern_matching

There are also utility functions that take a circuit and return a DAG.

.. autosummary::
    :toctree: api

    ~transforms.commutation_dag
    ~transforms.CommutationDAG
    ~transforms.CommutationDAGNode

Transform for circuit cutting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~.cut_circuit` transform accepts a QNode and returns a new function that cuts the original circuit,
allowing larger circuits to be split into smaller circuits that are compatible with devices that
have a restricted number of qubits.

.. autosummary::
    :toctree: api

    ~cut_circuit

The :func:`~.cut_circuit_mc` transform is designed to be used for cutting circuits which contain :func:`~.sample`
measurements and is implemented using a Monte Carlo method. Similarly to the :func:`~.cut_circuit`
transform, this transform accepts a QNode and returns a new function that cuts the original circuit.
This transform can also accept an optional classical processing function to calculate an
expectation value.

.. autosummary::
    :toctree: api

    ~cut_circuit_mc

There are also low-level functions that can be used to build up the circuit cutting functionalities:

.. autosummary::
    :toctree: api

    ~transforms.qcut.tape_to_graph
    ~transforms.qcut.replace_wire_cut_nodes
    ~transforms.qcut.fragment_graph
    ~transforms.qcut.graph_to_tape
    ~transforms.qcut.remap_tape_wires
    ~transforms.qcut.expand_fragment_tape
    ~transforms.qcut.expand_fragment_tapes_mc
    ~transforms.qcut.qcut_processing_fn
    ~transforms.qcut.qcut_processing_fn_sample
    ~transforms.qcut.qcut_processing_fn_mc
    ~transforms.qcut.CutStrategy
    ~transforms.qcut.kahypar_cut
    ~transforms.qcut.place_wire_cuts
    ~transforms.qcut.find_and_place_cuts

Transforms that act on tapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum tapes, and return one or
more tapes as well as a classical processing function.

.. autosummary::
    :toctree: api

    ~transforms.broadcast_expand
    ~transforms.measurement_grouping
    ~transforms.hamiltonian_expand

Decorators and utility functions
--------------------------------

The following decorators and convenience functions are provided
to help build custom QNode, quantum function, and tape transforms:

.. autosummary::
    :toctree: api

    ~single_tape_transform
    ~batch_transform
    ~qfunc_transform
    ~op_transform
    ~transforms.make_tape
    ~transforms.map_batch_transform
    ~transforms.create_expand_fn
    ~transforms.create_decomp_expand_fn
    ~transforms.expand_invalid_trainable
    ~transforms.expand_multipar
    ~transforms.expand_trainable_multipar
    ~transforms.expand_nonunitary_gen

Transforms for error mitigation
-------------------------------

.. autosummary::
    :toctree: api

    ~transforms.mitigate_with_zne
    ~transforms.fold_global
    ~transforms.poly_extrapolate
    ~transforms.richardson_extrapolate
"""
# Import the decorators first to prevent circular imports when used in other transforms
from .batch_transform import batch_transform, map_batch_transform
from .qfunc_transforms import make_tape, single_tape_transform, qfunc_transform
from .op_transforms import op_transform
from .batch_params import batch_params
from .batch_input import batch_input
from .batch_partial import batch_partial
from .classical_jacobian import classical_jacobian
from .condition import cond, Conditional
from .compile import compile
from .decompositions import zyz_decomposition, two_qubit_decomposition
from .defer_measurements import defer_measurements
from .hamiltonian_expand import hamiltonian_expand
from .split_non_commuting import split_non_commuting
from .measurement_grouping import measurement_grouping
from .metric_tensor import metric_tensor
from .adjoint_metric_tensor import adjoint_metric_tensor
from .insert_ops import insert
from .mitigate import mitigate_with_zne, fold_global, poly_extrapolate, richardson_extrapolate
from .optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
    merge_amplitude_embedding,
    remove_barrier,
    undo_swaps,
    pattern_matching,
    pattern_matching_optimization,
)
from .specs import specs
from .qmc import apply_controlled_Q, quantum_monte_carlo
from .unitary_to_rot import unitary_to_rot
from .commutation_dag import (
    commutation_dag,
    is_commuting,
    CommutationDAG,
    CommutationDAGNode,
    simplify,
)
from .tape_expand import (
    expand_invalid_trainable,
    expand_multipar,
    expand_nonunitary_gen,
    expand_trainable_multipar,
    create_expand_fn,
    create_decomp_expand_fn,
    set_decomposition,
)
from .transpile import transpile
from . import qcut
from .qcut import cut_circuit, cut_circuit_mc
from .broadcast_expand import broadcast_expand
