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
    ~draw
    ~draw_mpl
    ~transforms.get_unitary_matrix
    ~metric_tensor
    ~adjoint_metric_tensor
    ~specs
    ~transforms.mitigate_with_zne

Transforms that act on quantum functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum functions (Python functions
containing quantum operations) that are used to construct QNodes.

.. autosummary::
    :toctree: api

    ~adjoint
    ~ctrl
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

There are also utility functions and decompositions available that assist with
both transforms, and decompositions within the larger PennyLane codebase.

.. autosummary::
    :toctree: api

    ~transforms.zyz_decomposition
    ~transforms.two_qubit_decomposition
    ~transforms.set_decomposition

Transforms that act on tapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum tapes, and return one or
more tapes as well as a classical processing function.

.. autosummary::
    :toctree: api

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
    ~transforms.make_tape
    ~transforms.map_batch_transform
    ~transforms.create_expand_fn
    ~transforms.create_decomp_expand_fn
    ~transforms.expand_invalid_trainable
    ~transforms.expand_multipar
    ~transforms.expand_trainable_multipar
    ~transforms.expand_nonunitary_gen
"""
# Import the decorators first to prevent circular imports when used in other transforms
from .batch_transform import batch_transform, map_batch_transform
from .qfunc_transforms import make_tape, single_tape_transform, qfunc_transform
from .adjoint import adjoint
from .batch_params import batch_params
from .classical_jacobian import classical_jacobian
from .compile import compile
from .control import ControlledOperation, ctrl
from .decompositions import zyz_decomposition, two_qubit_decomposition
from .draw import draw, draw_mpl
from .hamiltonian_expand import hamiltonian_expand
from .measurement_grouping import measurement_grouping
from .metric_tensor import metric_tensor
from .adjoint_metric_tensor import adjoint_metric_tensor
from .insert_ops import insert
from .mitigate import mitigate_with_zne
from .optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
    merge_amplitude_embedding,
    remove_barrier,
    undo_swaps,
)
from .specs import specs
from .qmc import apply_controlled_Q, quantum_monte_carlo
from .unitary_to_rot import unitary_to_rot
from .get_unitary_matrix import get_unitary_matrix
from .tape_expand import (
    expand_invalid_trainable,
    expand_multipar,
    expand_nonunitary_gen,
    expand_trainable_multipar,
    create_expand_fn,
    create_decomp_expand_fn,
    set_decomposition,
)
