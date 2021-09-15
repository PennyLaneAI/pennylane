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

Thes transforms accept QNodes, and return new transformed functions
that compute the desired quantity.

.. autosummary::
    :toctree: api

    ~transforms.classical_jacobian
    ~draw
    ~metric_tensor
    ~specs

Transforms that act on quantum functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum functions (Python functions
containing quantum operations) that are used to construct QNodes.

.. autosummary::
    :toctree: api

    ~adjoint
    ~ctrl
    ~transforms.invisible
    ~apply_controlled_Q
    ~quantum_monte_carlo

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

There are also utility functions and decompositions available that assist with
both transforms, and decompositions within the larger PennyLane codebase.

.. autosummary::
    :toctree: api

    ~transforms.zyz_decomposition

Transforms that act on tapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept quantum tapes, and return one or
more tapes as well as a classical processing function.

.. autosummary::
    :toctree: api

    ~transforms.measurement_grouping
    ~transforms.metric_tensor_tape
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
"""
# Import the decorators first to prevent circular imports when used in other transforms
from .batch_transform import batch_transform
from .qfunc_transforms import make_tape, single_tape_transform, qfunc_transform
from .adjoint import adjoint
from .classical_jacobian import classical_jacobian
from .compile import compile
from .control import ControlledOperation, ctrl
from .decompositions import zyz_decomposition
from .draw import draw
from .hamiltonian_expand import hamiltonian_expand
from .invisible import invisible
from .measurement_grouping import measurement_grouping
from .metric_tensor import metric_tensor, metric_tensor_tape
from .optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
)
from .specs import specs
from .qmc import apply_controlled_Q, quantum_monte_carlo
from .unitary_to_rot import unitary_to_rot
