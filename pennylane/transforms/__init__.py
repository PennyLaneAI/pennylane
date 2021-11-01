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
Quantum transforms are composable transformations of quantum functions. A
quantum transform, when applied to a qfunc (a Python function containing
quantum operations) or a :class:`~pennylane.QNode`, will **transform** the
quantum circuit in order to achieve the desired result. For example, quantum
transforms can be used to:

- extract information from the quantum function (e.g., to extract unitary
  matrices or draw the underlying circuit),

- modify or compile the quantum function,

- generate new quantum functions that compute quantum properties and metrics,
  such as the gradient and the Fisher information matrix.

This module provides a selection of device-independent, differentiable quantum
gradient transforms. As such, not only is the output of a quantum transform
differentiable, but quantum transforms *themselves* can be differentiated with
respect to any floating point arguments.

In addition, this module also includes an API for writing your own quantum
transforms.

.. currentmodule:: pennylane

Overview
--------

Information
~~~~~~~~~~~

These transforms accept QNodes, and return new transformed functions
that compute the desired quantity. These transforms do not result
in quantum executions.

.. autosummary::
    :toctree: api

    ~draw
    ~specs
    ~transforms.get_unitary_matrix
    ~transforms.classical_jacobian


Multiple circuit transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms accept QNodes, and return new QNodes
that compute the desired quantity. The transformed QNode, when
executed, may result in multiple quantum circuit evaluations
under the hood.

    ~batch_params
    ~metric_tensor
    ~transforms.mitigate_with_zne
    ~transforms.hamiltonian_expand
    ~transforms.measurement_grouping


Circuit modification and compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This set of transforms map a single quantum function to a new quantum
function. This includes basic circuit compilation tasks.

.. autosummary::
    :toctree: api

    ~adjoint
    ~ctrl
    ~compile
    ~transforms.insert
    ~transforms.cancel_inverses
    ~transforms.commute_controlled
    ~transforms.merge_rotations
    ~transforms.single_qubit_fusion
    ~transforms.unitary_to_rot
    ~apply_controlled_Q
    ~quantum_monte_carlo

There are also utility functions and decompositions available that assist with
both transforms, and decompositions within the larger PennyLane codebase.

.. autosummary::
    :toctree: api

    ~transforms.zyz_decomposition
    ~transforms.two_qubit_decomposition


Custom transforms and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following decorators and convenience functions are provided
to help build custom QNode, quantum function, and tape transforms:

.. autosummary::
    :toctree: api

    ~single_tape_transform
    ~batch_transform
    ~qfunc_transform
    ~transforms.make_tape
    ~transforms.create_expand_fn
    ~transforms.expand_invalid_trainable
    ~transforms.expand_multipar
    ~transforms.expand_nonunitary_gen
"""
# Import the decorators first to prevent circular imports when used in other transforms
from .batch_transform import batch_transform
from .qfunc_transforms import make_tape, single_tape_transform, qfunc_transform
from .adjoint import adjoint
from .batch_params import batch_params
from .classical_jacobian import classical_jacobian
from .compile import compile
from .control import ControlledOperation, ctrl
from .decompositions import zyz_decomposition, two_qubit_decomposition
from .draw import draw
from .hamiltonian_expand import hamiltonian_expand
from .measurement_grouping import measurement_grouping
from .metric_tensor import metric_tensor
from .insert_ops import insert
from .mitigate import mitigate_with_zne
from .optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
)
from .specs import specs
from .qmc import apply_controlled_Q, quantum_monte_carlo
from .unitary_to_rot import unitary_to_rot
from .get_unitary_matrix import get_unitary_matrix
from .tape_expand import (
    expand_invalid_trainable,
    expand_multipar,
    expand_nonunitary_gen,
    create_expand_fn,
)
