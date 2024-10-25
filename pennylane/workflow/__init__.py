# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the core objects for managing a PennyLane workflow.

.. currentmodule:: pennylane

Execution functions and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~execute
    ~workflow.set_shots
    ~workflow.construct_tape
    ~workflow.construct_batch
    ~workflow.get_transform_program
    ~workflow.get_best_diff_method

Supported interfaces
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~workflow.interfaces.autograd
    ~workflow.interfaces.jax
    ~workflow.interfaces.jax_jit
    ~workflow.interfaces.tensorflow
    ~workflow.interfaces.tensorflow_autograph
    ~workflow.interfaces.torch

Jacobian Product Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~workflow.jacobian_products.JacobianProductCalculator
    ~workflow.jacobian_products.TransformJacobianProducts
    ~workflow.jacobian_products.DeviceDerivatives
    ~workflow.jacobian_products.DeviceJacobianProducts

.. include:: ../../pennylane/workflow/return_types_spec.rst

"""
from .get_best_diff_method import get_best_diff_method
from .get_gradient_fn import _get_gradient_fn
from .construct_batch import construct_batch, get_transform_program
from .construct_tape import construct_tape
from .execution import INTERFACE_MAP, SUPPORTED_INTERFACE_NAMES, execute
from .qnode import QNode, qnode
from .set_shots import set_shots
