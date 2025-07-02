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
    ~workflow.construct_tape
    ~workflow.construct_batch
    ~workflow.construct_execution_config
    ~workflow.get_transform_program
    ~workflow.get_best_diff_method
    ~workflow.set_shots

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
from .construct_batch import construct_batch, get_transform_program
from .construct_tape import construct_tape
from .construct_execution_config import construct_execution_config
from .execution import execute
from .get_best_diff_method import get_best_diff_method
from .qnode import QNode, qnode
from .resolution import (
    _resolve_execution_config,
    _resolve_mcm_config,
    _resolve_diff_method,
    _resolve_interface,
)
from .set_shots import set_shots
from ._cache_transform import _cache_transform
from ._setup_transform_program import _setup_transform_program
from .run import run
