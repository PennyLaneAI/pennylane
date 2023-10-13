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
This subpackage defines functions for interfacing devices' execution
capabilities with different machine learning libraries.

.. currentmodule:: pennylane

Execution functions and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~execute
    ~interfaces.cache_execute
    ~interfaces.set_shots

Supported interfaces
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~interfaces.autograd
    ~interfaces.jax
    ~interfaces.jax_jit
    ~interfaces.tensorflow
    ~interfaces.tensorflow_autograph
    ~interfaces.torch

Jacobian Product Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~interfaces.jacobian_products.JacobianProductCalculator
    ~interfaces.jacobian_products.TransformJacobianProducts
    ~interfaces.jacobian_products.DeviceDerivatives
    ~interfaces.jacobian_products.DeviceJacobianProducts

"""
from .execution import cache_execute, execute, INTERFACE_MAP, SUPPORTED_INTERFACES
from .set_shots import set_shots


class InterfaceUnsupportedError(NotImplementedError):
    """Exception raised when features not supported by an interface are
    attempted to be used."""
