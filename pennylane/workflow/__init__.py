# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

.. image:: architecture.png
  :width: 400
  :alt: Alternative text

Contents
--------

.. currentmodule:: pennylane.workflow

Construction Utilities:
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~build_workflow
    ~get_interface_boundary

Executor types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.workflow

.. autosummary::
    :toctree: api

    ~Executor
    ~DeviceExecutor
    ~TransformProgramLayer
    ~MultiProcessingLayer

    ~interfaces.dispatcher.NullLayer
    ~interfaces.autograd_boundary.AutogradLayer
    ~interfaces.jax_boundary.JaxLayer
    ~interfaces.torch_boundary.TorchLayer
    ~interfaces.tf_boundary.TFLayer

.. currentmodule:: pennylane.operation

.. inheritance-diagram:: Executor, DeviceExecutor, TransformProgramLayer, MultiProcessingLayer, NullLayer, AutogradLayer, JaxLayer, TorchLayer, TFLayer
    :parts: 1

Derivatives
~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~DerivativeExecutor
    ~DeviceDerivatives
    ~TransformDerivatives

.. inheritance-diagram:: DerivativeExecutor, DeviceDerivatives, TransformDerivatives

"""

from .executor import Executor, DeviceExecutor, TransformProgramLayer, MultiProcessingLayer

from .gradient_layers import DerivativeExecutor, DeviceDerivatives, TransformDerivatives

from .interfaces import get_interface_boundary

from .build_workflow import build_workflow
