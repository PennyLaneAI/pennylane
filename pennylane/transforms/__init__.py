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

QNode transforms
----------------

The following transforms act on QNodes. They return new transformed functions
that compute the desired quantity.

.. autosummary::
    :toctree: api

    ~transforms.classical_jacobian
    ~draw
    ~metric_tensor

Quantum function transforms
---------------------------

The following transforms act on quantum functions (Python functions
containing quantum operations) that are used *inside* QNodes.

.. autosummary::
    :toctree: api

    ~adjoint
    ~ctrl
    ~transforms.invisible
    ~quantum_monte_carlo

Tape transforms
---------------

The following transforms act on quantum tapes, and return one or
more tapes as well as a classical processing function.

.. autosummary::
    :toctree: api

    ~transforms.measurement_grouping
    ~transforms.metric_tensor_tape
"""
from .adjoint import adjoint
from .classical_jacobian import classical_jacobian
from .control import ControlledOperation, ctrl
from .draw import draw
from .invisible import invisible
from .measurement_grouping import measurement_grouping
from .metric_tensor import metric_tensor, metric_tensor_tape
from .qmc import quantum_monte_carlo
