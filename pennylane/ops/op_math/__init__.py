# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains classes and functions for Operator arithmetic.

Constructor Functions
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~adjoint
    ~ctrl
    ~op_sum
    ~prod
    ~s_prod

Symbolic Classes
~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.ops.op_math

.. autosummary::
    :toctree: api

    ~Adjoint
    ~ControlledOperation
    ~Controlled
    ~ControlledOp
    ~Pow
    ~Prod
    ~Sum
    ~SProd
    ~SymbolicOp

"""

from .adjoint_class import Adjoint
from .adjoint_constructor import adjoint
from .controlled_class import Controlled, ControlledOp

from .prod import prod, Prod

from .sum import op_sum, Sum

from .sprod import s_prod, SProd

from .control import ctrl, ControlledOperation
from .pow_class import Pow

from .symbolicop import SymbolicOp
