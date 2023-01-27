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
    ~exp
    ~op_sum
    ~pow
    ~prod
    ~s_prod
    ~ops.dot

Symbolic Classes
~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.ops.op_math

.. autosummary::
    :toctree: api

    ~Adjoint
    ~CompositeOp
    ~Controlled
    ~ControlledOp
    ~Evolution
    ~Exp
    ~Pow
    ~Prod
    ~Sum
    ~SProd
    ~SymbolicOp

Controlled Operator Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ControlledQubitUnitary

"""

from .adjoint import Adjoint, adjoint
from .composite import CompositeOp
from .controlled import Controlled, ControlledOp, ctrl
from .controlled_ops import ControlledQubitUnitary
from .dot import dot
from .exp import Evolution, Exp, exp
from .pow import Pow, pow
from .prod import Prod, prod
from .sprod import SProd, s_prod
from .sum import Sum, op_sum
from .symbolicop import SymbolicOp

controlled_qubit_ops = {
    "ControlledQubitUnitary",
}
