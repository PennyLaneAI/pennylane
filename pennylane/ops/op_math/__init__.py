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
    ~sum
    ~pow
    ~prod
    ~s_prod

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
    ~ScalarSymbolicOp

Controlled Operator Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~ControlledQubitUnitary
    ~CY
    ~CZ

Decompositions
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.ops.op_math

.. autosummary::
    :toctree: api

    ~ctrl_decomp_zyz
    ~ctrl_decomp_bisect

"""

from .adjoint import Adjoint, adjoint
from .composite import CompositeOp
from .controlled import Controlled, ControlledOp, ctrl
from .controlled_ops import ControlledQubitUnitary, CY, CZ
from .evolution import Evolution
from .exp import Exp, exp
from .pow import Pow, pow
from .prod import Prod, prod
from .sprod import SProd, s_prod
from .sum import Sum, sum
from .symbolicop import ScalarSymbolicOp, SymbolicOp
from .controlled_decompositions import ctrl_decomp_zyz, ctrl_decomp_bisect

controlled_qubit_ops = {"ControlledQubitUnitary", "CY", "CZ"}
