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
    ~cond
    ~change_op_basis
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
    ~ChangeOpBasis
    ~CompositeOp
    ~Conditional
    ~Controlled
    ~ControlledOp
    ~Evolution
    ~Exp
    ~LinearCombination
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
    ~CH
    ~CCZ
    ~CSWAP
    ~CNOT
    ~Toffoli
    ~MultiControlledX
    ~CRX
    ~CRY
    ~CRZ
    ~CRot
    ~ControlledPhaseShift

Decompositions
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.ops

.. autosummary::
    :toctree: api

    ~one_qubit_decomposition
    ~two_qubit_decomposition
    ~multi_qubit_decomposition
    ~sk_decomposition
    ~rs_decomposition

Control Decompositions
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.ops.op_math

.. autosummary::
    :toctree: api

    ~ctrl_decomp_zyz
    ~ctrl_decomp_bisect

"""

from .adjoint import Adjoint, adjoint
from .composite import CompositeOp
from .condition import Conditional, cond
from .controlled import Controlled, ControlledOp, ctrl
from .controlled_ops import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    ControlledQubitUnitary,
    CPhase,
    CRot,
    MultiControlledX,
    Toffoli,
)
from .decompositions import (
    one_qubit_decomposition,
    sk_decomposition,
    rs_decomposition,
    two_qubit_decomposition,
    multi_qubit_decomposition,
    ctrl_decomp_bisect,
    ctrl_decomp_zyz,
)
from .evolution import Evolution
from .exp import Exp, exp
from .linear_combination import LinearCombination
from .pow import Pow, pow
from .prod import Prod, prod
from .change_op_basis import ChangeOpBasis, change_op_basis
from .sprod import SProd, s_prod
from .sum import Sum, sum
from .symbolicop import ScalarSymbolicOp, SymbolicOp

controlled_qubit_ops = {
    "ControlledQubitUnitary",
    "CY",
    "CZ",
    "CH",
    "CCZ",
    "CSWAP",
    "CNOT",
    "Toffoli",
    "MultiControlledX",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "ControlledPhaseShift",
    "CPhase",
}
