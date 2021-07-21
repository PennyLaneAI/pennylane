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
This module contains the available built-in discrete-variable
quantum operations supported by PennyLane, as well as their conventions.
"""
import cmath
import functools
import warnings

# pylint:disable=abstract-method,arguments-differ,protected-access
import math
import numpy as np
import scipy
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.operation import AnyWires, AllWires, DiagonalOperation, Observable, Operation
from pennylane.templates.decorator import template
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation
from pennylane.utils import expand, pauli_eigs
from pennylane.wires import Wires

INV_SQRT2 = 1 / math.sqrt(2)


class AdjointError(Exception):
    """Exception for non-adjointable operations."""

    pass

