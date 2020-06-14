# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This subpackage contains the supported types of QNodes.

.. currentmodule:: pennylane.qnodes
"""
from .base import BaseQNode, QuantumFunctionError
from .cv import CVQNode
from .decorator import qnode, QNode
from .jacobian import JacobianQNode
from .qubit import QubitQNode
from .passthru import PassthruQNode
from .rev import ReversibleQNode
