# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the device class and context manager"""
from openqml.operation import Operation


class XDisplacement(Operation):
    r"""Continuous-variable position displacement in the phase space.

    .. math::
        X(x) = \exp\left(-i x \p/\hbar\right)

    Args:
        x (float): the position displacement.
        wires (int): the subsystem the Operation acts on.
    """

    def __init__(self, x, wires):
        super().__init__('XDisplacement', [x], wires)


class ZDisplacement(Operation):
    r"""Continuous-variable momentum displacement in the phase space.

    .. math::
        Z(p) = \exp\left(i p \x/\hbar\right)

    Args:
        p (float): the momentum displacement.
        wires (int): the subsystem the Operation acts on.
    """

    def __init__(self, p, wires):
        super().__init__('PDisplacement', [p], wires)
