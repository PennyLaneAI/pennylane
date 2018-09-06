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
"""Additional gates"""
from openqml import Operation
import projectq as pq

# Extra Operations in OpenQML provided by this plugin

class S(Operation):
    r"""S gate.

    .. math::
        S() = \begin(pmatrix)1&0\\0&i\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('S', [], wires)

class T(Operation):
    r"""T gate.

    .. math::
        T() = \begin(pmatrix)1&0\\0&exp(i \pi / 4)\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('T', [], wires)

class SqrtX(Operation):
    r"""Square toot X gate.

    .. math::
        SqrtX() = \begin(pmatrix)1+i&1-i\\1-i&1+i\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('SqrtX', [], wires)

class SqrtSwap(Operation):
    r"""Square SWAP gate.

    .. math::
        SqrtSwap() = \begin(pmatrix)1&0&0&0\\0&(1+i)/2&(1-i)/2&0\\0&(1-i)/2 &(1+i)/2&0\\0&0&0&1\end(pmatrix)

    Args:
        wires (seq[int]): the subsystems the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('SqrtSwap', [], wires)

class AllZ(Operation):
    r"""Z on all qubits.

    .. math::
        AllZ() = Z \otimes\dots\otimes Z
    """
    def __init__(self):
        super().__init__('AllZ', [], [])


# Wrapper classes for Operations that are missing a class in ProjectQ

class CNOT(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the CNOT gate.

    Contrary to other gates, ProjectQ does not have a class for the CNOT gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        #return pq.ops.C(pq.ops.XGate())
        return pq.ops.C(pq.ops.NOT)


class CZ(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the CNOT gate.

    Contrary to other gates, ProjectQ does not have a class for the CZ gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.C(pq.ops.ZGate())

class Toffoli(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the Toffoli gate.

    Contrary to other gates, ProjectQ does not have a class for the Toffoli gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.C(pq.ops.ZGate(), 2)

class AllZGate(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the AllZ gate.

    Contrary to other gates, ProjectQ does not have a class for the AllZ gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.Tensor(pq.ops.ZGate())

class Rot(pq.ops.BasicGate):
    """Class for the arbitrary single qubit rotation gate.

    ProjectQ does not currently have an arbitrary single qubit rotation gate.
    """
    def __new__(*par):
        raise NotImplementedError("ProjectQ does not currently have an arbitrary single qubit rotation gate.") #todo: update depending on https://github.com/ProjectQ-Framework/ProjectQ/issues/268

class QubitUnitary(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the QubitUnitary gate.

    ProjectQ does not currently have an arbitrary QubitUnitary gate.
    """
    def __new__(*par):
        raise NotImplementedError("ProjectQ does not currently have an arbitrary single qubit unitary gate.") #todo: update depending on https://github.com/ProjectQ-Framework/ProjectQ/issues/268
