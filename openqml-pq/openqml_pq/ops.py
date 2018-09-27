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
from openqml.operation import Expectation

import projectq as pq

# Extra Operations in OpenQML provided by this plugin

class S(Operation):
    r"""S gate.

    .. math::
        S() = \begin(pmatrix)1&0\\0&i\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    n_params = 0
    n_wires = 1


class T(Operation):
    r"""T gate.

    .. math::
        T() = \begin(pmatrix)1&0\\0&exp(i \pi / 4)\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    n_params = 0
    n_wires = 1

class SqrtX(Operation):
    r"""Square toot X gate.

    .. math::
        SqrtX() = \begin(pmatrix)1+i&1-i\\1-i&1+i\end(pmatrix)

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    n_params = 0
    n_wires = 1

class SqrtSwap(Operation):
    r"""Square SWAP gate.

    .. math::
        SqrtSwap() = \begin(pmatrix)1&0&0&0\\0&(1+i)/2&(1-i)/2&0\\0&(1-i)/2 &(1+i)/2&0\\0&0&0&1\end(pmatrix)

    Args:
        wires (seq[int]): the subsystems the Operation acts on.
    """
    n_params = 0
    n_wires = 2

class AllZ(Expectation):
    r"""Measure Z on all qubits.

    .. math::
        AllZ() = Z \otimes\dots\otimes Z
    """
    n_params = 0
    n_wires = 0 #todo: how to represent a gate that acts on all wires?


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

    ProjectQ does not currently have an arbitrary single qubit rotation gate, so we provide a class that return a suitable combination of rotation gates assembled into a single gate from the constructor of this class. 
    """
    def __new__(*par):
        gate3 = pq.ops.Rz(par[0])
        gate2 = pq.ops.Ry(par[1])
        gate1 = pq.ops.Rz(par[2])
        rot_gate = pq.ops.BasicGate()
        rot_gate.matrix = numpy.dot(gate3.matrix, gate2.matrix, gate1.matrix)
        return rot_gate

class QubitUnitary(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the QubitUnitary gate.

    ProjectQ does not currently have a real arbitrary QubitUnitary gate, but it allows to directly set the matrix of single qubit gates and can then still decompose them into the elementary gates set, so we do this here.
    """
    def __new__(*par):
        my_gate = pq.ops.BasicGate()
        my_gate.matrix = numpy.matrix(par)
