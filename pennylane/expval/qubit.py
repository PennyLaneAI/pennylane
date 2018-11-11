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
"""
.. _qubit_expval:

Qubit quantum expectations
==========================

.. currentmodule:: pennylane.expval.qubit

**Module name:** :mod:`pennylane.expval.qubit`

.. role:: html(raw)
   :format: html

This section contains the available built-in discrete-variable
quantum operations supported by PennyLane, as well as their conventions.

.. note:: Currently, all expectation commands return scalars.

:html:`<h3>Summary</h3>`

.. autosummary::
    PauliX
    PauliY
    PauliZ
    Hermitian


:html:`<h3>Code details</h3>`
"""

from pennylane.operation import Expectation


class PauliX(Expectation):
    r"""pennylane.expval.PauliX(wires)
    Returns the Pauli-X expectation value.

    This expectation command returns the value

    .. math::
        \braket{\sigma_z} = \braketT{\psi}{\cdots \otimes I\otimes \sigma_x\otimes I\cdots}{\psi}

    where :math:`\sigma_x` acts on the requested wire.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    par_domain = None


class PauliY(Expectation):
    r"""pennylane.expval.PauliY(wires)
    Returns the Pauli-Y expectation value.

    This expectation command returns the value

    .. math::
        \braket{\sigma_z} = \braketT{\psi}{\cdots \otimes I\otimes \sigma_y\otimes I\cdots}{\psi}

    where :math:`\sigma_y` acts on the requested wire

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    par_domain = None


class PauliZ(Expectation):
    r"""pennylane.expval.PauliZ(wires)
    Returns the Pauli-Z expectation value.

    This expectation command returns the value

    .. math::
        \braket{\sigma_z} = \braketT{\psi}{\cdots \otimes I\otimes \sigma_z\otimes I\cdots}{\psi}

    where :math:`\sigma_z` acts on the requested wire.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    par_domain = None


class Hermitian(Expectation):
    r"""pennylane.expval.Hermitian(A, wires)
    Returns the expectation value of an arbitrary Hermitian observable.

    For a Hermitian matrix :math:`A`, this expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wire.

    Args:
        A (array): square hermitian matrix.
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 1
    par_domain = 'A'
    grad_method = 'F'


all_ops = [PauliX, PauliY, PauliZ, Hermitian]

__all__ = [cls.__name__ for cls in all_ops]
