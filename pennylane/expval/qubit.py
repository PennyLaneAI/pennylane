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
    Hadamard
    Hermitian
    Identity

:html:`<h3>Code details</h3>`
"""

from pennylane.operation import Expectation


class PauliX(Expectation):
    r"""pennylane.expval.PauliX(wires)
    Expectation value of :class:`PauliX<pennylane.ops.qubit.PauliX>`.

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
    Expectation value of :class:`PauliY<pennylane.ops.qubit.PauliY>`.

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
    Expectation value of :class:`PauliZ<pennylane.ops.qubit.PauliZ>`.

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


class Hadamard(Expectation):
    r"""pennylane.expval.Hadamard(wires)
    Expectation value of the :class:`Hadamard<pennylane.ops.qubit.Hadamard>` observable.

    This expectation command returns the value

    .. math::
        \braket{H} = \braketT{\psi}{\cdots \otimes I\otimes H\otimes I\cdots}{\psi}

    where :math:`H` acts on the requested wire.

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
    Expectation value of an arbitrary Hermitian observable.

    For a Hermitian matrix :math:`A`, this expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`2^N\times 2^N`.

    Args:
        A (array): square hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = 0
    num_params = 1
    par_domain = "A"
    grad_method = "F"


# As both the qubit and the CV case need an Identity Expectation,
# and these need to reside in the same name space but have to have
# different types, this Identity class is not imported into expval
# directly (it is not put in __all__ below) and instead expval
# contains a placeholder class Identity that returns appropriate
# Identity instances via __new__() suitable for the respective device.
class Identity(Expectation):
    r"""pennylane.expval.Identity(wires)
    Expectation value of the identity observable :math:`\I`.

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.

    .. note::

        Can be used to check normalization in approximate simulators.

    """
    num_wires = 0
    num_params = 0
    par_domain = None
    grad_method = None


all_ops = [PauliX, PauliY, PauliZ, Hadamard, Hermitian]

__all__ = [cls.__name__ for cls in all_ops]
