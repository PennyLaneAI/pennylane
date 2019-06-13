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
"""Submodule containing core quantum operations supported by PennyLane.

.. currentmodule:: pennylane.ops

PennyLane supports a collection of built-in quantum operations,
including both discrete-variable (DV) gates as used in the qubit model,
and continuous-variable (CV) gates as used in the qumode model of quantum
computation.

Here, we summarize the built-in operations supported by PennyLane, as well
as the conventions chosen for their implementation.

.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in operations defined here as possible.

    If the convention differs between the built-in PennyLane operation
    and the corresponding operation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.


.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    ops/qubit
    ops/cv
"""

from .cv import *
from .qubit import *

from .cv import __all__ as _cv__all__
from .qubit import __all__ as _qubit__all__


class PlaceholderOperation:
    r"""A generic base class for constructing placeholders for operations that
    exist under the same name in CV and qubit-based devices.

    When instantiated inside a QNode context, returns an instance
    of the respective class in expval.cv or expval.qubit.
    """
    # pylint: disable=too-few-public-methods
    def __new__(cls, *args, **kwargs):
        # pylint: disable=protected-access
        if QNode._current_context is None:
            raise QuantumFunctionError("Quantum operations can only be used inside a qfunc.")

        supported_expectations = QNode._current_context.device.expectations

        # TODO: in the next breaking release, make it mandatory for plugins to declare
        # whether they target qubit or CV operations, to avoid needing to
        # inspect supported_expectation directly.
        if supported_expectations.intersection([cls for cls in _cv__all__]):
            return getattr(cv, cls.__name__)(*args, **kwargs)
        elif supported_expectations.intersection([cls for cls in _qubit__all__]):
            return getattr(qubit, cls.__name__)(*args, **kwargs)
        else:
            raise QuantumFunctionError("Unable to determine whether this device supports CV or qubit "
                                       "Operations when constructing this "+cls.__name__+" Observable.")


class Identity(PlaceholderOperation): #pylint: disable=too-few-public-methods,function-redefined
    r"""pennylane.ops.Identity(wires)
    Observable value of the identity observable :math:`\I`.

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.
    """
    operation = False
    observable = True

    num_wires = 0
    num_params = 0
    par_domain = None


__all__ = _cv__all__ + _qubit__all__ + [Identity.__name__]
