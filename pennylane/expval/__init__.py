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
"""Submodule containing core quantum expectation values supported by PennyLane.

.. currentmodule:: pennylane.expval

PennyLane also supports a collection of built-in quantum **expectations**,
including both discrete-variable (DV) expectations as used in the qubit model,
and continuous-variable (CV) expectations as used in the qumode model of quantum
computation.

Here, we summarize the built-in expectations supported by PennyLane, as well
as the conventions chosen for their implementation.

.. note::

    All quantum operations in PennyLane are top level; they can be accessed
    via ``qml.OperationName``. Expectation values, however, are contained within
    the :mod:`pennylane.expval`, and are thus accessed via ``qml.expval.ExpectationName``.


.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in expectations defined here as possible.

    If the convention differs between the built-in PennyLane expectation
    and the corresponding expectation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.


.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    expval/qubit
    expval/cv
"""

from pennylane.qnode import QNode, QuantumFunctionError

from . import cv
from . import qubit

from .qubit import * #pylint: disable=unused-wildcard-import,wildcard-import
from .cv import * #pylint: disable=unused-wildcard-import,wildcard-import

from .cv import __all__ as _cv__all__
from .qubit import __all__ as _qubit__all__


class PlaceholderExpectation():
    r"""pennylane.expval.PlaceholderExpectation()
    A generic base class for constructing placeholders for operations that
    exist under the same name in CV and qubit-based devices.

    When instantiated inside a QNode context, returns an instance
    of the respective class in expval.cv or expval.qubit.
    """
    # pylint: disable=too-few-public-methods
    def __new__(cls, *args, **kwargs):
        # pylint: disable=protected-access
        if QNode._current_context is None:
            if cls.__name__ in _qubit__all__:
                supported_expectations = set(_qubit__all__)
            else:
                supported_expectations = set(_cv__all__)
        else:
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
                                       "Operations when constructing this "+cls.__name__+" Expectation.")


class Identity(PlaceholderExpectation): #pylint: disable=too-few-public-methods,function-redefined
    r"""pennylane.expval.Identity(wires)
    Expectation value of the identity observable :math:`\I`.

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.

    This is a placeholder for the Identity classes in expval.qubit and expval.cv
    and instantiates the Identity appropriate for the respective device.
    """
    num_wires = 0
    num_params = 0
    par_domain = None


class Variance:
    r"""A generic base class for constructing variance expectations for
    expectation values that exist for CV and qubit-based devices.

    When instantiated inside a QNode context, returns an instance
    of the respective class in expval.cv or expval.qubit, with
    class attribute ``return_type`` set to ``'variance'``.
    """
    def __getattr__(self, item):
        if item not in __all__:
            raise AttributeError("module 'pennylane.var' has no attribute '{}'".format(item))

        if QNode._current_context is None:
            if item in _qubit__all__:
                expvals = set(_qubit__all__)
            else:
                expvals = set(_cv__all__)
        else:
            expvals = QNode._current_context.device.expectations

        if expvals.intersection([item for item in _cv__all__]):
            expval_class = getattr(cv, item)
        elif expvals.intersection([item for item in _qubit__all__]):
            expval_class = getattr(qubit, item)
        else:
            raise QuantumFunctionError("Unable to determine whether this device supports CV or qubit "
                                       "Operations when constructing the {} Variance.".format(item))

        return type(item, (expval_class,), {"return_type": "variance"})

    def __dir__(self):
        return __all__


__all__ = _cv__all__ + _qubit__all__ + [Identity.__name__]
