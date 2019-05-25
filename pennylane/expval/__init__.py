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
        from pennylane.qnode import QNode, QuantumFunctionError

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


__all__ = _cv__all__ + _qubit__all__ + [Identity.__name__]


class VarianceFactory:
    r"""A class factory with dynamic attributes for constructing variance of
    observables values that are defined in expval.cv and expval.qubit.

    When instantiated inside a QNode context, returns an instance
    of the respective class in expval.cv or expval.qubit, with
    class attribute ``return_type`` set to ``'variance'``.

    To use the dynamic attribute loading, you may use this class as
    follows:

    >>> var = VarianceFactory()
    >>> var.PauliX
    >>> var.Homodyne
    >>> var.qubit.Hermitian # only qubit observables
    >>> var.cv.MeanPhoton # only CV observables
    """
    def __init__(self, observables=__all__, submodule=''):
        self.observables = observables
        self.submodule = submodule

    def __getattr__(self, item):
        # pylint: disable=protected-access,too-many-branches
        if item == 'qubit':
            # allows the construct qml.var.qubit.PauliX
            return VarianceFactory(observables=_qubit__all__, submodule='.qubit')

        if item == 'cv':
            # allows the construct qml.cv.qubit.Homodyne
            return VarianceFactory(observables=_cv__all__, submodule='.cv')

        if item not in self.observables:
            raise AttributeError("module 'pennylane.var{}' has no attribute '{}'".format(self.submodule, item))

        if self.observables == __all__:
            from pennylane.qnode import QNode, QuantumFunctionError

            if QNode._current_context is not None:
                # inside a QNode
                # get the QNode device supported expectations
                expvals = QNode._current_context.device.expectations

                # inspect the resulting expectations to determine if
                # the device is a CV or a qubit device
                # TODO: in the next breaking release, make it mandatory for plugins to declare
                # whether they target qubit or CV operations, to avoid needing to
                # inspect supported_expectation directly.
                if expvals.intersection([item for item in _cv__all__]):
                    expval_class = getattr(cv, item)
                elif expvals.intersection([item for item in _qubit__all__]):
                    expval_class = getattr(qubit, item)
                else:
                    raise QuantumFunctionError("Unable to determine whether this device supports CV or qubit "
                                               "Operations when constructing the {} Variance.".format(item))
            else:
                # # unable to determine which operation is requested
                # if item in _qubit__all__ and item in _cv__all__:
                #     raise AttributeError("Variance operator exists for both CV and Qubit circuit. "
                #                          "Please specify by using either pennylane.var.cv.Name or "
                #                          "pennylane.var.qubit.Name.")

                if item in _qubit__all__:
                    expval_class = getattr(qubit, item)
                else:
                    expval_class = getattr(cv, item)
        else:
            # the variance was called via the cv or qubit attribute
            if self.observables == _qubit__all__:
                expval_class = getattr(qubit, item)
            elif self.observables == _cv__all__:
                expval_class = getattr(cv, item)

        # return a class inheriting from the expectation class,
        # but with return type now set to variance
        docstring = expval_class.__doc__.replace('expval', 'var').replace('expectation', 'variance')
        return type(item, (expval_class,), {"return_type": "variance", "__doc__": docstring})

    def __dir__(self):
        return self.observables
