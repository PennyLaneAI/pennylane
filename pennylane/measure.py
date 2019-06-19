# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
r"""
Measurements
============

**Module name:** :mod:`pennylane.measure`

.. currentmodule:: pennylane.measure

This module contains the functions for computing expectation values,
variances, and single shot measurements of quantum observables.

Summary
^^^^^^^

.. autosummary::
   expval

Code details
^^^^^^^^^^^^
"""
import warnings

import pennylane as qml

from .qnode import QNode, QuantumFunctionError
from .operation import Observable


class ExpvalFactory:
    r"""Expectation value of the supplied observable.

    Args:
        op (Observable): a quantum observable object
    """

    def __call__(self, op):
        if not isinstance(op, Observable):
            raise QuantumFunctionError(
                "{} is not an observable: cannot be used with expval".format(op.name)
            )

        if QNode._current_context is not None:
            # delete operations from QNode queue
            QNode._current_context.queue.remove(op)

        # set return type to be an expectation value
        op.return_type = "expectation"

        if QNode._current_context is not None:
            # add observable to QNode observable queue
            QNode._current_context._append_op(op)

        return op

    def __getattr__(self, name):
        # This to allow backwards compatibility with the previous
        # module/attribute UI for requesting expectation values. Note that a deprecation
        # warning will be raised if this UI is used.
        #
        # Once fully deprecated, this method can be removed,
        # and the ExpvalFactory function can be converted into a
        # simple expval() function using the code within __call__.
        warnings.warn(
            "Calling qml.expval.Observable() is deprecated. "
            "Please use the new qml.expval(qml.Observable()) form.",
            DeprecationWarning,
        )

        if name in qml.ops.__all_obs__:  # pylint: disable=no-member
            obs_class = getattr(qml.ops, name)
            return type(name, (obs_class,), {"return_type": "expectation"})

        if name in qml.ops.__all_ops__:  # pylint: disable=no-member
            raise AttributeError("{} is not an observable: cannot be used with expval".format(name))

        raise AttributeError("module 'pennylane' has no observable '{}'".format(name))


expval = ExpvalFactory()
r"""Expectation value of the supplied observable.

Args:
    op (Observable): a quantum observable object
"""
