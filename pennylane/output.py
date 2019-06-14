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
# pylint: disable=protected-access
r"""
Output from a quantum system - expval, measure and var
======================================================

**Module name:** :mod:`pennylane.output`

.. currentmodule:: pennylane.output

Functions
---------

This module contains the functions for computing expectation values,
variances, and single shot measurements of quantum observables.

Summary
^^^^^^^

.. autosummary::
   expval
   measure
   var

Code details
^^^^^^^^^^^^
"""
import pennylane as qml

import warnings

from .qnode import QNode, QuantumFunctionError
from .operation import Observable


class ExpvalFactory:
    r"""pennylane.output.expval()
    Expectation value of the operator supplied :math:`\langle O \rangle`.

    Args:
        op (:class:`~pennylane.operation.Operation`): A quantum operator 
        object which is run inside a `QNode`, e.g., `PauliX(wires=[0])`
    """

    def __call__(self, op):
        if QNode._current_context is None:
            raise QuantumFunctionError("Expectation values can only be used inside a qfunc.")

        if not isinstance(op, Observable):
            raise QuantumFunctionError("Only observables can be accepted")

        # delete operations from QNode queue
        QNode._current_context.queue.remove(op)

        # set return type to be an expectation value
        op.return_type = "expectation"

        # add observable to QNode observable queue
        QNode._current_context._append_op(op)

        return op

    def __getattr__(self, name):
        warnings.warn("Calling pennylane.expval.Observable() is deprecated. "
                      "Please use the new pennylane.expval(qml.Observable()) form.")

        if name == "__all__":
            op_list = []

            for op in qml.ops.__all__:
                obs_class = getattr(qml.ops, op)

                if issubclass(obs_class, Observable):
                    op_list.append(op)

            return op_list

        if name == "_cv__all__":
            op_list = []

            for op in qml.ops.__all__:
                obs_class = getattr(qml.ops, op)

                if issubclass(obs_class, CVObservable):
                    op_list.append(op)

            return op_list

        if name == "_qubit__all__":
            op_list = []

            for op in qml.ops.__all__:
                obs_class = getattr(qml.ops, op)

                if issubclass(obs_class, Observable) and not issubclass(obs_class, CV):
                    op_list.append(op)

            return op_list

        if name in qml.ops.__all__:
            obs_class = getattr(qml.ops, name)
            return type(name, (obs_class,), {"return_type": "expectation"})

    __getattribute__ = __getattr__


expval = ExpvalFactory()
