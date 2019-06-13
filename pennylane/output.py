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

This module contains the functions for computing expectation values of
quantum operations, variances and making single shot measurements.

* The :function:`pennylane.output.expval` can be used to get the expectation
  value of any quantum operation defined in `pennylane.ops`. For example a
  unitary quantum gate such as `PauliX`.

* The :function:`pennylane.outputs.measure` makes a single shot measurement of
  some quantum operation and generates a sample.

* The :function:`pennylane.outputs.var` computes the variance of the
  expectation value for a quantum operation.

Summary
^^^^^^^

.. autosummary::
   expval
   measure
   var

Code details
^^^^^^^^^^^^
"""
from pennylane.operation import Operation
import pennylane as pl
import pennylane.ops as ops


def expval(op):
    r"""pennylane.output.expval()
    Expectation value of the operator supplied :math:`\langle O \rangle`.

    .. math::
        E[O] = \text{Tr}(O \rho)

    Args:
        op (:class:`~pennylane.operation.Operation`): A quantum operator 
        object which is run inside a `QNode`, e.g., `PauliX(wires=[0])`
    """
    if not isinstance(op, Operation):
        msg = "Invalid operator for expectation. "
        msg += "Please use operations defined in `pennylane.ops` subclassing "
        msg += "`pennylane.operation.Operation`"

        raise ValueError(msg)

    if op.name in pl.expval.__all__:
        return getattr(pl.expval, op.name)(*op.params, wires=op.wires)
