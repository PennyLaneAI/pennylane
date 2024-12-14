# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
A transform for decomposing quantum circuits into user-defined gate sets. Offers an alternative to the more device-focused decompose transform.
"""
# pylint: disable=protected-access
# pylint: disable=unnecessary-lambda-assignment

import warnings
from collections.abc import Iterable

import pennylane as qml
from pennylane.capture.primitives import ctrl_transform_prim
from pennylane.transforms.decompose import _operator_decomposition_gen


class DecomposeInterpreter(qml.capture.PlxprInterpreter):
    """Plxpr Interpreter for applying the ``decompose`` transform to callables or jaxpr
    when program capture is enabled.
    """

    def __init__(self, gate_set=None, max_expansion=None):
        self.max_expansion = max_expansion

        if gate_set is None:
            gate_set = set(qml.ops.__all__)

        if isinstance(gate_set, (str, type)):
            gate_set = set([gate_set])

        if isinstance(gate_set, Iterable):
            gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
            gate_names = set(gate for gate in gate_set if isinstance(gate, str))
            self.gate_set = lambda op: (op.name in gate_names) or isinstance(op, gate_types)
        else:
            self.gate_set = gate_set

        super().__init__()

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Function to determine whether or not an operator needs to be decomposed or not.

        Args:
            op (qml.operation.Operator): Operator to check.

        Returns:
            bool: Whether or not ``op`` is valid or needs to be decomposed. ``True`` means
            that the operator does not need to be decomposed.
        """
        if not op.has_decomposition:
            if not self.gate_set(op):
                warnings.warn(
                    f"Operator {op.name} does not define a decomposition and was not "
                    f"found in the target gate set. To remove this warning, add the operator name "
                    f"({op.name}) or type ({type(op)}) to the gate set.",
                    UserWarning,
                )
            return True
        return self.gate_set(op)

    def decompose_operation(self, op: qml.operation.Operator):
        """Decompose a PennyLane operation instance if it does not satisfy the
        provided gate set.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`, :meth:`~.interpret_operation`.
        """
        if self.gate_set(op):
            return self.interpret_operation(op)

        qml.capture.disable()
        try:
            decomposition = list(
                _operator_decomposition_gen(
                    op, self.stopping_condition, max_expansion=self.max_expansion
                )
            )
        finally:
            qml.capture.enable()

        return [self.interpret_operation(decomp_op) for decomp_op in decomposition]

    def interpret_operation_eqn(self, eqn):
        """Interpret an equation corresponding to an operator.

        Args:
            eqn (jax.core.JaxprEqn): a jax equation for an operator.

        See also: :meth:`~.interpret_operation`.

        """
        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            op = eqn.primitive.impl(*invals, **eqn.params)
        if eqn.outvars[0].__class__.__name__ == "DropVar":
            return self.decompose_operation(op)
        return op


@DecomposeInterpreter.register_primitive(ctrl_transform_prim)
def handle_ctrl_transform(*_, **__):  # pylint: disable=missing-function-docstring
    raise NotImplementedError
