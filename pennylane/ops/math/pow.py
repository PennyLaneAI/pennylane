# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-few-public-methods,function-redefined

import pennylane as qml


class Pow(qml.operation.Operator):
    """Arithmetic operator subclass representing the power of an operator."""

    def __init__(self, op, exponent, do_queue=True, id=None):

        self.hyperparameters["base_op"] = op
        self.hyperparameters["exponent"] = exponent

        super().__init__(*op.parameters, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"Pow({op}, {exponent})"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.hyperparameters['base_op']}**{self.hyperparameters['exponent']}"

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, exponent=None, base_op=None, **hyperparameters):

        if exponent == 0:
            return []
        if isinstance(exponent, int):
            if exponent > 0:
                return [base_op] * exponent
            if exponent < 0:
                return [qml.inverse(base_op)] * abs(
                    exponent
                )  # is this correct? A^(-n) = A^(-1)A^(-1)...

        return qml.operation.DecompositionUndefinedError

    @staticmethod
    def compute_matrix(*params, exponent=None, base_op=None, **hyperparams):
        return base_op.get_matrix() ** exponent  # check if this works for non-integer exponents


def pow(
    op,
    exponent,
):
    try:
        # there is a custom power defined
        return op.pow(exponent)
    except AttributeError:
        # default to an abstract arithmetic class
        return Pow(op, exponent)
