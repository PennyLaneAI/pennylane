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
    def __init__(self, op, exponent, do_queue=True, id=None):

        self.hyperparameters["base"] = op
        self.hyperparameters["exponent"] = exponent

        super().__init__(*op.parameters, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"Pow({op}, {exponent})"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.hyperparameters['base']}**{self.hyperparameters['exponent']}"

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, exponent=None, base=None, **hyperparameters):

        if exponent == 0:
            return []
        if isinstance(exponent, int):
            if exponent > 0:
                return [base] * exponent
            if exponent < 0:
                return [qml.inverse(base)] * abs(
                    exponent
                )  # is this correct? A^(-n) = A^(-1)A^(-1)...

        if exponent == 0.5:
            return [qml.ops.math.Sqrt(base)]
        else:
            raise ValueError(
                "Taking non-integer-valued powers of operators other than 1/2 (i.e., the square root) "
                "is currently not supported."
            )

    @staticmethod
    def compute_matrix(*params, exponent=None, base=None, **hyperparams):
        return base.get_matrix() ** exponent  # check if this covers all cases


def pow(
    op,
    exponent,
):
    return Pow(op, exponent)
