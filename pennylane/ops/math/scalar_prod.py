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


class ScalarProd(qml.operation.Operator):
    """Arithmetic operator subclass representing the scalar product of an operator."""

    def __init__(self, scalar, op, do_queue=True, id=None):
        self.hyperparameters["scalar"] = scalar
        self.hyperparameters["op"] = op

        super().__init__(*op.parameters, scalar, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"{scalar}  {op.name}"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.hyperparameters['scalar']} {self.hyperparameters['op']}"

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_terms(*params, scalar=None, op=None, **hyperparams):
        return [scalar], [op]

    @staticmethod
    def compute_matrix(*params, scalar=None, op=None, **hyperparams):
        return scalar * op.get_matrix()

    @staticmethod
    def compute_eigvals(*params, scalar=None, op=None, **hyperparams):
        return scalar * op.get_eigvals()


def scalar_prod(scalar, op):
    return ScalarProd(scalar, op)
