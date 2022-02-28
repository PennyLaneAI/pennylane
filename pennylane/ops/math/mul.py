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


class ScalarMul(qml.operation.Operator):

    def __init__(self, op, scalar, do_queue=True, id=None):
        self.hyperparameters['scalar'] = scalar
        self.hyperparameters['op'] = op

        super().__init__(*op.parameters, scalar, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"{scalar}  {op.name}"

    @property
    def num_wires(self):
        return len(self.wires)

    @classmethod
    def compute_terms(cls, *params, **hyperparams):
        return [hyperparams["scalar"]], [hyperparams["op"]]
