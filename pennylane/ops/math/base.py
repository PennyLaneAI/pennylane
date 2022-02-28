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


class OpWrapper(qml.operation.Operator):
    num_wires = qml.operation.AnyWires

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters["base"] = base
        self.hyperparameters.update(base.hyperparameters)
        super().__init__(*base.parameters, wires=base.wires, do_queue=do_queue, id=id)
        self._name = f"{self.__class__.__name__}({self.base.name})"

    @property
    def num_wires(self):
        return self.base.num_wires

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def wires(self):
        return self.base.wires
