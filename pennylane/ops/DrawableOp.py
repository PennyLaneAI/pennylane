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
"""
This module contains the DrawableOperation (pseudo) operation that helps with
circuit drawing.
"""
from pennylane.wires import Wires
from pennylane.operation import Operation


class _DrawableOperation(Operation):

    def __init__(self, wires=Wires([]), control_wires=None, target_wires=None, label=None, do_queue=None, id=None):
        self._label = label
        self.cntrl_wires = control_wires
        self.target_wires = target_wires
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    def label(self, *args, **kwargs):
        return "Block_U" if self._label is None else self._label
