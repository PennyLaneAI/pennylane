# Copyright 2025 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Nodes for use in qcut.
"""

import uuid
import warnings

from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.operation import Operation


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id: str | None = None, node_uid: str | None = None):
        if id is not None:
            warnings.warn(
                "The 'id' kwarg has been renamed to 'node_uid'. Access through 'id' will be removed in v0.46.",
                PennyLaneDeprecationWarning,
            )
            # Only override if meas_uid wasn't explicitly provided
            if node_uid is None:
                node_uid = id
        self._node_uid: str = node_uid or str(uuid.uuid4())

        super().__init__(wires=wires)

    @property
    def node_uid(self) -> str:
        """Custom UID for this node."""
        return self._node_uid

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id: str | None = None, node_uid: str | None = None):
        if id is not None:
            warnings.warn(
                "The 'id' kwarg has been renamed to 'node_uid'. Access through 'id' will be removed in v0.46.",
                PennyLaneDeprecationWarning,
            )
            # Only override if meas_uid wasn't explicitly provided
            if node_uid is None:
                node_uid = id
        self._node_uid: str = node_uid or str(uuid.uuid4())

        super().__init__(wires=wires)

    @property
    def node_uid(self) -> str:
        """Custom UID for this node."""
        return self._node_uid

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label
