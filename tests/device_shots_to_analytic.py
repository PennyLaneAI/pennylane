# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This device wraps a target device and makes every execution analytic.

It can be used to eliminate flakiness in shot vector tests.
"""
import copy
from types import MethodType

import pennylane as qml


def shots_to_analytic(dev: qml.devices.Device) -> qml.devices.Device:
    """Make all executions analytic to prevent flakiness in tests.

    .. code-block:: python

        @qml.qnode(shots_to_analytic(qml.device('default.qubit', shots=(1,1,1))))
        def f(x):
            qml.RX(2*x, 0)
            return qml.expval(qml.Z(0))

        f(0.5)

    .. code-block::

        (0.5403023058681398, 0.5403023058681398, 0.5403023058681398)

    """

    analytic_dev = copy.deepcopy(dev)
    original_execute = analytic_dev.execute

    # pylint: disable=unused-argument
    def new_execute(self, circuits, execution_config=None):
        execution_config = execution_config or qml.devices.ExecutionConfig()
        results = []
        for c in circuits:
            if c.shots.has_partitioned_shots:
                res = original_execute(c.copy(shots=None), execution_config)
                results.append([res] * c.shots.num_copies)
            else:
                results.append(original_execute(c.copy(shots=None), execution_config))
        return tuple(results)

    analytic_dev.execute = MethodType(new_execute, dev)
    return analytic_dev
