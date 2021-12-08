# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains functionality to enable task-based workflows with devices.
"""
import numpy as np
from typing import Dict, List, Union

import pennylane as qml
from pennylane.devices import TaskQubit

def taskify(dev: qml.Device, return_future: bool = False, gen_report: Union[bool, str] = False):
    """
    Returns a proxy-qubit device with the device argument as the intiantiable backend.

    >>> d_dev = qml.device("default.qubit", wires=["a","b",2])
    >>> t_dev = qml.taskify(dev)
    >>> <TaskQubit device (wires=3, shots=None) at 0x7f66fcbbeee0>
    """
    return TaskQubit(dev.wires, 
                    shots=dev.shots, 
                    analytic=None, 
                    backend=dev.short_name, 
                    gen_report=gen_report, 
                    future=return_future
    )