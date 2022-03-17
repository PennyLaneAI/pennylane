# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

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
This module contains entropy functions.
"""
import pennylane as qml
import numpy as np
from functools import wraps
from scipy import linalg as la


def entropy(circuit, subsystem=None, base=2):
    @wraps(circuit)
    def wrapper(*args, **kwargs):
        if isinstance(circuit, qml.QNode):
            circuit(*args, **kwargs)

            dev = circuit.device
            state = circuit.device._state

            # Generalize to all devices.
            if dev.short_name != "default.mixed":
                entropy = 0
                return entropy

            if subsystem:
                density_matrix = dev.density_matrix(wires=qml.wires.Wires(subsystem))
                dev._state = state
            else:
                density_matrix = dev.state

            evs = la.eigvals(density_matrix)

            entropy = 0
            for ev in evs:
                ev = np.maximum(0, np.real(ev))
                if not np.allclose(ev, 0, atol=1e-5):
                    entropy += -ev * np.log2(ev)

            return entropy
        else:
            raise qml.QuantumFunctionError("Must provide a QNode")
        return wrapper

    return wrapper


def relative_entropy(circuit_1, circuit_2):
    @wraps(circuit_1, circuit_2)
    def wrapper(*args, **kwargs):
        if isinstance(circuit_1, qml.QNode):
            circuit_1(*args, **kwargs)

            dev = circuit_1.device
            state = circuit_1.device._state

            if dev.short_name != "default.mixed":
                entropy = 0
                return entropy

            return relative_entropy
        else:
            raise qml.QuantumFunctionError("Must provide a QNode")
        return wrapper

    return wrapper


def mutual_information(state_1, state_2):
    return 1
