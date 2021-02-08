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
"""Unit tests for the JAX interface"""
import pytest
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.tape import JacobianTape, qnode, QNode, QubitParamShiftTape


def test_qnode_intergration():
	dev = qml.device("default.mixed", wires=2) # A non-JAX device

	@qml.qnode(dev, interface="jax")
	def circuit(weights):
		qml.RX(weights[0], wires=0)
		qml.RZ(weights[0], wires=1)
		return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

	weights = jnp.array([0.1, 0.2])
	val = circuit(weights)
	assert "DeviceArray" in val.__repr__()

