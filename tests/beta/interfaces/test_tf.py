# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the tf interface"""
import pytest
import numpy as np
import tensorflow as tf

import pennylane as qml
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import expval, var, sample, probs
from pennylane.beta.interfaces.tf import TFInterface


class TestTFQuantumTape:
    """Test the autograd interface applied to a tape"""

    def test_interface_str(self):
        """Test that the interface string is correctly identified as tf"""
        with TFInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "tf"
        assert isinstance(tape, TFInterface)

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = tf.Variable(0.1)
        b = tf.constant(0.2)
        c = tf.Variable(0.3)
        d = tf.constant(0.4)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.Rot(a, b, c, wires=0)
                qml.RX(d, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliX(0))

        assert qtape.trainable_params == {0, 2}
        assert np.all(qtape.get_parameters() == [a, c])
