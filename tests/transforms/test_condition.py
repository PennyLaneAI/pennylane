# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.transforms.condition import ConditionalTransformError


class TestCond:
    """Tests that verify that the cond transform works as expect."""

    def test_cond_error(self):
        """Test that a qfunc can also used with qml.cond even when an else
        qfunc is provided."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            return qml.state()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, f)()

    def test_cond_error_else(self):
        """Test that a qfunc can also used with qml.cond even when an else
        qfunc is provided."""
        dev = qml.device("default.qubit", wires=3)

        def f():
            qml.PauliX(0)

        def g():
            return qml.state()

        with pytest.raises(
            ConditionalTransformError, match="contain no measurements can be applied conditionally"
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, f, g)()

    @pytest.mark.parametrize("inp", [1, "string", qml.PauliZ(0)])
    def test_cond_error_unrecognized_input(self, inp):
        """Test that a qfunc can also used with qml.cond even when an else
        qfunc is provided."""
        dev = qml.device("default.qubit", wires=3)

        with pytest.raises(
            ConditionalTransformError,
            match="Only operations and quantum functions with no measurements",
        ):
            m_0 = qml.measure(1)
            qml.cond(m_0, inp)()
