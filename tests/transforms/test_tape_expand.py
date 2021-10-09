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
Unit tests for tape expansion stopping criteria and expansion functions.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane.transforms.tape_expand import StoppingCriterion, get_expand_fn


class TestStoppingCriterion:
    @pytest.mark.parametrize(
        "fn, arg, expected",
        [
            (lambda x: True, -1, True),
            (lambda x: True, 10, True),
            (lambda x: x < 4, -1, True),
            (lambda x: x < 4, 10, False),
        ],
    )
    def test_basic_functionality(self, fn, arg, expected):
        """Test initialization and calling of StoppingCriterion."""
        crit = StoppingCriterion(fn)
        assert crit(arg) == expected

    def test_not(self):
        """Test that logical negation works."""
        crit = StoppingCriterion(lambda x: x < 4)
        ncrit = ~crit
        assert crit(-2) and not ncrit(-2)
        assert not crit(10) and ncrit(10)

    def test_and(self):
        """Test that logical conjunction works."""
        crit_0 = StoppingCriterion(lambda x: x > 4)
        crit_1 = StoppingCriterion(lambda x: x < 9)
        crit = crit_0 & crit_1
        assert not crit(-2)
        assert crit(6)
        assert not crit(10)

    def test_or(self):
        """Test that logical or works."""
        crit_0 = StoppingCriterion(lambda x: x < 4)
        crit_1 = StoppingCriterion(lambda x: x > 9)
        crit = crit_0 | crit_1
        assert crit(-2)
        assert not crit(6)
        assert crit(10)


class TestCriteria:

    rx = qml.RX(qml.numpy.array(0.3, requires_grad=True), wires=1)
    stiff_rx = qml.RX(0.3, wires=1)
    cnot = qml.CNOT(wires=[1, 0])
    rot = qml.Rot(*qml.numpy.array([0.1, -0.7, 0.2], requires_grad=True), wires=0)
    stiff_rot = qml.Rot(0.1, -0.7, 0.2, wires=0)
    exp = qml.expval(qml.PauliZ(0))

    def test_has_generator(self):
        """Test has_generator criterion."""
        assert qml.transforms.has_generator(self.rx)
        assert not qml.transforms.has_generator(self.cnot)
        assert not qml.transforms.has_generator(self.rot)
        assert not qml.transforms.has_generator(self.exp)

    def test_has_grad_method(self):
        """Test has_grad_method criterion."""
        assert qml.transforms.has_grad_method(self.rx)
        assert qml.transforms.has_grad_method(self.rot)
        assert not qml.transforms.has_grad_method(self.cnot)

    def test_has_multiple_params(self):
        """Test has_multiple_params criterion."""
        assert not qml.transforms.has_multiple_params(self.rx)
        assert qml.transforms.has_multiple_params(self.rot)
        assert not qml.transforms.has_multiple_params(self.cnot)

    def test_has_no_params(self):
        """Test has_no_params criterion."""
        assert not qml.transforms.has_no_params(self.rx)
        assert not qml.transforms.has_no_params(self.rot)
        assert qml.transforms.has_no_params(self.cnot)

    def test_has_unitary_generator(self):
        """Test has_unitary_generator criterion."""
        assert qml.transforms.has_unitary_generator(self.rx)
        assert not qml.transforms.has_unitary_generator(self.rot)
        assert not qml.transforms.has_unitary_generator(self.cnot)

    def test_is_measurement(self):
        """Test is_measurement criterion."""
        assert not qml.transforms.is_measurement(self.rx)
        assert not qml.transforms.is_measurement(self.rot)
        assert not qml.transforms.is_measurement(self.cnot)
        assert qml.transforms.is_measurement(self.exp)

    def test_is_trainable(self):
        """Test is_trainable criterion."""
        assert qml.transforms.is_trainable(self.rx)
        assert not qml.transforms.is_trainable(self.stiff_rx)
        assert qml.transforms.is_trainable(self.rot)
        assert not qml.transforms.is_trainable(self.stiff_rot)
        assert not qml.transforms.is_trainable(self.cnot)


class TestGetExpandFn:

    crit_0 = (~qml.transforms.is_trainable) | (
        qml.transforms.has_generator & qml.transforms.is_trainable
    )
    with qml.tape.JacobianTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RY(qml.numpy.array(2.1, requires_grad=True), wires=1)
        qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

    def test_get_expand_fn(self):
        """Test creation of expand_fn."""
        get_expand_fn(depth=10, stop_at=self.crit_0)

    def test_get_expand_fn_expansion(self):
        """Test expansion with created expand_fn."""
        expand_fn = get_expand_fn(depth=10, stop_at=self.crit_0)
        new_tape = expand_fn(self.tape)
        assert new_tape.operations[0] == self.tape.operations[0]
        assert new_tape.operations[1] == self.tape.operations[1]
        assert [op.name for op in new_tape.operations[2:]] == ["RZ", "RY", "RZ"]
        assert np.allclose([op.data for op in new_tape.operations[2:]], [[0.5], [0.2], [-0.1]])
        assert [op.wires for op in new_tape.operations[2:]] == [qml.wires.Wires(0)] * 3

    def test_get_expand_fn_dont_expand(self):
        """Test expansion is skipped with depth=0."""
        expand_fn = get_expand_fn(depth=0, stop_at=self.crit_0)

        new_tape = expand_fn(self.tape)
        assert new_tape.operations == self.tape.operations
