# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.template.decorator` module.
Integration tests should be placed into ``test_templates.py``.
"""
import pennylane as qml
from pennylane.templates.decorator import template

def expected_queue(wires):
    return [qml.RX(2 * i, wires=[wire]) for i, wire in enumerate(wires)] + [qml.RY(3 * i, wires=[wire]) for i, wire in enumerate(wires)]

def dummy_template(wires):
    for i, wire in enumerate(wires):
        qml.RX(2 * i, wires=[wire])

    for i, wire in enumerate(wires):
        qml.RY(3 * i, wires=[wire])

@template 
def decorated_dummy_template(wires):
    for i, wire in enumerate(wires):
        qml.RX(2 * i, wires=[wire])

    for i, wire in enumerate(wires):
        qml.RY(3 * i, wires=[wire])

class TestDecorator:
    """Tests the template decorator."""

    def test_dummy_template(self):
        """Test the decorator for a dummy template."""
        @template
        def my_template(wires):
            dummy_template(wires)

        res = my_template([0, 1])
        expected = expected_queue([0, 1])

        for res_op, exp_op in zip(res, expected):
            assert res_op.name == exp_op.name
            assert res_op.wires == exp_op.wires
            assert res_op.params == exp_op.params

    def test_decorated_dummy_template(self):
        """Test the decorator for a already decorated template."""
        res = decorated_dummy_template([0, 1])

        expected = expected_queue([0, 1])

        for res_op, exp_op in zip(res, expected):
            assert res_op.name == exp_op.name
            assert res_op.wires == exp_op.wires
            assert res_op.params == exp_op.params

    def test_decorated_decorated_dummy_template(self):
        """Test the decorator for decorating an already decorated template."""
        @template
        def my_template(wires):
            decorated_dummy_template(wires)

        res = my_template([0, 1])
        expected = expected_queue([0, 1])

        for res_op, exp_op in zip(res, expected):
            assert res_op.name == exp_op.name
            assert res_op.wires == exp_op.wires
            assert res_op.params == exp_op.params