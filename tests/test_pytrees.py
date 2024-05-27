# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the pennylane pytrees module
"""

import pytest

import pennylane as qml
from pennylane.pytrees import PyTreeStructure, flatten, leaf, register_pytree, unflatten


def test_structure_repr_str():
    """Test the repr of the structure class."""
    op = qml.RX(0.1, wires=0)
    _, structure = qml.pytrees.flatten(op)
    expected = "PyTreeStructure(RX, (<Wires = [0]>, ()), [PyTreeStructure()])"
    assert repr(structure) == expected
    expected_str = "PyTree(RX, (<Wires = [0]>, ()), [Leaf])"
    assert str(structure) == expected_str


def test_register_new_class():
    """Test that new objects can be registered, flattened, and unflattened."""

    # pylint: disable=too-few-public-methods
    class MyObj:
        """a dummy object."""

        def __init__(self, a):
            self.a = a

    def obj_flatten(obj):
        return (obj.a,), None

    def obj_unflatten(data, _):
        return MyObj(data[0])

    register_pytree(MyObj, obj_flatten, obj_unflatten)

    obj = MyObj(0.5)

    data, structure = flatten(obj)
    assert data == [0.5]
    assert structure == PyTreeStructure(MyObj, None, [leaf])

    new_obj = unflatten([1.0], structure)
    assert isinstance(new_obj, MyObj)
    assert new_obj.a == 1.0


def test_list():
    """Test that pennylane treats list as a pytree."""

    x = [1, 2, [3, 4]]

    data, structure = flatten(x)
    assert data == [1, 2, 3, 4]
    assert structure == PyTreeStructure(
        list, None, [leaf, leaf, PyTreeStructure(list, None, [leaf, leaf])]
    )

    new_x = unflatten([5, 6, 7, 8], structure)
    assert new_x == [5, 6, [7, 8]]


def test_tuple():
    """Test that pennylane can handle tuples as pytrees."""
    x = (1, 2, (3, 4))

    data, structure = flatten(x)
    assert data == [1, 2, 3, 4]
    assert structure == PyTreeStructure(
        tuple, None, [leaf, leaf, PyTreeStructure(tuple, None, [leaf, leaf])]
    )

    new_x = unflatten([5, 6, 7, 8], structure)
    assert new_x == (5, 6, (7, 8))


def test_dict():
    """Test that pennylane can handle dictionaries as pytees."""

    x = {"a": 1, "b": {"c": 2, "d": 3}}

    data, structure = flatten(x)
    assert data == [1, 2, 3]
    assert structure == PyTreeStructure(
        dict, ("a", "b"), [leaf, PyTreeStructure(dict, ("c", "d"), [leaf, leaf])]
    )
    new_x = unflatten([5, 6, 7], structure)
    assert new_x == {"a": 5, "b": {"c": 6, "d": 7}}


@pytest.mark.usefixtures("new_opmath_only")
def test_nested_pl_object():
    """Test that we can flatten and unflatten nested pennylane object."""

    tape = qml.tape.QuantumScript(
        [qml.adjoint(qml.RX(0.1, wires=0))],
        [qml.expval(2 * qml.X(0))],
        shots=50,
        trainable_params=(0, 1),
    )

    data, structure = flatten(tape)
    assert data == [0.1, 2, None]

    wires0 = qml.wires.Wires(0)
    op_structure = PyTreeStructure(
        tape[0].__class__, (), [PyTreeStructure(qml.RX, (wires0, ()), [leaf])]
    )
    list_op_struct = PyTreeStructure(list, None, [op_structure])

    sprod_structure = PyTreeStructure(
        qml.ops.SProd, (), [leaf, PyTreeStructure(qml.X, (wires0, ()), [])]
    )
    meas_structure = PyTreeStructure(
        qml.measurements.ExpectationMP, (("wires", None),), [sprod_structure, leaf]
    )
    list_meas_struct = PyTreeStructure(list, None, [meas_structure])
    tape_structure = PyTreeStructure(
        qml.tape.QuantumScript,
        (tape.shots, tape.trainable_params),
        [list_op_struct, list_meas_struct],
    )

    assert structure == tape_structure

    new_tape = unflatten([3, 4, None], structure)
    expected_new_tape = qml.tape.QuantumScript(
        [qml.adjoint(qml.RX(3, wires=0))],
        [qml.expval(4 * qml.X(0))],
        shots=50,
        trainable_params=(0, 1),
    )
    assert qml.equal(new_tape, expected_new_tape)
