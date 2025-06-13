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
"""
Tests for the allocation module.
"""
import uuid

import pytest

import pennylane as qml
from pennylane.allocation import (
    Allocate,
    Deallocate,
    DynamicWire,
    _get_allocate_prim,
    _get_deallocate_prim,
    allocate,
    deallocate,
    safe_allocate,
)


class TestDynamicWire:

    def test_uuid_intialization(self):
        """Test that a uuid is created upon initialization."""
        w = DynamicWire()
        assert isinstance(w.key, uuid.UUID)

        w2 = DynamicWire(w.key)
        assert w2.key == w.key

    def test_dynamic_wire_repr(self):
        """Test the repr of a Dynamic Wire"""
        assert repr(DynamicWire()) == "<DynamicWire>"

    def test_equality(self):
        """Test that equality can be determined based on key."""

        a = DynamicWire()
        assert a != 2

        b = DynamicWire()
        assert a != b

        c = DynamicWire(key=a.key)
        assert a == c

    def test_hash(self):
        """Test that the hash of the dynamic wire depends on the uuid."""

        a = DynamicWire()
        b = DynamicWire()
        c = DynamicWire(key=a.key)
        assert len({a, b, c}) == 2
        assert c in {a, b}
        assert b not in {a}


class TestAllocateOp:

    def test_valid_operation(self):
        """Test that Allocate is a valid Operator."""
        op = Allocate.from_num_wires(3)
        qml.ops.functions.assert_valid(op)

    def test_allocate_from_num_wires(self):
        """Test that the op can be instantiated with from_num_wires"""

        op = Allocate.from_num_wires(3, require_zeros=False)
        assert len(op.wires) == 3
        assert op.hyperparameters == {"require_zeros": False}
        assert not op.require_zeros

    def test_normal_initialization(self):
        """Test that the op can also be initialized with already created dynamic wires."""
        wires = [DynamicWire() for _ in range(5)]
        op = Allocate(wires, require_zeros=False)
        assert op.wires == qml.wires.Wires(wires)
        assert op.hyperparameters == {"require_zeros": False}
        assert not op.require_zeros

    def test_default_hyperparameters(self):
        """Test the values of the default hyperparameters."""

        op = Allocate.from_num_wires(2)
        assert op.require_zeros

        op2 = Allocate(DynamicWire())
        assert op2.require_zeros


def test_Deallocate_validity():
    """Test that Deallocate is a valid operation."""
    wires = [DynamicWire(), DynamicWire()]
    op = Deallocate(wires, reset_to_original=True)
    assert op.reset_to_original
    assert op.hyperparameters == {"reset_to_original": True}
    assert op.wires == qml.wires.Wires(wires)
    qml.ops.functions.assert_valid(op)


def test_allocate_function():
    """Test that allocate returns dynamic wires and queues an Allocate op."""
    with qml.queuing.AnnotatedQueue() as q:
        wires = allocate(4)
    assert isinstance(wires, qml.wires.Wires)
    assert len(wires) == 4
    assert all(isinstance(w, DynamicWire) for w in wires)

    assert len(q) == 1
    op = q.queue[0]
    assert op.wires == wires
    assert op.require_zeros


def test_allocate_kwargs():
    """Test that the kwargs to allocate get passed to the op."""

    with qml.queuing.AnnotatedQueue() as q:
        allocate(3, require_zeros=False)

    op = q.queue[0]
    assert not op.require_zeros


class TestDeallocate:

    def test_single_dynamic_wire(self):
        """Test that deallocate can accept a single dynamic wire."""
        wire = DynamicWire()
        with qml.queuing.AnnotatedQueue() as q:
            op = deallocate(wire)
        assert op.wires == qml.wires.Wires((wire,))

        assert len(q.queue) == 1
        assert op is q.queue[0]

    def test_error_non_dynamic_wire(self):
        """Test that an error is raised if a non-dynamic wire is attempted to be deallocated."""
        with pytest.raises(ValueError, match="only accept DynamicWire wires."):
            deallocate((DynamicWire, 1))

    def test_multiple_dynamic_wires(self):
        """Test multiple dynamic wires can be deallocated."""

        wires = [DynamicWire(), DynamicWire()]
        with qml.queuing.AnnotatedQueue() as q:
            op = deallocate(wires)
        assert op.wires == qml.wires.Wires(wires)

        assert len(q.queue) == 1
        assert op is q.queue[0]

    def test_reset_to_original(self):
        """Test that reset_to_original is passed to the operator."""

        op = deallocate(DynamicWire(), reset_to_original=True)
        assert op.reset_to_original
        assert op.hyperparameters == {"reset_to_original": True}


def test_safe_allocate():
    """Test that safe_allocate allocates and deallocates qubits."""

    with qml.queuing.AnnotatedQueue() as q:
        with safe_allocate(3, require_zeros=False, reset_to_original=True) as wires:
            assert len(wires) == 3
            assert all(isinstance(w, DynamicWire) for w in wires)
            assert len(set(wires)) == 3

            qml.I(wires)

    assert len(q.queue) == 3
    qml.assert_equal(q.queue[0], Allocate(wires, require_zeros=False))
    qml.assert_equal(q.queue[1], qml.I(wires))
    qml.assert_equal(q.queue[2], Deallocate(wires, reset_to_original=True))


@pytest.mark.jax
@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize("use_context", (True, False))
def test_capturing_allocate_and_deallocate(use_context):
    """Test that allocate and deallcoate can be captured."""

    import jax

    def f():
        if use_context:
            with safe_allocate(2, require_zeros=True, reset_to_original=True) as wires:
                qml.H(wires[0])
                qml.Z(wires[1])
        else:
            w, w2 = allocate(2, require_zeros=True)
            qml.H(w)
            qml.Z(w2)
            deallocate((w, w2), reset_to_original=True)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 4
    assert jaxpr.eqns[0].primitive == _get_allocate_prim()
    assert len(jaxpr.eqns[0].invars) == 0
    assert jaxpr.eqns[0].params == {"num_wires": 2, "require_zeros": True}
    assert len(jaxpr.eqns[0].outvars) == 2
    assert all(v.aval.shape == () for v in jaxpr.eqns[0].outvars)
    for v in jaxpr.eqns[0].outvars:
        assert v.aval.dtype == jax.numpy.int64

    assert jaxpr.eqns[1].invars[0] is jaxpr.eqns[0].outvars[0]
    assert jaxpr.eqns[2].invars[0] is jaxpr.eqns[0].outvars[1]

    assert jaxpr.eqns[3].primitive == _get_deallocate_prim()
    assert jaxpr.eqns[3].params == {"reset_to_original": True}
    assert jaxpr.eqns[3].invars == jaxpr.eqns[0].outvars

    with pytest.raises(NotImplementedError):
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    with pytest.raises(NotImplementedError):
        deallocate(2)
