# Copyright 2025 Xanadu Quantum Technologies Inc.

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
from pennylane import allocate, deallocate
from pennylane.allocation import (
    Allocate,
    AllocateState,
    Deallocate,
    DynamicRegister,
    DynamicWire,
    allocate_prim,
    deallocate_prim,
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
        assert hash(a) == hash(c)
        assert hash(a) != hash(b)


class TestAllocateOp:

    @pytest.mark.jax
    def test_valid_operation(self):
        """Test that Allocate is a valid Operator."""
        op = Allocate.from_num_wires(3)
        qml.ops.functions.assert_valid(op)

    def test_allocate_from_num_wires(self):
        """Test that the op can be instantiated with from_num_wires"""

        op = Allocate.from_num_wires(3, state=AllocateState.ANY, restored=True)
        assert len(op.wires) == 3
        assert op.hyperparameters == {"state": AllocateState.ANY, "restored": True}
        assert op.state == AllocateState.ANY
        assert op.restored

    def test_normal_initialization(self):
        """Test that the op can also be initialized with already created dynamic wires."""
        wires = [DynamicWire() for _ in range(5)]
        op = Allocate(wires, state=AllocateState.ANY, restored=True)
        assert op.wires == qml.wires.Wires(wires)
        assert op.hyperparameters == {"state": AllocateState.ANY, "restored": True}
        assert op.state == AllocateState.ANY
        assert op.restored

    def test_default_hyperparameters(self):
        """Test the values of the default hyperparameters."""

        op = Allocate.from_num_wires(2)
        assert op.state == AllocateState.ZERO
        assert not op.restored

        op2 = Allocate(DynamicWire())
        assert op2.state == AllocateState.ZERO
        assert not op.restored


def test_dynamic_register_not_hashable():
    """Test that the DynamicRegister is not hashable."""

    reg = DynamicRegister([DynamicWire()])
    with pytest.raises(TypeError, match="unhashable type"):
        hash(reg)

    with pytest.raises(qml.exceptions.WireError, match="Wires must be hashable"):
        qml.wires.Wires((0, reg))


def test_Deallocate_validity():
    """Test that Deallocate is a valid operation."""
    wires = [DynamicWire(), DynamicWire()]
    op = Deallocate(wires)
    assert op.wires == qml.wires.Wires(wires)
    qml.ops.functions.assert_valid(op)


def test_error_bad_state():
    """Test that a VAlueError is raised for an unsupported state."""

    with pytest.raises(ValueError, match="is not a valid AllocateState"):
        allocate(2, state="no")


def test_allocate_function():
    """Test that allocate returns dynamic wires and queues an Allocate op."""
    with qml.queuing.AnnotatedQueue() as q:
        wires = allocate(4)
    assert isinstance(wires, DynamicRegister)
    assert len(wires) == 4
    assert isinstance(wires[:3], qml.wires.Wires)
    assert all(isinstance(w, DynamicWire) for w in wires)

    assert len(q) == 1
    op = q.queue[0]
    assert isinstance(op, Allocate)
    assert op.wires == qml.wires.Wires(wires)
    assert op.state == AllocateState.ZERO


def test_allocate_kwargs():
    """Test that the kwargs to allocate get passed to the op."""

    with qml.queuing.AnnotatedQueue() as q:
        allocate(3, state="any", restored=True)

    op = q.queue[0]
    assert op.state == AllocateState.ANY
    assert op.restored


class TestDeallocate:

    def test_single_dynamic_wire(self):
        """Test that deallocate can accept a single dynamic wire."""
        wire = DynamicWire()
        with qml.queuing.AnnotatedQueue() as q:
            op = deallocate(wire)
        assert op.wires == qml.wires.Wires((wire,))

        assert len(q.queue) == 1
        assert op is q.queue[0]
        assert isinstance(op, Deallocate)

    def test_error_non_dynamic_wire(self):
        """Test that an error is raised if a non-dynamic wire is attempted to be deallocated."""
        with pytest.raises(ValueError, match="only accepts DynamicWire wires."):
            deallocate((DynamicWire, 1))

    def test_multiple_dynamic_wires(self):
        """Test multiple dynamic wires can be deallocated."""

        wires = [DynamicWire(), DynamicWire()]
        with qml.queuing.AnnotatedQueue() as q:
            op = deallocate(wires)
        assert op.wires == qml.wires.Wires(wires)

        assert len(q.queue) == 1
        assert op is q.queue[0]
        assert isinstance(op, Deallocate)


def test_dynamic_register_repr():
    """Test the repr for the DynamicRegister."""

    reg = DynamicRegister((DynamicWire(), DynamicWire()))
    assert repr(reg) == "<DynamicRegister: size=2>"


def test_allocate_context_manager():
    """Test that allocate when used as context manager allocates and deallocates qubits."""

    with qml.queuing.AnnotatedQueue() as q:
        with allocate(3, state="any", restored=True) as wires:
            assert len(wires) == 3
            assert all(isinstance(w, DynamicWire) for w in wires)
            assert len(set(wires)) == 3

            qml.I(wires)

    assert len(q.queue) == 3
    qml.assert_equal(q.queue[0], Allocate(wires, state=AllocateState.ANY, restored=True))
    qml.assert_equal(q.queue[1], qml.I(wires))
    qml.assert_equal(q.queue[2], Deallocate(wires))


@pytest.mark.jax
@pytest.mark.capture
class TestCaptureIntegration:

    @pytest.mark.parametrize("use_context", (True, False))
    def test_capturing_allocate_and_deallocate(self, use_context):
        """Test that allocate and deallcoate can be captured."""

        import jax

        def f():
            if use_context:
                with allocate(2, state="zero", restored=True) as wires:
                    qml.H(wires[0])
                    qml.Z(wires[1])
            else:
                w, w2 = allocate(2, state="zero", restored=True)
                qml.H(w)
                qml.Z(w2)
                deallocate((w, w2))

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == allocate_prim
        assert len(jaxpr.eqns[0].invars) == 0
        assert jaxpr.eqns[0].params == {
            "num_wires": 2,
            "state": AllocateState.ZERO,
            "restored": True,
        }
        assert len(jaxpr.eqns[0].outvars) == 2
        assert all(v.aval.shape == () for v in jaxpr.eqns[0].outvars)
        for v in jaxpr.eqns[0].outvars:
            assert v.aval.dtype == jax.numpy.int64

        assert jaxpr.eqns[1].invars[0] is jaxpr.eqns[0].outvars[0]
        assert jaxpr.eqns[2].invars[0] is jaxpr.eqns[0].outvars[1]

        assert jaxpr.eqns[3].primitive == deallocate_prim
        assert jaxpr.eqns[3].params == {}
        assert jaxpr.eqns[3].invars == jaxpr.eqns[0].outvars

        with pytest.raises(NotImplementedError):
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    def test_deallocate_single_wire(self):
        """Test deallocate can accept a single wire."""

        import jax

        def f():
            [w] = allocate(1)
            qml.X(w)
            deallocate(w)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == allocate_prim
        assert len(jaxpr.eqns[0].invars) == 0
        assert jaxpr.eqns[0].params == {
            "num_wires": 1,
            "state": AllocateState.ZERO,
            "restored": False,
        }
        assert len(jaxpr.eqns[0].outvars) == 1
        assert all(v.aval.shape == () for v in jaxpr.eqns[0].outvars)
        for v in jaxpr.eqns[0].outvars:
            assert v.aval.dtype == jax.numpy.int64

        assert jaxpr.eqns[1].invars[0] is jaxpr.eqns[0].outvars[0]

        assert jaxpr.eqns[2].primitive == deallocate_prim
        assert jaxpr.eqns[2].params == {}
        assert jaxpr.eqns[2].invars == jaxpr.eqns[0].outvars

    def test_no_implementation(self):
        """Test that (de)allocation has no concrete implementation."""
        with pytest.raises(NotImplementedError):
            allocate(2)

        with pytest.raises(NotImplementedError):
            with allocate(2) as wires:
                qml.X(wires)

        with pytest.raises(NotImplementedError):
            deallocate(2)

    def test_no_dynamic_allocation_size(self):
        """Test that allocation size must be static with capture."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def c(n: int):
            allocate(n)

        with pytest.raises(
            NotImplementedError,
            match="Number of allocated wires must be static when capture is enabled.",
        ):
            c(1)


@pytest.mark.integration
class TestDeviceIntegration:

    @pytest.mark.parametrize("dev_name", ("default.qubit",))
    @pytest.mark.parametrize("device_wires", (None, (0, 1, 2)))
    def test_reuse_without_mcms(self, dev_name, device_wires):
        """Test that a dynamic allocations that do not require mcms can be executed."""

        @qml.qnode(qml.device(dev_name, wires=device_wires))
        def c():
            with allocate(1, restored=True) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 0))
                qml.H(wires)

            with allocate(1) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 1))
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res1, res2 = c()
        assert qml.math.allclose(res1, 0)
        assert qml.math.allclose(res2, 0)

    @pytest.mark.parametrize("dev_name", ("default.qubit",))
    @pytest.mark.parametrize("device_wires", (None, (0, 1, 2, 3)))
    @pytest.mark.parametrize("mcm_method", ("tree-traversal", "deferred", "one-shot"))
    def test_reuse_with_mcms(self, dev_name, device_wires, mcm_method):
        """Test that a simple dynamic allocation can be executed."""

        @qml.set_shots(5000 if mcm_method == "one-shot" else None)
        @qml.qnode(qml.device(dev_name, wires=device_wires), mcm_method=mcm_method)
        def c():
            with allocate(1, restored=False) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 0))
                qml.H(wires)

            with allocate(1) as wires:
                qml.H(wires)
                qml.CNOT((wires[0], 1))
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        res1, res2 = c()
        atol = 0.05 if mcm_method == "one-shot" else 1e-6
        assert qml.math.allclose(res1, 0, atol=atol)
        assert qml.math.allclose(res2, 0, atol=atol)
