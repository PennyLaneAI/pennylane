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
"""Tests for Subroutine and SubroutineOp"""

# pylint: disable=unused-argument
import inspect
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pytest

import pennylane as qp
from pennylane.decomposition import resource_rep
from pennylane.ops import CNOT, Adjoint, PauliX, PauliZ
from pennylane.templates import AbstractArray, Subroutine, SubroutineOp, subroutine_resource_rep
from pennylane.templates.core import (
    CollectedSubroutine,
    _make_signature_key,
    adjoint_subroutine_resource_rep,
    change_op_basis_subroutine_resource_rep,
)


class TestInitialization:

    def test_default_creation(self):
        """Test the creation of a simple subroutine."""

        def Simple(x, wires):
            qp.RX(x, wires=wires)

        S = Subroutine(Simple)

        assert S.name == "Simple"
        assert S.signature == inspect.signature(Simple)

        a, k = S.setup_inputs("a", wires="b")
        assert a == ("a",)
        assert k == {"wires": "b"}

        assert S.static_argnames == tuple()
        assert S.wire_argnames == ("wires",)
        assert S.dynamic_argnames == ("x",)
        assert S.exact_resources

        with qp.queuing.AnnotatedQueue() as q:
            S.definition(0.5, 0)

        assert len(q.queue) == 1
        qp.assert_equal(q.queue[0], qp.RX(0.5, 0))

        resources = S.compute_resources(0.5, wires=0)
        expected = defaultdict(int)
        expected[qp.resource_rep(qp.RX)] = 1
        assert resources == expected

    def test_wire_argnames(self):
        """Test that wire argnames can be specified."""

        def WireArgnames(x, reg1, reg2, y):
            pass

        S = Subroutine(WireArgnames, wire_argnames=("reg1", "reg2"))

        assert S.wire_argnames == (
            "reg1",
            "reg2",
        )
        assert S.dynamic_argnames == ("x", "y")

    def test_static_argnames_and_setup_inputs(self):
        """Test static argnames and setup_inputs can be provided."""

        def setup_inputs(x, y, wires, z):
            return (x, y), {"wires": wires, "z": tuple(z)}

        def T(x, y, wires, z):
            pass

        S = Subroutine(T, setup_inputs=setup_inputs, static_argnames=("z",))
        assert S.static_argnames == ("z",)
        assert S.dynamic_argnames == (
            "x",
            "y",
        )

        a, k = S.setup_inputs(0.5, 1.2, "a", [1, 2, 3])
        assert a == (0.5, 1.2)
        assert k == {"wires": "a", "z": (1, 2, 3)}

    def test_string_static_argnames_static_argnums(self):
        """Test that if a string is provided to static argnums or static argnames, it's wrapped in a tuple."""

        def f(metadata, register):
            pass

        S = Subroutine(f, static_argnames="metadata", wire_argnames="register")

        assert S.wire_argnames == ("register",)
        assert S.static_argnames == ("metadata",)

    def test_providing_compute_resources(self):
        """Test that compute_resources can be set."""

        def compute_resources(x, wires, metadata):
            assert metadata == 2  # test filled in with default values.
            assert isinstance(wires, qp.wires.Wires)
            return {qp.RX: len(wires)}

        @partial(Subroutine, compute_resources=compute_resources)
        def HasResources(x, wires, metadata=2):
            for w in wires:
                qp.RX(x, w)

        out = HasResources.compute_resources(0.5, [0, 1, 2])
        assert out == {qp.RX: 3}
        # test wires setup properly
        out2 = HasResources.compute_resources(0.5, 0)
        assert out2 == {qp.RX: 1}

    @pytest.mark.parametrize("exact_resources", (True, False))
    def test_setting_exact_resources(self, exact_resources):
        """Test that exact_resources can be set."""

        @partial(Subroutine, exact_resources=exact_resources)
        def f(wires):
            qp.X(wires)

        assert f.exact_resources == exact_resources

    def test_error_if_wire_argnames_not_present(self):
        """Test that an error is raised if a wire argname is not present in the signature."""

        def f(register):
            pass

        with pytest.raises(
            ValueError, match="wire argname 'wires' not present in function signature."
        ):
            Subroutine(f)


def Example1(x, y, reg1, reg2, pauli_words):
    qp.PauliRot(x, pauli_words[0], reg1)
    qp.PauliRot(y, pauli_words[1], reg2)
    return 2


def Example1SetupInputs(x, y, reg1, reg2, pauli_words):
    return (x, y, reg1, reg2, tuple(pauli_words)), {}


def Example1Resources(x, y, reg1, reg2, pauli_words):
    return {
        qp.resource_rep(qp.PauliRot, pauli_word=pw): num
        for pw, num in Counter(pauli_words).items()
    }


Example1Subroutine = Subroutine(
    Example1,
    wire_argnames=("reg1", "reg2"),
    static_argnames=("pauli_words",),
    setup_inputs=Example1SetupInputs,
    compute_resources=Example1Resources,
)


def generate_subroutine_op_example(*args, **kwargs):
    bound_args = Example1Subroutine.signature.bind(*args, **kwargs)

    with qp.queuing.AnnotatedQueue() as q:
        output = Example1(*args, **kwargs)
    return SubroutineOp(Example1Subroutine, bound_args, decomposition=q.queue, output=output)


def test_operator_method():
    """Test that the operator method returns a SubroutineOp"""

    @Subroutine
    def f(x, wires):
        qp.RX(x, wires)
        return 2

    op = f.operator(0.5, 0)
    assert isinstance(op, SubroutineOp)
    assert op.output == 2
    qp.equal(op.decomposition()[0], qp.RX(0.5, 0))


def test_fallback_creating_resources_AbstractArray():
    """Test that the fallback for calculating resources works with AbstractArray's."""

    @partial(Subroutine, static_argnames="rotation")
    def f(params, wires, rotation):
        for (
            p,
            w,
        ) in zip(params["a"], wires):
            qp.PauliRot(p, rotation, w)
        qp.MultiControlledX(wires)

    p = AbstractArray((3,), float)
    w = AbstractArray((3,))

    resources = f.compute_resources({"a": p}, w, "Z")
    expected = defaultdict(int)
    expected[qp.resource_rep(qp.PauliRot, pauli_word="Z")] = 3

    r = qp.resource_rep(
        qp.MultiControlledX,
        num_control_wires=2,
        num_zero_control_values=0,
        num_work_wires=0,
        work_wire_type="borrowed",
    )
    expected[r] = 1
    assert resources == expected


def test_fallback_resources_error():
    """Test that if an error occurs when using the resources fallback, we get a more informative error."""

    @qp.templates.Subroutine
    def f(wires):
        raise ValueError("AHHHH")

    with pytest.raises(
        ValueError, match="Fallback for computing resources for <Subroutine: f> failed."
    ):
        f.compute_resources(qp.templates.AbstractArray((2,)))


class TestSubroutineOp:

    op1 = generate_subroutine_op_example(
        0.5, 0.6, qp.wires.Wires((0, "a")), qp.wires.Wires(("a", 1)), ("XY", "YZ")
    )

    def test_no_primitive_bind_call(self):
        """Test that SubroutineOp has no _primitive_bind_call implementation."""

        with pytest.raises(ValueError, match="SubroutineOp's should never be directly"):
            SubroutineOp._primitive_bind_call()  # pylint: disable=protected-access

    def test_isinstance_with_Subroutine(self):
        """Test that SubroutineOp's are instances of their corresponding Subroutine."""

        # pylint: disable=isinstance-second-argument-not-valid-type
        assert isinstance(self.op1, Example1Subroutine)

    def test_basic_validity(self):
        """Test that subroutine op passes basic validity checks."""
        qp.ops.functions.assert_valid(self.op1, skip_pickle=True)

    def test_repr(self):
        """Test that SubroutineOp has a clean repr."""

        assert (
            repr(self.op1)
            == "<Example1(x=0.5, y=0.6, reg1=Wires([0, 'a']), reg2=Wires(['a', 1]), pauli_words=('XY', 'YZ'))>"
        )

    def test_set_wires(self):
        """Test that the wires for a subroutineop are set."""
        assert self.op1.wires == qp.wires.Wires([0, "a", 1])

    def test_set_data(self):
        """Test that data can automatically be extracted."""
        assert self.op1.data == (0.5, 0.6)

    def test_output(self):
        """Test accessing the output."""
        assert self.op1.output == 2

    def test_label(self):
        """Test that the label uses the name of the subroutine itself."""

        assert self.op1.label() == "Example1"

    def test_accessing_subroutine(self):
        """Test the subroutine can be accessed."""
        assert self.op1.subroutine == Example1Subroutine

    def test_map_wires(self):
        """Test that map_wires can handle multiple registers."""

        new_op = self.op1.map_wires({0: 3, "a": 4})
        assert new_op.wires == qp.wires.Wires([3, 4, 1])
        decomp = new_op.decomposition()
        qp.assert_equal(decomp[0], qp.PauliRot(0.5, "XY", (3, 4)))
        qp.assert_equal(decomp[1], qp.PauliRot(0.6, "YZ", (4, 1)))

    def test_decomposition(self):
        """Test that decomposition behaves like normal."""

        with qp.queuing.AnnotatedQueue() as q:
            decomp = self.op1.decomposition()

        decomp2 = self.op1.decomposition()  # no queuing context

        for d in [q.queue, decomp, decomp2]:
            qp.assert_equal(d[0], qp.PauliRot(0.5, "XY", (0, "a")))
            qp.assert_equal(d[1], qp.PauliRot(0.6, "YZ", ("a", 1)))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_integration(self):
        """Test that SubroutineOp can be used with the decompose transform in a basic use case."""

        tape = qp.tape.QuantumScript([self.op1])

        [new_tape], _ = qp.transforms.decompose(tape, gate_set={qp.PauliRot})

        qp.assert_equal(new_tape[0], qp.PauliRot(0.5, "XY", (0, "a")))
        qp.assert_equal(new_tape[1], qp.PauliRot(0.6, "YZ", ("a", 1)))


class TestSubroutineCall:

    def test_setup_inputs_run(self):
        """Test that setup_inputs is run and can make user inputs hashable."""

        def my_setup_inputs(wires, pauli_words):
            return (), {"wires": wires, "pauli_words": tuple(pauli_words)}

        @partial(Subroutine, setup_inputs=my_setup_inputs)
        def f(wires, pauli_words):
            assert isinstance(pauli_words, tuple)
            qp.PauliRot(0.5, pauli_words[0], wires)

        with qp.queuing.AnnotatedQueue() as q:
            out = f(0, ["X"])
        op = q.queue[0]
        assert out is None
        assert op.bound_args.arguments["pauli_words"] == ("X",)

    def test_fill_in_default_values(self):
        """Test that we will in default values when binding."""

        @partial(Subroutine, static_argnames="metadata")
        def f(wires, metadata="default_value"):
            pass

        with qp.queuing.AnnotatedQueue() as q:
            out = f(0)
        assert out is None
        op = q.queue[0]
        assert op.bound_args.arguments["metadata"] == "default_value"

    @pytest.mark.usefixtures("ignore_id_deprecation")
    def test_handle_id(self):
        """Test that Subroutine's can handle accepting an id."""

        @Subroutine
        def f(wires):
            pass

        op = f.operator(0, id="val")

        assert op.id == "val"

    def test_mcm_outputs(self):
        """Test that a subroutine can return mcms."""

        @Subroutine
        def mcm_output(wires):
            return [qp.measure(w) for w in wires]

        with qp.queuing.AnnotatedQueue() as q:
            out = mcm_output((0, 1))

        assert len(out) == 2
        assert isinstance(out, list)
        assert isinstance(out[0], qp.ops.MeasurementValue)
        assert isinstance(out[1], qp.ops.MeasurementValue)

        assert len(q.queue) == 1
        op = q.queue[0]
        assert op.output is out
        assert op.subroutine == mcm_output


@pytest.mark.capture
class TestSubroutineCapture:

    def test_setup_inputs_program_capture(self):
        """Test that setup_inputs can make inputs hashable for use with program capture."""

        import jax  # pylint: disable=import-outside-toplevel

        def my_setup_inputs(wires, pauli_words):
            return (), {"wires": wires, "pauli_words": tuple(pauli_words)}

        @partial(Subroutine, setup_inputs=my_setup_inputs, static_argnames="pauli_words")
        def f(wires, pauli_words):
            assert isinstance(pauli_words, tuple)
            qp.PauliRot(0.5, pauli_words[0], wires)

        def w():
            f(0, ["X"])

        jaxpr = jax.make_jaxpr(w)()
        assert jaxpr.eqns[-1].primitive == qp.capture.primitives.quantum_subroutine_prim
        inner_jaxpr = jaxpr.eqns[-1].params["jaxpr"]
        # pylint: disable=protected-access
        assert inner_jaxpr.eqns[-1].primitive == qp.PauliRot._primitive
        assert inner_jaxpr.eqns[-1].params["pauli_word"] == "X"

    def test_different_forms_of_wires(self):
        """Test that wires can be provided as literal integers, traced integers, lists, tuple, and arrays."""

        import jax

        @qp.templates.Subroutine
        def f(wires):
            assert wires.shape == (1,)
            qp.X(wires[0])

        jaxpr1 = jax.make_jaxpr(f)(0)

        def w1():
            return f(0)

        jaxpr2 = jax.make_jaxpr(w1)()

        jaxpr3 = jax.make_jaxpr(f)([0])
        jaxpr4 = jax.make_jaxpr(f)((0,))

        def w2():
            return f([0])

        jaxpr5 = jax.make_jaxpr(w2)()

        for jaxpr in [jaxpr1, jaxpr2, jaxpr3, jaxpr4, jaxpr5]:
            assert jaxpr.eqns[-1].primitive == qp.capture.primitives.quantum_subroutine_prim

            assert jaxpr.eqns[-1].invars[0].aval.shape == (1,)
            assert "int" in jaxpr.eqns[-1].invars[0].aval.dtype.name

    def test_mcm_return(self):
        """Test that a Subroutine can return classical values."""

        import jax

        @qp.templates.Subroutine
        def f(wires):
            return [qp.measure(w) for w in wires]

        def w(wires):
            out = f(wires)
            assert isinstance(out, list)
            assert len(out) == 3
            for m in out:
                # scalar output
                assert m.shape == ()
            return out

        jaxpr = jax.make_jaxpr(w)((0, 1, 2))

        eqn = jaxpr.eqns[-1]  # setup has some slicing and dicing
        assert len(eqn.outvars) == 3  # the three measurement values
        assert eqn.primitive == qp.capture.primitives.quantum_subroutine_prim

    def test_autograph_not_propagated_through(self):
        """Test that autograph would propagate through a Subroutine."""

        import jax  # pylint: disable=import-outside-toplevel

        @Subroutine
        def f(x, wires):
            if x > 0:
                qp.X(wires)
            else:
                qp.Y(wires)

        @qp.capture.run_autograph
        def w(x):
            f(x, 0)

        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.make_jaxpr(w)(0.5)

    def test_manual_autograph_use(self):
        """Test that autograph can be manually applied to a Subroutine."""

        import jax  # pylint: disable=import-outside-toplevel

        @Subroutine
        @qp.capture.run_autograph
        def f(x, wires):
            if x > 0:
                qp.X(wires)

        jaxpr = jax.make_jaxpr(f)(0.5, 0)

        subroutine_eqn = jaxpr.eqns[-1]
        assert subroutine_eqn.primitive == qp.capture.primitives.quantum_subroutine_prim

        inner_xpr = subroutine_eqn.params["jaxpr"]
        assert inner_xpr.eqns[-1].primitive == qp.capture.primitives.cond_prim

    def test_id_ignored(self):
        """Test that id is ignored with program capture."""

        import jax  # pylint: disable=import-outside-toplevel

        @Subroutine
        def f(wires):
            pass

        def w():
            return f(0, id="val")

        jaxpr = jax.make_jaxpr(w)()
        assert "id" not in jaxpr.eqns[-1].params


@pytest.mark.capture
class TestCollectedSubroutine:

    def test_no_abstract_capturing(self):
        """Test that CollectedSubroutine can't occur during an abstract evaluation."""

        jax = pytest.importorskip("jax")

        def f():
            CollectedSubroutine("bla", [qp.X(0)])

        with pytest.raises(NotImplementedError, match="should never be hit"):
            jax.make_jaxpr(f)()

    def test_adjoint_of_subroutine_impl(self):
        """Test that if the adjoint of a subroutine is called without make_jaxpr and capture is enabled,
        we get the adjoint of a CollectedSubroutine."""

        @Subroutine
        def f(wires):
            qp.X(wires)

        with qp.queuing.AnnotatedQueue() as q:
            qp.adjoint(f)(0)

        [adj_op] = q.queue
        assert isinstance(adj_op, qp.ops.Adjoint)
        base = adj_op.base
        assert isinstance(base, CollectedSubroutine)
        assert base.name == "f"
        qp.assert_equal(base.decomposition()[0], qp.X(0))


@pytest.mark.integration
class TestTapePLIntegration:

    def test_basic_execution_integration(self):
        """Test that a basic subroutine can be executed by restrictive devices."""

        @Subroutine
        def Tester(x, y, wires):
            qp.RX(x, wires[0])
            qp.RY(y, wires[1])

        @qp.qnode(qp.device("reference.qubit", wires=2))
        def c(x, y):
            Tester(x, y, wires=(0, 1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        x = 0.5
        y = 1.2
        res1, res2 = c(x, y)
        assert qp.math.allclose(res1, qp.math.cos(x))
        assert qp.math.allclose(res2, qp.math.cos(y))

    def test_subroutine_gradient(self):
        """Test that SubroutineOp can be handled by gradients (namely parameter shift.)"""

        @Subroutine
        def Tester(x, y, wires):
            qp.RX(x, wires[0])
            qp.RY(y, wires[1])

        @qp.qnode(qp.device("reference.qubit", wires=2))
        def c(x, y):
            Tester(x, y, wires=(0, 1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        g0 = qp.grad(lambda x, y: c(x, y)[0])(qp.numpy.array(0.5), 1.2)
        assert qp.math.allclose(g0, -qp.math.sin(0.5))

    def test_mcm_integration(self):
        """Test that subroutines can return the results of mid circuit measurements."""

        @qp.templates.Subroutine
        def MCMTester(wires):
            return [qp.measure(wires=w, reset=True) for w in wires]

        @qp.qnode(qp.device("default.qubit", wires=2), mcm_method="tree-traversal")
        def c(x):
            qp.X(0)
            m1, m2 = MCMTester((0, 1))
            qp.cond(m1, qp.RX)(x, 0)
            qp.cond(m2, qp.RX)(x, 1)
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        r1, r2 = c(0.5)
        assert qp.math.allclose(r1, qp.math.cos(0.5))
        assert qp.math.allclose(r2, 1)

    def test_drawing(self):
        """Test that subroutines can be drawn."""

        @qp.templates.Subroutine
        def Tester(x, y, wires):
            qp.RX(x, wires[0])
            qp.RY(y, wires[1])

        @qp.qnode(qp.device("reference.qubit", wires=2))
        def c(x, y):
            Tester(x, y, wires=(0, 1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        out = qp.draw(c)(0.5, 1.2)
        assert out == "0: ─╭Tester(0.50,1.20)─┤  <Z>\n1: ─╰Tester(0.50,1.20)─┤  <Z>"

        out2 = qp.draw(c, decimals=None)(0.5, 1.2)
        assert out2 == "0: ─╭Tester─┤  <Z>\n1: ─╰Tester─┤  <Z>"

    def test_specs(self):
        """Test that subroutines show up as gate types in specs."""

        @qp.templates.Subroutine
        def Tester(x, y, wires):
            qp.RX(x, wires[0])
            qp.RY(y, wires[1])

        @qp.qnode(qp.device("reference.qubit", wires=2))
        def c(x, y):
            Tester(x, y, wires=(0, 1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        specs = qp.specs(c, level="top")(0.5, 1.2)
        assert specs.resources.gate_types["Tester"] == 1


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestGraphDecomposition:

    def test_creating_abstract_array(self):
        """Test basic checks for creating an AbstractArray."""

        a = qp.templates.AbstractArray((2, 2, 3), np.float64)
        assert a.shape == (2, 2, 3)
        assert a.dtype is np.dtype(np.float64)

        b = qp.templates.AbstractArray(())
        assert b.shape == ()
        assert b.dtype is np.dtype(np.int64)

        assert a != b
        assert hash(a)
        assert b == qp.templates.AbstractArray(())

    @pytest.mark.torch
    def test_torch_dtype_converted_to_numpy(self):
        """Test that torch data types are converted to numpy data types."""

        import torch

        x = torch.tensor(0.5, dtype=torch.float64)
        a = qp.templates.AbstractArray((), x.dtype)
        assert a.dtype is np.dtype(np.float64)

    def test_inbuilt_type_promotion_to_numpy(self):
        """Test that python types are converted to numpy types."""
        assert AbstractArray((), int).dtype is np.dtype(np.int64)
        assert AbstractArray((), float).dtype is np.dtype(np.float64)
        assert AbstractArray((), complex).dtype is np.dtype(np.complex128)

    def test_abstract_array_len(self):
        """Test that AbstractArray's have a length."""

        a = qp.templates.AbstractArray((2, 3, 3))
        assert len(a) == 18

    def test_resource_keys(self):
        """Test that the SubroutineOp resource keys are subroutine and signature key."""

        assert SubroutineOp.resource_keys == frozenset(("subroutine", "signature_key"))

    def test_accessing_resource_params(self):
        """Test that the resource_params creates the signature key properly."""

        op1 = generate_subroutine_op_example(
            0.5, 0.6, qp.wires.Wires((0, "a")), qp.wires.Wires(("a", 1)), ("XY", "YZ")
        )
        rp = op1.resource_params
        assert list(rp.keys()) == ["subroutine", "signature_key"]
        assert rp["subroutine"] == Example1Subroutine

        struct = qp.pytrees.flatten(0.5)[1]
        key = (
            (struct, (AbstractArray((), float),)),
            (struct, (AbstractArray((), float),)),
            AbstractArray((2,)),
            AbstractArray((2,)),
            ("XY", "YZ"),
        )
        assert rp["signature_key"] == key

    # pylint: disable=too-many-statements
    def test_change_op_basis_subroutine_resource_rep_with_a_subroutine(self):
        """Test creating a CompressedResourceRep specific to templates within change_op_basis with a subroutine and a nested resource_rep."""

        # use a non-standard order
        @partial(Subroutine, static_argnames="a", wire_argnames=("reg1", "reg2"))
        def f(a, reg1, reg2, x):
            pass

        x = {"a": AbstractArray((3,), float)}
        rr = change_op_basis_subroutine_resource_rep(
            partial(f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))),
            resource_rep(qp.PauliX),
        )
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "ChangeOpBasis"

        assert isinstance(rr.params["target_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["target_op"].name == "PauliX"
        assert rr.params["target_op"].op_type == PauliX

        assert isinstance(rr.params["compute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["compute_op"].name == "SubroutineOp"
        assert rr.params["compute_op"].op_type == SubroutineOp
        assert rr.params["compute_op"].params == {
            "subroutine": f,
            "signature_key": _make_signature_key(
                f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))
            ),
        }

        assert isinstance(rr.params["uncompute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["uncompute_op"].name == "Adjoint(SubroutineOp)"
        assert rr.params["uncompute_op"].op_type == Adjoint
        assert rr.params["uncompute_op"].params == {
            "base_class": SubroutineOp,
            "base_params": {
                "subroutine": f,
                "signature_key": _make_signature_key(
                    f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))
                ),
            },
        }

    def test_change_op_basis_subroutine_resource_rep_with_an_op_and_a_resource_rep(self):
        """Test creating a CompressedResourceRep specific to templates within change_op_basis with an op and a nested resource_rep."""

        rr = change_op_basis_subroutine_resource_rep(
            qp.PauliZ(0),
            resource_rep(qp.PauliX),
        )
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "ChangeOpBasis"

        assert isinstance(rr.params["compute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["compute_op"].name == "PauliZ"
        assert rr.params["compute_op"].op_type == PauliZ

        assert isinstance(rr.params["target_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["target_op"].name == "PauliX"
        assert rr.params["target_op"].op_type == PauliX

        assert isinstance(rr.params["uncompute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["uncompute_op"].name == "Adjoint(PauliZ)"
        assert rr.params["uncompute_op"].op_type == Adjoint

    def test_change_op_basis_subroutine_resource_rep_with_a_resource_rep_and_a_subroutine(self):
        """Test creating a CompressedResourceRep specific to templates within change_op_basis with a subroutine and a nested resource_rep."""

        @partial(Subroutine, static_argnames="a", wire_argnames=("reg1", "reg2"))
        def f(a, reg1, reg2, x):
            pass

        x = {"a": AbstractArray((3,), float)}
        rr = change_op_basis_subroutine_resource_rep(
            resource_rep(qp.PauliX),
            partial(f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))),
        )
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "ChangeOpBasis"

        assert isinstance(rr.params["compute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["compute_op"].name == "PauliX"
        assert rr.params["compute_op"].op_type == PauliX

        assert isinstance(rr.params["target_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["target_op"].name == "SubroutineOp"
        assert rr.params["target_op"].op_type == SubroutineOp
        assert rr.params["target_op"].params == {
            "subroutine": f,
            "signature_key": _make_signature_key(
                f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))
            ),
        }

        assert isinstance(rr.params["uncompute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["uncompute_op"].name == "Adjoint(PauliX)"
        assert rr.params["uncompute_op"].op_type == Adjoint

    def test_change_op_basis_subroutine_resource_rep_with_a_subroutine_uncompute(self):
        """Test creating a CompressedResourceRep specific to templates within change_op_basis with a subroutine uncompute."""

        @partial(Subroutine, static_argnames="a", wire_argnames=("reg1", "reg2"))
        def f(a, reg1, reg2, x):
            pass

        x = {"a": AbstractArray((3,), float)}
        rr = change_op_basis_subroutine_resource_rep(
            qp.CNOT([0, 1]),
            qp.PauliX(0),
            subroutine_resource_rep(f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))),
        )
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "ChangeOpBasis"

        assert isinstance(rr.params["compute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["compute_op"].name == "CNOT"
        assert rr.params["compute_op"].op_type == CNOT

        assert isinstance(rr.params["target_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["target_op"].name == "PauliX"
        assert rr.params["target_op"].op_type == PauliX

        assert isinstance(rr.params["uncompute_op"], qp.decomposition.CompressedResourceOp)
        assert rr.params["uncompute_op"].name == "SubroutineOp"
        assert rr.params["uncompute_op"].op_type == SubroutineOp
        assert rr.params["uncompute_op"].params == {
            "subroutine": f,
            "signature_key": _make_signature_key(
                f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))
            ),
        }

    def test_adjoint_subroutine_resource_rep(self):
        """Test creating a CompressedResourceRep specific to adjoint templates."""

        # use a non-standard order
        @partial(Subroutine, static_argnames="a", wire_argnames=("reg1", "reg2"))
        def f(a, reg1, reg2, x):
            pass

        x = {"a": AbstractArray((3,), float)}
        rr = adjoint_subroutine_resource_rep(
            f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,))
        )
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "Adjoint(SubroutineOp)"
        assert rr.params["base_params"]["subroutine"] == f

        s = qp.pytrees.flatten(x)[1]

        # note that order is reflected in the call signature order, not order
        # provided to adjoint_subroutine_resource_rep
        expected_signature_key = (
            "X",
            AbstractArray(()),
            AbstractArray((2,)),
            (s, (AbstractArray((3,), float),)),
        )
        assert rr.params["base_params"]["signature_key"] == expected_signature_key

        rr_all_positional = adjoint_subroutine_resource_rep(
            f, "X", AbstractArray(()), AbstractArray((2,)), x
        )
        assert rr_all_positional == rr
        assert hash(rr) == hash(rr_all_positional)

        # test against slight changes to make sure they are picked up in the condensed rep
        diff_pytree = {"b": AbstractArray((3,), float)}
        rr_diff_pytree = subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_pytree
        )
        assert rr != rr_diff_pytree

        diff_len = {"a": AbstractArray((4,), float)}
        rr_diff_len = adjoint_subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_len
        )
        assert rr != rr_diff_len

        diff_dtype = {"a": AbstractArray((3,), np.int32)}
        rr_dtype = adjoint_subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_dtype
        )
        assert rr != rr_dtype

        diff_num_wires = adjoint_subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((3,)), x=x
        )
        assert diff_num_wires != rr

        diff_metadata = adjoint_subroutine_resource_rep(
            f, "Y", AbstractArray(()), reg2=AbstractArray((2,)), x=x
        )
        assert rr != diff_metadata

    def test_subroutine_resource_rep(self):
        """Test creating a CompressedResourceRep specific to templates."""

        # use a non-standard order
        @partial(Subroutine, static_argnames="a", wire_argnames=("reg1", "reg2"))
        def f(a, reg1, reg2, x):
            pass

        x = {"a": AbstractArray((3,), float)}
        rr = subroutine_resource_rep(f, "X", AbstractArray(()), x=x, reg2=AbstractArray((2,)))
        assert isinstance(rr, qp.decomposition.CompressedResourceOp)
        assert rr.name == "SubroutineOp"
        assert rr.params["subroutine"] == f

        s = qp.pytrees.flatten(x)[1]

        # note that order is reflected in the call signature order, not order
        # provided to subroutine_resource_rep
        expected_signature_key = (
            "X",
            AbstractArray(()),
            AbstractArray((2,)),
            (s, (AbstractArray((3,), float),)),
        )
        assert rr.params["signature_key"] == expected_signature_key

        rr_all_positional = subroutine_resource_rep(
            f, "X", AbstractArray(()), AbstractArray((2,)), x
        )
        assert rr_all_positional == rr
        assert hash(rr) == hash(rr_all_positional)

        # test against slight changes to make sure they are picked up in the condensed rep
        diff_pytree = {"b": AbstractArray((3,), float)}
        rr_diff_pytree = subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_pytree
        )
        assert rr != rr_diff_pytree

        diff_len = {"a": AbstractArray((4,), float)}
        rr_diff_len = subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_len
        )
        assert rr != rr_diff_len

        diff_dtype = {"a": AbstractArray((3,), np.int32)}
        rr_dtype = subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((2,)), x=diff_dtype
        )
        assert rr != rr_dtype

        diff_num_wires = subroutine_resource_rep(
            f, "X", AbstractArray(()), reg2=AbstractArray((3,)), x=x
        )
        assert diff_num_wires != rr

        diff_metadata = subroutine_resource_rep(
            f, "Y", AbstractArray(()), reg2=AbstractArray((2,)), x=x
        )
        assert rr != diff_metadata

    def test_subroutine_resource_rep_default_values(self):
        """Test that subroutine_resource_rep fills in default values."""

        @partial(Subroutine, static_argnames="a")
        def f(wires, a="a"):
            pass

        rr = subroutine_resource_rep(f, AbstractArray(()))
        expected_key = (AbstractArray(()), "a")
        assert rr.params["signature_key"] == expected_key

        rr2 = subroutine_resource_rep(f, AbstractArray(()), "a")
        assert rr == rr2

    def test_pytree_array_input_resource_params(self):
        """Test calculating the resource params when the dynamic input has a pytree structure."""

        @qp.templates.Subroutine
        def f(x, wires):
            pass

        x = {"a": np.zeros((3, 4), dtype=np.float32), "b": np.ones((5, 4), dtype=np.int32)}
        op = f.operator(x, wires=0)

        struct = qp.pytrees.flatten(x)[1]

        expected_signature_key = (
            (
                struct,
                (AbstractArray((3, 4), dtype=np.float32), AbstractArray((5, 4), dtype=np.int32)),
            ),
            AbstractArray((1,)),
        )
        assert op.resource_params["signature_key"] == expected_signature_key

    def test_adjoint_of_subroutine(self):
        """Test that we can take the adjoint of a subroutine with graph decomps."""

        def RXLayerResources(params, wires):
            return {qp.RX: qp.math.shape(params)[0]}

        @partial(qp.templates.Subroutine, compute_resources=RXLayerResources)
        def RXLayer(params, wires):
            for i in range(params.shape[0]):
                qp.RX(params[i], wires[i])

        op = qp.adjoint(RXLayer.operator(np.array([1, 2, 3]), (1, 2, 3)))
        tape = qp.tape.QuantumScript([op])
        new_tape = qp.decompose(tape)[0][0]
        qp.assert_equal(new_tape[0], qp.RX(-3, 3))
        qp.assert_equal(new_tape[1], qp.RX(-2, 2))
        qp.assert_equal(new_tape[2], qp.RX(-1, 1))

    def test_ctrl_of_subroutine(self):
        """ "Test that graph decompositions can handle the ctrl of a subroutineop."""

        def RXLayerResources(params, wires):
            return {qp.RX: qp.math.shape(params)[0]}

        @partial(qp.templates.Subroutine, compute_resources=RXLayerResources)
        def RXLayer(params, wires):
            for i in range(params.shape[0]):
                qp.RX(params[i], wires[i])

        op = qp.ctrl(RXLayer.operator(np.array([1, 2, 3]), (1, 2, 3)), (4, 5))
        tape = qp.tape.QuantumScript([op])
        new_tape = qp.decompose(tape, max_expansion=1)[0][0]
        qp.assert_equal(new_tape[0], qp.ctrl(qp.RX(1, 1), (4, 5)))
        qp.assert_equal(new_tape[1], qp.ctrl(qp.RX(2, 2), (4, 5)))
        qp.assert_equal(new_tape[2], qp.ctrl(qp.RX(3, 3), (4, 5)))

    def test_operator_decompose_to_subroutine(self):
        """Test that an Operator can decompose to a subroutine, and that the choice
        of decomposition involving a subroutine can be optimally chosen based on the gateset.

        Here we have two subroutines, one with RX's and one with RY's.  We chose between the
        two of them based on the gate_set provided to the decompose transform.

        """

        def RXLayerResources(params, wires):
            return {qp.RX: qp.math.shape(params)[0]}

        @partial(qp.templates.Subroutine, compute_resources=RXLayerResources)
        def RXLayer(params, wires):
            for i in range(params.shape[0]):
                qp.RX(params[i], wires[i])

        def RYLayerResources(params, wires):
            return {qp.RY: qp.math.shape(params)[0]}

        @partial(qp.templates.Subroutine, compute_resources=RYLayerResources)
        def RYLayer(params, wires):
            for i in range(params.shape[0]):
                qp.RY(params[i], wires[i])

        # pylint: disable=too-few-public-methods
        class SubroutineDemoOp(qp.operation.Operator):

            resource_keys = frozenset(())

            @property
            def resource_params(self):
                return {}

        rx_rr = subroutine_resource_rep(RXLayer, AbstractArray((3,), float), AbstractArray((3,)))
        ry_rr = subroutine_resource_rep(RYLayer, AbstractArray((3,), float), AbstractArray((3,)))

        @qp.decomposition.register_resources({rx_rr: 1})
        def rx_decomp(wires):
            RXLayer(np.array([0.0, 1.0, 2.0]), wires)

        @qp.decomposition.register_resources({ry_rr: 1})
        def ry_decomp(wires):
            RYLayer(np.array([0.0, 1.0, 2.0]), wires)

        qp.add_decomps(SubroutineDemoOp, rx_decomp, ry_decomp)

        op = SubroutineDemoOp(wires=(0, 1, 2))
        tape = qp.tape.QuantumScript([op])

        tape_rx = qp.decompose(tape, gate_set={qp.RX})[0][0]
        qp.assert_equal(tape_rx[0], qp.RX(0.0, 0))
        qp.assert_equal(tape_rx[1], qp.RX(1.0, 1))
        qp.assert_equal(tape_rx[2], qp.RX(2.0, 2))

        tape_ry = qp.decompose(tape, gate_set={qp.RY})[0][0]
        qp.assert_equal(tape_ry[0], qp.RY(0.0, 0))
        qp.assert_equal(tape_ry[1], qp.RY(1.0, 1))
        qp.assert_equal(tape_ry[2], qp.RY(2.0, 2))

    def test_inexact_resources_testing(self):
        """Test that assert_valid will work on a Subroutine with inexact resources."""

        def r(wires):
            return {qp.X: 2}

        @partial(Subroutine, compute_resources=r, exact_resources=False)
        def f(wires):
            qp.X(wires)

        op = f.operator(0)
        qp.ops.functions.assert_valid(op, skip_pickle=True, skip_capture=True)

    def test_compute_resources_fallback(self):
        """Test that the compute_resources fallback allows integration with decomps by default."""

        @partial(Subroutine, static_argnames="rotation")
        def f(params, wires, rotation):
            for (
                p,
                w,
            ) in zip(params, wires):
                qp.PauliRot(p, rotation, w)
            qp.MultiControlledX(wires)

        params = np.array([0.5, 1.2, 3.4])
        wires = [0, 1, 2]
        tape = qp.tape.QuantumScript([f.operator(params, wires, "X")])
        [decomposed], _ = qp.decompose(tape, gate_set=qp.gate_sets.ALL_OPS)
        print(decomposed.circuit)
        qp.assert_equal(decomposed[0], qp.PauliRot(0.5, "X", 0))
        qp.assert_equal(decomposed[1], qp.PauliRot(1.2, "X", 1))
        qp.assert_equal(decomposed[2], qp.PauliRot(3.4, "X", 2))
        qp.assert_equal(decomposed[3], qp.MultiControlledX(wires))
