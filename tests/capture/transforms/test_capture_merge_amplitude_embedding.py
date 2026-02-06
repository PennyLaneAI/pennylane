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
"""Unit tests for the ``MergeAmplitudeEmbeddingInterpreter`` class"""
# pylint:disable=protected-access, wrong-import-position
import pytest

import pennylane as qp
from pennylane.transforms.core import TransformError

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pennylane.capture.primitives import transform_prim


def _find_eq_with_name(jaxpr, name):
    for eq in jaxpr.eqns:
        if name in eq.params:
            return eq.params[name]
        return None


from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    jacobian_prim,
    measure_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.merge_amplitude_embedding import (
    MergeAmplitudeEmbeddingInterpreter,
    merge_amplitude_embedding_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestRepeatedQubitTransformErrors:
    """Test TransformError is raised when operations exist before the AmplitudeEmbedding operators."""

    def test_repeated_wire_with_mcm(self):
        """Test that an error is raised if AmplitudeEmbedding acts on the same wires as an MCM."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.Hadamard(wires=2)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.measure(wires=1)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.Z(0))

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(qfunc)()

    def test_repeated_wire_error(self):
        """Test that an error is raised if AmplitudeEmbedding acts on the same wire"""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.X(0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(qfunc)()

    def test_repeated_traced_wire_error(self):
        """Test that an error is raised if AmplitudeEmbedding acts on the same traced wire"""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(w):
            qp.X(w)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=w)

        with pytest.raises(
            TransformError,
            match="Cannot apply qp.AmplitudeEmbedding after operators with dynamic wires.",
        ):
            jax.make_jaxpr(qfunc)(1)

    def test_simple_repeated_qubit_error(self):
        """Test that an error is raised if a qubit in the AmplitudeEmbedding had operations applied to it before."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.CNOT(wires=[0.0, 1.0])
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(qfunc)()

    def test_repeated_qubit_error_before_higher_order_prim(self):
        """Test that wire collision can be detected before a higher-order primitive is applied."""

        def ctrl_fn():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qp.X(0)  # Same wire as AE in ctrl_fn
            qp.ctrl(ctrl_fn, [2, 3])()

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(f)()

    def test_repeated_qubit_after_higher_order_prim(self):
        """Test that wire collision is able to be detected after a higher order primitive is applied."""

        def ctrl_fn():
            qp.X(0)

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qp.ctrl(ctrl_fn, [2])()  # ctrl_fn has wires 0
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(f)()

    @pytest.mark.parametrize("collision_wire", [0, 1, 2])
    def test_collision_before_cond_prim(self, collision_wire):
        """Test that an error is raised if a qubit in the AmplitudeEmbedding had operations applied to it before."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.cond(x > 2)
            def cond_f():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)

            @cond_f.else_if(x > 1)
            def _else_if():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            @cond_f.otherwise
            def _else():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=2)

            qp.X(collision_wire)  # Each branch of the cond will have a collision with this wire
            cond_f()

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(f)(3)

    @pytest.mark.parametrize("collision_wire", [0, 1, 2])
    def test_collision_after_cond_prim(self, collision_wire):
        """Test that visited wires are correctly collected during cond."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.cond(x > 2)
            def cond_f():
                qp.Z(0)

            @cond_f.else_if(x > 1)
            def _else_if():
                qp.Y(1)

            @cond_f.otherwise
            def _else():
                qp.X(2)

            cond_f()  # Each branch of the cond will have a collision with this wire
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=collision_wire)

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(f)(3)

    @pytest.mark.parametrize("collision_wire", [0, 1, 2, 3])
    def test_initial_wires_memory_before_cond(self, collision_wire):
        """Tests that the initial wires before a cond aren't forgotten."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.cond(x > 2)
            def cond_f():
                qp.Z(1)

            @cond_f.else_if(x > 1)
            def _else_if():
                qp.Y(2)

            @cond_f.otherwise
            def _else():
                qp.X(3)

            qp.X(0)
            cond_f()
            # visited wires after cond should be 0 (before cond) and 1, 2, 3 (during cond)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=collision_wire)

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            jax.make_jaxpr(f)(3)

    def test_mixed_higher_order_primitives(self):
        """Test that wire collisions through higher order primitives are detected."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.for_loop(3)
            def loop(i):
                qp.RX(i, 0)

            @qp.cond(x > 2)
            def cond_f():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            @cond_f.else_if(x > 1)
            def _else_if():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            @cond_f.otherwise
            def _else():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            loop()  # Contains wires 0
            cond_f()  # Also contains wires 0

        with pytest.raises(
            TransformError,
            match="qp.AmplitudeEmbedding cannot be applied on wires already used by other operations.",
        ):
            args = (3,)
            jax.make_jaxpr(f)(*args)


class TestMergeAmplitudeEmbeddingInterpreter:
    """Test the MergeAmplitudeEmbeddingInterpreter class works correctly."""

    def test_circuit_with_traced_states(self):
        """Test that the transform works correctly when the circuit has arguments."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(state1, state2):
            qp.AmplitudeEmbedding(state1, wires=0)
            qp.Hadamard(wires=0)
            qp.AmplitudeEmbedding(state2, wires=1)
            return qp.expval(qp.Z(0))

        args = (jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]))
        jaxpr = jax.make_jaxpr(qfunc)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        expected_ops = [
            qp.AmplitudeEmbedding(jnp.array([0.0, 0.0, 0.0, 1.0]), wires=[0, 1]),
            qp.Hadamard(wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_circuit_with_traced_wires(self):
        """Test that the transform works correctly when the circuit has traced wires."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(wires1, wires2):
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=wires1)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=wires2)
            qp.Hadamard(wires=0)
            return qp.expval(qp.Z(0))

        args = (0, 1)
        jaxpr = jax.make_jaxpr(qfunc)(*args)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        expected_ops = [
            qp.AmplitudeEmbedding([0.0, 0.0, 0.0, 1.0], wires=[0, 1]),
            qp.Hadamard(wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_circuit_with_no_merge_required(self):
        """Test that the transform works correctly when no merge is required."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.Hadamard(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qp.AmplitudeEmbedding([0.0, 1.0], wires=[1]),
            qp.Hadamard(wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_merge_simple(self):
        """Test that the transform works correctly for a simple example."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.Hadamard(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            qp.AmplitudeEmbedding(jnp.array([0.0, 0.0, 0.0, 1.0]), wires=[2, 3])
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_state = [0.0] * 16
        expected_state[-1] = 1.0
        expected_ops = [
            qp.AmplitudeEmbedding(expected_state, wires=[0, 1, 2, 3]),
            qp.Hadamard(wires=[0]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_batch_preservation(self):
        """Test that the batch dimension is preserved after the transform."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.AmplitudeEmbedding(jnp.array([[0, 1], [1, 0]]), wires=0)  # |1> and |0>
            qp.AmplitudeEmbedding(
                jnp.array([1, 0]), wires=1
            )  # |0> (batch will be extended to |0> and |0>)
            qp.AmplitudeEmbedding(jnp.array([[0, 1], [1, 0]]), wires=2)  # |1> and |0>
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        state1 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # |1 0 1>
        state2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # |0 0 0>
        expected_ops = [
            qp.AmplitudeEmbedding((state1, state2), wires=[0, 1, 2]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

        expected_meas = [
            qp.expval(qp.PauliZ(0)),
        ]
        meas = collector.state["measurements"]
        assert meas == expected_meas

    def test_returned_ops_are_not_merged(self):
        """Test that ops that are returned by the function being transformed are not ."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qp.AmplitudeEmbedding(jnp.array([1, 0]), wires=0)
            return qp.AmplitudeEmbedding(jnp.array([1, 0]), wires=1)

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        expected_ops = [
            qp.AmplitudeEmbedding([1.0, 0.0], wires=[0]),
            qp.AmplitudeEmbedding([1.0, 0.0], wires=[1]),
        ]

        ops = collector.state["ops"]
        assert ops == expected_ops

    def test_amplitude_embedding_before_dynamic_wires_op(self):
        """Test that AmplitudeEmbeddings are merged and applied
        before ops with dynamic wires."""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(w):
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=1)
            qp.H(w)

        jaxpr = jax.make_jaxpr(circuit)(0)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0)

        ops = collector.state["ops"]
        expected_ops = [
            qp.AmplitudeEmbedding(qp.math.array([1.0, 0.0, 0.0, 0.0]), wires=[0, 1]),
            qp.H(0),
        ]
        assert ops == expected_ops

    def test_dynamic_wire_ops_before_error(self):
        """Test that an error is raised if ops with dynamic wires are used before AmplitudeEmbedding"""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(w):
            qp.H(w)
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=[0])

        with pytest.raises(
            TransformError,
            match="Cannot apply qp.AmplitudeEmbedding after operators with dynamic wires.",
        ):
            _ = jax.make_jaxpr(f)(2)

    def test_dynamic_wire_embeddings(self):
        """Test that AmplitudeEmbeddings with dynamic wires can be merged. If there is overlap,
        an error will be raised at runtime."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(w):
            qp.AmplitudeEmbedding(jnp.array([1.0, 0]), [w])
            qp.H(0)
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), [1])

        jaxpr = jax.make_jaxpr(f)(0)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0)
        ops = collector.state["ops"]
        expected = [qp.AmplitudeEmbedding(qp.math.array([1.0, 0.0, 0.0, 0.0]), [0, 1]), qp.H(0)]

        assert ops == expected

        # If first embedding has wire 1, error will happen at runtime
        collector = CollectOpsandMeas()
        with pytest.raises(qp.wires.WireError, match="Wires must be unique"):
            collector.eval(jaxpr.jaxpr, jaxpr.consts, 1)

    def test_dynamic_wire_embedding_after_op(self):
        """Test that an error is raised if an AmplitudeEmbedding with dynamic wires is applied
        after other non-AmplitudeEmbedding ops."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(w):
            qp.H(0)
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), [w])

        with pytest.raises(
            TransformError, match="Cannot apply qp.AmplitudeEmbedding with dynamic wires"
        ):
            _ = jax.make_jaxpr(f)(0)


def test_plxpr_to_plxpr_transform():
    """Test that the plxpr transform works correctly for a simple example."""

    def qfunc():
        qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
        qp.Hadamard(wires=0)
        qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
        return qp.expval(qp.Z(0))

    jaxpr = jax.make_jaxpr(qfunc)()
    transformed_jaxpr = merge_amplitude_embedding_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {})
    assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
    assert len(transformed_jaxpr.eqns) == 4

    collector = CollectOpsandMeas()
    collector.eval(transformed_jaxpr.jaxpr, transformed_jaxpr.consts)

    expected_ops = [
        qp.AmplitudeEmbedding([0.0, 0.0, 0.0, 1.0], wires=[0, 1]),
        qp.Hadamard(0),
    ]

    ops = collector.state["ops"]
    assert ops == expected_ops

    expected_meas = [
        qp.expval(qp.PauliZ(0)),
    ]
    meas = collector.state["measurements"]
    assert meas == expected_meas


class TestHigherOrderPrimitiveIntegration:
    """Test that the transform works correctly when applied with higher order primitives."""

    def test_measure_prim(self):
        """Test that the transform works correctly with a valid MCM."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.Hadamard(wires=2)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            qp.measure(wires=0)

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(jaxpr.eqns[0].params["n_wires"], 2)
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == measure_prim

    @pytest.mark.xfail(
        reason="The transform does not currently merge through higher order primitives. See sc-85439."
    )
    def test_measure_prim_block(self):
        """Test that the transform works correctly when the merge is blocked by a MCM."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qp.Hadamard(wires=2)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.measure(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

        jaxpr = jax.make_jaxpr(qfunc)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(jaxpr.eqns[0].params["n_wires"], 2)
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == measure_prim

    def test_dynamic_wires_embedding_after_measure(self):
        """Test that an error is raised if an AmplitudeEmbedding with dynamic wires is applied after a
        qp.measure."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(w):
            qp.measure(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=w)

        with pytest.raises(
            TransformError, match="Cannot apply qp.AmplitudeEmbedding with dynamic wires"
        ):
            _ = jax.make_jaxpr(qfunc)(1)

    def test_ctrl_transform_prim(self):
        """Test that the transform works correctly when applied with ctrl_transform_prim."""

        def ctrl_fn():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            qp.X(0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.ctrl(ctrl_fn, [3, 4])()
            qp.RY(0, 1)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 3
        # TODO: This AE should be merged with the one in ctrl_fn, limitation of PC
        assert jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert jaxpr.eqns[1].primitive == ctrl_transform_prim
        assert jaxpr.eqns[2].primitive == qp.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["jaxpr"]
        assert inner_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(inner_jaxpr.eqns[0].params["n_wires"], 2)
        assert inner_jaxpr.eqns[1].primitive == qp.X._primitive

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_transform_prim(self, lazy):
        """Test that the transform works correctly when applied with adjoint_transform_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            def g():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.X(0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            qp.adjoint(g, lazy=lazy)()

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(inner_jaxpr.eqns[0].params["n_wires"], 2)
        assert inner_jaxpr.eqns[1].primitive == qp.X._primitive

    def test_cond_prim_only_true_branch(self):
        """Test that the transform works correctly when applied with cond_prim with just a true branch."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.cond(x > 2)
            def cond_f():
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            cond_f()

        args = (3,)
        jaxpr = jax.make_jaxpr(f)(*args)
        assert jaxpr.eqns[1].primitive == cond_prim

        # True branch
        branch = jaxpr.eqns[1].params["jaxpr_branches"][0]
        assert len(branch.eqns) == 1
        assert branch.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(branch.eqns[0].params["n_wires"], 2)

    def test_cond_prim_all_cond_branches(self):
        """Test that the transform works correctly when applied with cond_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qp.cond(x > 2)
            def cond_f():
                qp.Z(0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=2)
                return qp.expval(qp.Z(0))

            @cond_f.else_if(x > 1)
            def _else_if():
                qp.Y(1)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=2)
                return qp.expval(qp.Y(0))

            @cond_f.otherwise
            def _else():
                qp.X(2)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
                return qp.expval(qp.X(0))

            out = cond_f()
            return out

        args = (3,)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][0]
        assert qp.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qp.AmplitudeEmbedding._primitive,
            qp.Z._primitive,
            qp.Z._primitive,
            qp.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim
            for eqn, exp_prim in zip(branch.eqns, expected_primitives, strict=True)
        )

        # Elif branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][1]
        assert qp.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qp.AmplitudeEmbedding._primitive,
            qp.Y._primitive,
            qp.Y._primitive,
            qp.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim
            for eqn, exp_prim in zip(branch.eqns, expected_primitives, strict=True)
        )

        # Else branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][2]
        assert qp.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qp.AmplitudeEmbedding._primitive,
            qp.X._primitive,
            qp.X._primitive,
            qp.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim
            for eqn, exp_prim in zip(branch.eqns, expected_primitives, strict=True)
        )

    def test_for_loop_prim(self):
        """Test that the transform works correctly when applied with for_loop_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(n):

            # pylint:disable=unused-argument
            @qp.for_loop(n)
            def h(i):
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.Hadamard(0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)

            h()

        jaxpr = jax.make_jaxpr(f)(3)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(inner_jaxpr.eqns[0].params["n_wires"], 2)
        assert inner_jaxpr.eqns[1].primitive == qp.Hadamard._primitive

    def test_while_loop_prim(self):
        """Test that the transform works correctly when applied with while_loop_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(n):

            @qp.while_loop(lambda i: i < n)
            def h(i):
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
                qp.Hadamard(0)
                qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
                return i + 1

            h(0)

        jaxpr = jax.make_jaxpr(f)(3)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[1].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(inner_jaxpr.eqns[1].params["n_wires"], 2)
        assert inner_jaxpr.eqns[2].primitive == qp.Hadamard._primitive

    def test_qnode_prim(self):
        """Test that the transform works correctly when applied with qnode_prim."""
        dev = qp.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        @qp.qnode(dev)
        def circuit():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.Hadamard(0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(qfunc_jaxpr.eqns) == 4
        assert qfunc_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(qfunc_jaxpr.eqns[0].params["n_wires"], 2)
        assert qfunc_jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qp.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_jacobian_prim(self):
        """Test that the transform works correctly when applied with jacobian_prim."""

        dev = qp.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        @qp.qnode(dev)
        def circuit(a, b):
            qp.AmplitudeEmbedding(jnp.array([a, b]), wires=0)
            qp.Hadamard(0)
            qp.AmplitudeEmbedding(jnp.array([a, b]), wires=1)
            return qp.expval(qp.Z(0))

        f = qp.grad(circuit)
        jaxpr = jax.make_jaxpr(f)(0, 1)
        assert jaxpr.eqns[0].primitive == jacobian_prim
        inner_jaxpr = _find_eq_with_name(jaxpr, "jaxpr")
        qfunc_jaxpr = _find_eq_with_name(inner_jaxpr, "qfunc_jaxpr")
        # Get all operators in qfunc
        qfunc_jaxpr = qfunc_jaxpr.replace(
            eqns=[
                eqn
                for eqn in qfunc_jaxpr.eqns
                if getattr(eqn.primitive, "prim_type", "") == "operator"
            ]
        )
        assert len(qfunc_jaxpr.eqns) == 3
        assert qfunc_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(qfunc_jaxpr.eqns[0].params["n_wires"], 2)
        assert qfunc_jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qp.PauliZ._primitive


class TestDynamicWiresControlFlowPrimitivesIntegration:
    """Unit tests for using dynamic wires with control flow primitives."""

    @pytest.mark.parametrize("embedding_location", ["cond", "after"])
    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_cond_op_before(self, embedding_location, dynamic_wires):
        """Test that applying an AmplitudeEmbedding with dynamic wires inside or after a cond after other
        ops are already applied raises an error."""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(x, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0
            qp.H(op_wires)

            @qp.cond(x < 2)
            def cond_fn():
                if embedding_location == "cond":
                    qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
                qp.H(0)

            @cond_fn.otherwise
            def _():
                qp.Y(0)

            cond_fn()
            if embedding_location == "after":
                qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(1.5, 0)

    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_cond_dyn_wires_one_branch_embedding_other_branch(self, dynamic_wires):
        """Test that an operator in one branch, and an AmplitudeEmbedding in another, where either has
        dynamic wires, does not raise an error."""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(x, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0

            @qp.cond(x < 2)
            def cond_fn():
                qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)

            @cond_fn.otherwise
            def _():
                qp.Y(op_wires)

            cond_fn()
            return qp.expval(qp.Z(0))

        # No error should be raised
        _ = jax.make_jaxpr(circuit)(1.5, 0)

    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_cond_op_inside_embedding_after(self, dynamic_wires):
        """Test that applying an AmplitudeEmbedding after a cond that contains an op where
        either can have dynamic wires raises an error"""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(x, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0

            @qp.cond(x < 2)
            def cond_fn():
                qp.H(op_wires)

            cond_fn()
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(1.5, 0)

    @pytest.mark.parametrize("embedding_location", ["loop", "after"])
    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_for_loop_op_before(self, dynamic_wires, embedding_location):
        """Test that an error is raised if an op is applied before a for loop with an embedding inside or after
        the loop where either can have dynamic wires"""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(n, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0
            qp.H(op_wires)

            @qp.for_loop(n)
            def loop_fn(i):  # pylint: disable=unused-argument
                if embedding_location == "loop":
                    qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
                qp.H(0)

            loop_fn()
            if embedding_location == "after":
                qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(5, 0)

    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_for_loop_op_inside_embedding_after(self, dynamic_wires):
        """Test that an error is raised if an op is applied inside a for loop with an embedding after
        the loop where either can have dynamic wires"""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(n, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0

            @qp.for_loop(n)
            def loop_fn(i):  # pylint: disable=unused-argument
                qp.H(op_wires)

            loop_fn()
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(5, 0)

    @pytest.mark.parametrize("embedding_location", ["loop", "after"])
    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_while_loop_op_before(self, dynamic_wires, embedding_location):
        """Test that an error is raised if an op is applied before a while loop with an embedding inside or after
        the loop where either can have dynamic wires"""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(x, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0
            qp.H(op_wires)

            @qp.while_loop(lambda arg: arg < 2)
            def loop_fn(arg):
                if embedding_location == "loop":
                    qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
                qp.H(0)
                return arg - 1

            loop_fn(x)
            if embedding_location == "after":
                qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), wires=embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(1.5, 0)

    @pytest.mark.parametrize("dynamic_wires", ["op", "embedding", "both"])
    def test_while_loop_op_inside_embedding_after(self, dynamic_wires):
        """Test that an error is raised if an op is applied inside a while loop with an embedding after
        the loop where either can have dynamic wires"""

        @MergeAmplitudeEmbeddingInterpreter()
        def circuit(x, w):
            op_wires = w if dynamic_wires in ("op", "both") else 0
            embedding_wires = w if dynamic_wires in ("embedding", "both") else 0

            @qp.while_loop(lambda arg: arg < 2)
            def loop_fn(arg):
                qp.H(op_wires)
                return arg - 1

            loop_fn(x)
            qp.AmplitudeEmbedding(jnp.array([1.0, 0.0]), embedding_wires)
            return qp.expval(qp.Z(0))

        with pytest.raises(TransformError, match="it is indeterminable if the wires overlap"):
            _ = jax.make_jaxpr(circuit)(1.5, 0)


class TestExpandPlxprTransformIntegration:
    """Test that the transform works with expand_plxpr_transform"""

    def test_example(self):
        """Test that the transform works with expand_plxpr_transform"""

        @qp.transforms.optimization.merge_amplitude_embedding
        def qfunc():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.Hadamard(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert jaxpr.eqns[0].primitive == transform_prim
        assert (
            jaxpr.eqns[0].params["transform"]
            == qp.transforms.optimization.merge_amplitude_embedding
        )

        transformed_qfunc = qp.capture.expand_plxpr_transforms(qfunc)
        transformed_jaxpr = jax.make_jaxpr(transformed_qfunc)()
        assert len(transformed_jaxpr.eqns) == 4
        assert transformed_jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(transformed_jaxpr.eqns[0].params["n_wires"], 2)
        assert transformed_jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert transformed_jaxpr.eqns[2].primitive == qp.PauliZ._primitive
        assert transformed_jaxpr.eqns[3].primitive == qp.measurements.ExpectationMP._obs_primitive

    def test_decorator(self):
        """Test that the transform works with the decorator"""

        @qp.capture.expand_plxpr_transforms
        @qp.transforms.optimization.merge_amplitude_embedding
        def qfunc():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.Hadamard(wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qp.AmplitudeEmbedding._primitive
        assert qp.math.allclose(jaxpr.eqns[0].params["n_wires"], 2)
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qp.PauliZ._primitive
        assert jaxpr.eqns[3].primitive == qp.measurements.ExpectationMP._obs_primitive
