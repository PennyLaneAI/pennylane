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

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.transforms.optimization.merge_amplitude_embedding import (
    MergeAmplitudeEmbeddingInterpreter,
    merge_amplitude_embedding_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestMergeAmplitudeEmbeddingInterpreter:
    """Test the MergeAmplitudeEmbeddingInterpreter class works correctly."""

    def test_repeated_qubit_error(self):
        """Test that an error is raised if a qubit in the AmplitudeEmbedding had operations applied to it before."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qml.CNOT(wires=[0.0, 1.0])
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)

        with pytest.raises(qml.DeviceError, match="applied in the same qubit"):
            jax.make_jaxpr(qfunc)()

    def test_circuit_with_arguments(self):
        """Test that the transform works correctly when the circuit has arguments."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(state1, state2):
            qml.AmplitudeEmbedding(state1, wires=0)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(state2, wires=1)
            return qml.expval(qml.Z(0))

        states = (jax.numpy.array([0.0, 1.0]), jax.numpy.array([0.0, 1.0]))
        jaxpr = jax.make_jaxpr(qfunc)(*states)

        assert jaxpr.eqns[-4].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[-4].params["n_wires"], 2)
        assert jaxpr.eqns[-3].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_circuit_with_traced_wires(self):
        """Test that the transform works correctly when the circuit has traced wires."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc(wires1, wires2):
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=wires1)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=wires2)
            return qml.expval(qml.Z(0))

        states = (jax.numpy.array([0.0]), jax.numpy.array([1.0]))
        jaxpr = jax.make_jaxpr(qfunc)(*states)

        assert jaxpr.eqns[-4].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[-4].params["n_wires"], 2)
        assert jaxpr.eqns[-3].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_circuit_with_no_merge_required(self):
        """Test that the transform works correctly for a simple example."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[0].params["n_wires"], 1)
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_merge_simple(self):
        """Test that the transform works correctly for a simple example."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[0].params["n_wires"], 2)
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_batch_preservation(self):
        """Test that the batch dimension is preserved after the transform."""

        @MergeAmplitudeEmbeddingInterpreter()
        def qfunc():
            qml.AmplitudeEmbedding(jax.numpy.array([[1, 0], [0, 1]]), wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([1, 0]), wires=1)
            qml.AmplitudeEmbedding(jax.numpy.array([[0, 1], [1, 0]]), wires=2)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[0].params["n_wires"], 3)
        assert jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_returned_op_is_not_cancelled(self):
        """Test that ops that are returned by the function being transformed are not cancelled."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qml.AmplitudeEmbedding(jax.numpy.array([1, 0]), wires=0)
            return qml.AmplitudeEmbedding(jax.numpy.array([1, 0]), wires=1)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[0].params["n_wires"], 1)
        assert jaxpr.eqns[1].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[1].params["n_wires"], 1)
        assert jaxpr.jaxpr.outvars[0] == jaxpr.eqns[1].outvars[0]


# pylint:disable=too-few-public-methods
class TestMergeAmplitudeEmbeddingPlxprTransform:
    """Test that the plxpr transform works as expected."""

    def test_merge_simple(self):
        """Test that the plxpr transform works correctly for a simple example."""

        def qfunc():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        args = ()
        transformed_jaxpr = merge_amplitude_embedding_plxpr_to_plxpr(
            jaxpr.jaxpr, jaxpr.consts, [], {}, *args
        )
        assert len(transformed_jaxpr.eqns) == 4
        assert transformed_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(transformed_jaxpr.eqns[0].params["n_wires"], 2)
        assert transformed_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestHigherOrderPrimitiveIntegration:
    """Test that the transform works correctly when applied with higher order primitives."""

    def test_ctrl_transform_prim(self):
        """Test that the transform works correctly when applied with ctrl_transform_prim."""

        def ctrl_fn():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.X(0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            qml.ctrl(ctrl_fn, [2, 3])()
            qml.RY(0, 1)

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == ctrl_transform_prim
        assert jaxpr.eqns[1].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert inner_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(inner_jaxpr.eqns[0].params["n_wires"], 2)
        assert inner_jaxpr.eqns[1].primitive == qml.X._primitive

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_transform_prim(self, lazy):
        """Test that the transform works correctly when applied with adjoint_transform_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f():
            def g():
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
                qml.X(0)
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)

            qml.adjoint(g, lazy=lazy)()

        jaxpr = jax.make_jaxpr(f)()
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == adjoint_transform_prim
        assert jaxpr.eqns[0].params["lazy"] == lazy

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(inner_jaxpr.eqns[0].params["n_wires"], 2)
        assert inner_jaxpr.eqns[1].primitive == qml.X._primitive

    def test_cond_prim(self):

        @MergeAmplitudeEmbeddingInterpreter()
        def f(x):
            @qml.cond(x > 2)
            def cond_f():
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
                return qml.expval(qml.Z(0))

            @cond_f.else_if(x > 1)
            def _():
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
                return qml.expval(qml.Y(0))

            @cond_f.otherwise
            def _():
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
                return qml.expval(qml.X(0))

            out = cond_f()
            return out

        args = (3,)
        jaxpr = jax.make_jaxpr(f)(*args)
        # First 2 primitives are the conditions for the true and elif branches
        assert jaxpr.eqns[2].primitive == cond_prim

        # True branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][0]
        assert qml.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qml.AmplitudeEmbedding._primitive,
            qml.Z._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

        # Elif branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][1]
        assert qml.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qml.AmplitudeEmbedding._primitive,
            qml.Y._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

        # Else branch
        branch = jaxpr.eqns[2].params["jaxpr_branches"][2]
        assert qml.math.allclose(branch.eqns[0].params["n_wires"], 2)
        expected_primitives = [
            qml.AmplitudeEmbedding._primitive,
            qml.X._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        assert all(
            eqn.primitive == exp_prim for eqn, exp_prim in zip(branch.eqns, expected_primitives)
        )

    def test_for_loop_prim(self):
        """Test that the transform works correctly when applied with for_loop_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(n):
            @qml.for_loop(n)
            def g(i):
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=i)

            g()

        jaxpr = jax.make_jaxpr(f)(3)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == for_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 1
        assert inner_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive

    def test_while_loop_prim(self):
        """Test that the transform works correctly when applied with while_loop_prim."""

        @MergeAmplitudeEmbeddingInterpreter()
        def f(n):
            @qml.while_loop(lambda i: i < n)
            def g(i):
                qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=i)
                return i + 1

            g(0)

        jaxpr = jax.make_jaxpr(f)(3)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == while_loop_prim

        inner_jaxpr = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert len(inner_jaxpr.eqns) == 2
        assert inner_jaxpr.eqns[1].primitive == qml.AmplitudeEmbedding._primitive

    def test_qnode_prim(self):
        """Test that the transform works correctly when applied with qnode_prim."""
        dev = qml.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        @qml.qnode(dev)
        def circuit():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.Hadamard(0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_qnode_prim_with_arguments(self):
        """Test that the transform works correctly when applied with qnode_prim."""
        dev = qml.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        @qml.qnode(dev)
        def circuit(state1):
            qml.AmplitudeEmbedding(state1, wires=0)
            qml.Hadamard(0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(jax.numpy.array([0.0, 1.0]))

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive == qml.AmplitudeEmbedding._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.Hadamard._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_grad_prim(self):
        """Test that the transform works correctly when applied with grad_prim."""

        dev = qml.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        def f(a, b):
            @qml.qnode(dev)
            def circuit(a, b):
                qml.AmplitudeEmbedding(jax.numpy.array([a, b]), wires=0)
                qml.Hadamard(0)
                qml.AmplitudeEmbedding(jax.numpy.array([a, b]), wires=1)
                return qml.expval(qml.Z(0))

            return qml.grad(circuit)(a, b)

        jaxpr = jax.make_jaxpr(f)(0.0, 1.0)

        assert jaxpr.eqns[0].primitive == grad_prim
        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive == qml.AmplitudeEmbedding._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.Hadamard._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_jacobian_prim(self):
        """Test that the transform works correctly when applied with jacobian_prim."""

        dev = qml.device("default.qubit", wires=2)

        @MergeAmplitudeEmbeddingInterpreter()
        def f(a, b):
            @qml.qnode(dev)
            def circuit(a, b):
                qml.AmplitudeEmbedding(jax.numpy.array([a, b]), wires=0)
                qml.Hadamard(0)
                qml.AmplitudeEmbedding(jax.numpy.array([a, b]), wires=1)
                return qml.expval(qml.Z(0))

            return qml.jacobian(circuit)(a, b)

        jaxpr = jax.make_jaxpr(f)(0.0, 1.0)

        assert jaxpr.eqns[0].primitive == jacobian_prim
        grad_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        qfunc_jaxpr = grad_jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[-4].primitive == qml.AmplitudeEmbedding._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.Hadamard._primitive
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestExpandPlxprTransformIntegration:
    """Test that the transform works with expand_plxpr_transform"""

    def test_example(self):
        """Test that the transform works with expand_plxpr_transform"""

        @qml.transforms.optimization.merge_amplitude_embedding
        def qfunc():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()

        assert (
            jaxpr.eqns[0].primitive
            == qml.transforms.optimization.merge_amplitude_embedding._primitive
        )

        transformed_qfunc = qml.capture.expand_plxpr_transforms(qfunc)
        transformed_jaxpr = jax.make_jaxpr(transformed_qfunc)()
        assert len(transformed_jaxpr.eqns) == 4
        assert transformed_jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(transformed_jaxpr.eqns[0].params["n_wires"], 2)
        assert transformed_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_decorator(self):
        """Test that the transform works with the decorator"""

        @qml.capture.expand_plxpr_transforms
        @qml.transforms.optimization.merge_amplitude_embedding
        def qfunc():
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=0)
            qml.Hadamard(wires=0)
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(qfunc)()
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.AmplitudeEmbedding._primitive
        assert qml.math.allclose(jaxpr.eqns[0].params["n_wires"], 2)
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive
