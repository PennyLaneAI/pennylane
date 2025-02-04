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
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
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
            qml.AmplitudeEmbedding(jax.numpy.array([0.0, 1.0]), wires=1)

        with pytest.raises(qml.DeviceError, match="applied in the same qubit"):
            jax.make_jaxpr(qfunc)()

    def test_merge_simple(self):
        """Test that the transform works correctly for a simple example."""

        @MergeAmplitudeEmbeddingInterpreter()
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
        assert transformed_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive


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
        assert transformed_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert transformed_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestHigherOrderPrimitiveIntegration:
    """Test that the transform works correctly when applied with higher order primitives."""

    def test_adjoint_transform_prim(self):
        pass

    def test_cond_prim(self):
        pass

    def test_for_loop_prim(self):
        pass

    def test_grad_prim(self):
        pass

    def test_jacobian_prim(self):
        pass

    def test_qnode_prim(self):
        pass

    def test_while_loop_prim(self):
        pass
