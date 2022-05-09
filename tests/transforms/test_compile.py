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

import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.wires import Wires

from pennylane.transforms.compile import compile
from pennylane.transforms import unitary_to_rot
from pennylane.transforms.optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
)

from test_optimization.utils import compare_operation_lists


def build_qfunc(wires):
    def qfunc(x, y, z):
        qml.Hadamard(wires=wires[0])
        qml.RZ(z, wires=wires[2])
        qml.CNOT(wires=[wires[2], wires[1]])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RX(x, wires=wires[0])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(-z, wires=wires[2])
        qml.RX(y, wires=wires[0])
        qml.PauliY(wires=wires[2])
        qml.CY(wires=[wires[1], wires[2]])
        return qml.expval(qml.PauliZ(wires=wires[0]))

    return qfunc


class TestCompile:
    """Unit tests for compile function."""

    def test_compile_invalid_pipeline(self):
        """Test that error is raised for an invalid function in the pipeline"""
        qfunc = build_qfunc([0, 1, 2])
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        transformed_qfunc = compile(pipeline=[cancel_inverses, isinstance])(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        with pytest.raises(ValueError, match="Invalid transform function"):
            transformed_qnode(0.1, 0.2, 0.3)

    def test_compile_invalid_num_passes(self):
        """Test that error is raised for an invalid number of passes."""
        qfunc = build_qfunc([0, 1, 2])
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        transformed_qfunc = compile(num_passes=1.3)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        with pytest.raises(ValueError, match="Number of passes must be an integer"):
            transformed_qnode(0.1, 0.2, 0.3)

    def test_compile_mixed_tape_qfunc_transform(self):
        """Test that we can interchange tape and qfunc transforms."""

        wires = [0, 1, 2]
        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=wires)

        pipeline = [
            commute_controlled(direction="right").tape_fn,
            cancel_inverses,
            merge_rotations().tape_fn,
        ]

        transformed_qfunc = compile(pipeline=pipeline)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)

        names_expected = ["Hadamard", "CNOT", "RX", "CY", "PauliY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires([wires[1], wires[2]]),
            Wires(wires[2]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(
        "transform_name,num_passes",
        [
            ("merge_rotations", 1),
            ("commute_controlled", 1),
            ("merge_rotations", 3),
            ("commute_controlled", 2),
        ],
    )
    def test_compile_mock_calls(self, transform_name, num_passes, mocker):
        """Test that functions in the pipeline are called the correct number of times."""

        class DummyTransforms:
            def run_pipeline(self):
                pipeline = [
                    qml.transforms.single_tape_transform(DummyTransforms.merge_rotations),
                    qml.transforms.single_tape_transform(DummyTransforms.commute_controlled),
                ]

                wires = [0, 1, 2]
                qfunc = build_qfunc(wires)
                dev = qml.device("default.qubit", wires=Wires(wires))

                transformed_qfunc = compile(pipeline=pipeline, num_passes=num_passes)(qfunc)
                transformed_qnode = qml.QNode(transformed_qfunc, dev)
                transformed_result = transformed_qnode(0.3, 0.4, 0.5)

            @staticmethod
            def merge_rotations(tape):
                return qml.transforms.merge_rotations.tape_fn(tape)

            @staticmethod
            def commute_controlled(tape):
                return qml.transforms.commute_controlled.tape_fn(tape, direction="left")

        spy = mocker.spy(DummyTransforms, transform_name)

        d = DummyTransforms()
        d.run_pipeline()

        assert len(spy.call_args_list) == num_passes


class TestCompileIntegration:
    """Integration tests to verify outputs of compilation pipelines."""

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_compile_empty_pipeline(self, wires):
        """Test that an empty pipeline returns the original function."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=wires)

        qnode = qml.QNode(qfunc, dev)

        transformed_qfunc = compile(pipeline=[])(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = [op.name for op in qnode.qtape.operations]
        wires_expected = [op.wires for op in qnode.qtape.operations]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_compile_default_pipeline(self, wires):
        """Test that the default pipeline returns the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        transformed_qfunc = compile()(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "CY", "PauliY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires([wires[1], wires[2]]),
            Wires(wires[2]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_compile_pipeline_with_non_default_arguments(self, wires):
        """Test that using non-default arguments returns the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        pipeline = [
            commute_controlled(direction="left"),
            cancel_inverses,
            merge_rotations(atol=1e-6),
        ]

        transformed_qfunc = compile(pipeline=pipeline)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "PauliY", "CY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_compile_multiple_passes(self, wires):
        """Test that running multiple passes produces the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        # Rotation merging will not occur at all until commuting gates are
        # pushed through
        pipeline = [merge_rotations, commute_controlled(direction="left"), cancel_inverses]

        transformed_qfunc = compile(pipeline=pipeline, num_passes=2)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["Hadamard", "CNOT", "RX", "PauliY", "CY"]
        wires_expected = [
            Wires(wires[0]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    @pytest.mark.parametrize(("wires"), [["a", "b", "c"], [0, 1, 2], [3, 1, 2], [0, "a", 4]])
    def test_compile_decompose_into_basis_gates(self, wires):
        """Test that running multiple passes produces the correct results."""

        qfunc = build_qfunc(wires)
        dev = qml.device("default.qubit", wires=Wires(wires))

        qnode = qml.QNode(qfunc, dev)

        pipeline = [commute_controlled(direction="left"), cancel_inverses, merge_rotations]

        basis_set = ["CNOT", "RX", "RY", "RZ"]

        transformed_qfunc = compile(pipeline=pipeline, basis_set=basis_set)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        original_result = qnode(0.3, 0.4, 0.5)
        transformed_result = transformed_qnode(0.3, 0.4, 0.5)
        assert np.allclose(original_result, transformed_result)

        names_expected = [
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "CNOT",
            "RX",
            "RZ",
            "RY",
            "RZ",
            "RY",
            "CNOT",
            "RY",
            "CNOT",
        ]

        wires_expected = [
            Wires(wires[0]),
            Wires(wires[0]),
            Wires(wires[0]),
            Wires(wires[2]),
            Wires([wires[2], wires[1]]),
            Wires(wires[0]),
            Wires(wires[1]),
            Wires(wires[2]),
            Wires(wires[2]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
            Wires(wires[2]),
            Wires([wires[1], wires[2]]),
        ]

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)

    def test_compile_template(self):
        """Test that functions with templates are correctly expanded and compiled."""

        # Push commuting gates to the right and merging rotations gives a circuit
        # with alternating RX and CNOT gates
        def qfunc(x, params):
            qml.templates.AngleEmbedding(x, wires=range(3))
            qml.templates.BasicEntanglerLayers(params, wires=range(3))
            return qml.expval(qml.PauliZ(wires=2))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)

        pipeline = [commute_controlled, merge_rotations]
        transformed_qfunc = compile(pipeline=pipeline)(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        x = np.array([0.1, 0.2, 0.3])
        params = np.ones((2, 3))

        original_result = qnode(x, params)
        transformed_result = transformed_qnode(x, params)
        assert np.allclose(original_result, transformed_result)

        names_expected = ["RX", "CNOT"] * 6
        wires_expected = [
            Wires(0),
            Wires([0, 1]),
            Wires(1),
            Wires([1, 2]),
            Wires(2),
            Wires([2, 0]),
        ] * 2

        compare_operation_lists(transformed_qnode.qtape.operations, names_expected, wires_expected)


def qfunc(x, params):
    qml.templates.AngleEmbedding(x, wires=range(3))
    qml.templates.BasicEntanglerLayers(params, wires=range(3))
    return qml.expval(qml.PauliZ(wires=2))


pipeline = [commute_controlled(direction="left"), merge_rotations]

transformed_qfunc = compile(pipeline=pipeline)(qfunc)

dev = qml.device("default.qubit", wires=3)

expected_op_list = ["RX"] * 3 + ["CNOT", "CNOT", "RX", "CNOT", "RX", "RX"] + ["CNOT"] * 3

expected_wires_list = [
    Wires(0),
    Wires(1),
    Wires(2),
    Wires([0, 1]),
    Wires([1, 2]),
    Wires(0),
    Wires([2, 0]),
    Wires(1),
    Wires(2),
    Wires([0, 1]),
    Wires([1, 2]),
    Wires([2, 0]),
]


class TestCompileInterfaces:
    """Test that the top-level compile function works across all interfaces."""

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_compile_autograd(self, diff_method):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev, interface="autograd", diff_method=diff_method)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="autograd", diff_method=diff_method
        )

        x = np.array([0.1, 0.2, 0.3], requires_grad=False)
        params = np.ones((2, 3))

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(x, params), transformed_qnode(x, params))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(x, params), qml.grad(transformed_qnode)(x, params)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.torch
    def test_compile_torch(self):
        """Test QNode and gradient in torch interface."""
        import torch

        original_qnode = qml.QNode(qfunc, dev, interface="torch", diff_method="parameter-shift")
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="torch", diff_method="parameter-shift"
        )

        original_x = torch.tensor([0.3, -0.2, 0.8], requires_grad=False)
        original_params = torch.ones((2, 3), requires_grad=True)

        transformed_x = torch.tensor([0.3, -0.2, 0.8], requires_grad=False)
        transformed_params = torch.ones((2, 3), requires_grad=True)

        original_result = original_qnode(original_x, original_params)
        transformed_result = transformed_qnode(transformed_x, transformed_params)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_params.grad, transformed_params.grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_compile_tf(self, diff_method):
        """Test QNode and gradient in tensorflow interface."""
        import tensorflow as tf

        original_qnode = qml.QNode(qfunc, dev, interface="tf", diff_method=diff_method)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="tf", diff_method=diff_method
        )

        original_x = tf.Variable([0.8, -0.6, 0.4], dtype=tf.float64)
        original_params = tf.Variable(tf.ones((2, 3), dtype=tf.float64))

        transformed_x = tf.Variable([0.8, -0.6, 0.4], dtype=tf.float64)
        transformed_params = tf.Variable(tf.ones((2, 3), dtype=tf.float64))

        original_result = original_qnode(original_x, original_params)
        transformed_result = transformed_qnode(transformed_x, transformed_params)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        with tf.GradientTape() as tape:
            loss = original_qnode(original_x, original_params)
        original_grad = tape.gradient(loss, original_params)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_x, transformed_params)
        transformed_grad = tape.gradient(loss, transformed_params)

        assert qml.math.allclose(original_grad, transformed_grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_compile_jax(self, diff_method):
        """Test QNode and gradient in JAX interface."""
        import jax
        from jax import numpy as jnp

        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        original_qnode = qml.QNode(qfunc, dev, interface="jax", diff_method=diff_method)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="jax", diff_method=diff_method
        )

        x = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
        params = jnp.ones((2, 3), dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(x, params), transformed_qnode(x, params))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode, argnums=(1))(x, params),
            jax.grad(transformed_qnode, argnums=(1))(x, params),
            atol=1e-7,
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_compile_jax_jit(self, diff_method):
        """Test that compilation pipelines work with jax.jit, unitary_to_rot, and fusion."""
        import jax
        from jax import numpy as jnp
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2)

        def test_qfunc(x):
            qml.Rot(x, x + 1, x + 2, wires=0)
            qml.RX(2 * x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(qml.RX.compute_matrix(x), wires=1)
            qml.QubitUnitary(qml.RZ.compute_matrix(x + 1), wires=1)
            return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

        original_qnode = qml.QNode(test_qfunc, dev, interface="jax", diff_method=diff_method)

        pipeline = [cancel_inverses, unitary_to_rot, single_qubit_fusion]

        compiled_qfunc = qml.compile(pipeline=pipeline)(test_qfunc)
        compiled_qnode = qml.QNode(compiled_qfunc, dev, interface="jax", diff_method=diff_method)

        jitted_compiled_qnode = jax.jit(compiled_qnode)

        x = jnp.array(0.1, dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(x), jitted_compiled_qnode(x))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(x),
            jax.grad(jitted_compiled_qnode)(x),
            atol=1e-7,
        )
