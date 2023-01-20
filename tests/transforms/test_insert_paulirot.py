# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the qfunc_transform ``insert_paulirot`` and its utility functions.
"""

from functools import partial
import pytest

import numpy as np

import pennylane as qml
from pennylane.ops.qubit.matrix_ops import pauli_basis, pauli_words
from pennylane.transforms.insert_paulirot import (
    insert_paulirot,
    get_one_parameter_generators,
    get_one_parameter_coeffs,
)


class TestGetOneParameterGenerators:
    """Tests for the effective generators computing function
    get_one_parameter_generators."""

    @pytest.mark.jax
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_Omegas_jax(self, n, use_jit):
        """Test that generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        fn = (
            jax.jit(get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_generators
        )
        Omegas = fn(theta, n, "jax")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(jnp.allclose(O.conj().T, -O) for O in Omegas)

    @pytest.mark.tf
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_Omegas_tf(self, n):
        """Test that generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        Omegas = get_one_parameter_generators(theta, n, "tf")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_Omegas_torch(self, n):
        """Test that generators are computed correctly in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "torch")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_Omegas_autograd(self, n):
        """Test that generators are computed correctly in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "autograd")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    def test_Omegas_raises_autograd(self):
        """Test that computing generators raises an error when attempting to use Autograd."""
        with pytest.raises(NotImplementedError, match="expm is not differentiable in Autograd"):
            get_one_parameter_generators(None, None, "autograd")

    def test_Omegas_raises_unknown_interface(self):
        """Test that computing generators raises an error when attempting
        to use an unknown interface."""
        with pytest.raises(NotImplementedError, match="The interface interface is not supported"):
            get_one_parameter_generators(None, None, "interface")


@pytest.mark.parametrize("n", [1, 2])
class TestGetOneParameterGeneratorsDiffability:
    """Tests for the effective generators computing function
    get_one_parameter_generators to be differentiable."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_Omegas_jacobian_jax(self, n, use_jit):
        """Test that generators are differentiable in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d), dtype=jnp.complex128)
        fn = (
            jax.jit(get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_generators
        )
        dOmegas = jax.jacobian(fn, holomorphic=True)(theta, n, "jax")
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.tf
    def test_Omegas_jacobian_tf(self, n):
        """Test that generators are differentiable in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        with tf.GradientTape() as t:
            Omegas = get_one_parameter_generators(theta, n, "tf")
        dOmegas = t.jacobian(Omegas, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.torch
    def test_Omegas_jacobian_torch(self, n):
        """Test that generators are differentiable in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        def fn(theta):
            return get_one_parameter_generators(theta, n, "torch")
        dOmegas = torch.autograd.functional.jacobian(fn, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_Omegas_jacobian_autograd(self, n):
        """Test that generators are differentiable in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        dOmegas = qml.jacobian(get_one_parameter_generators)(theta, n, "autograd")
        assert dOmegas.shape == (d, 2**n, 2**n, d)


@pytest.mark.parametrize("n", [1, 2, 3])
class TestGetOneParameterCoeffs:
    """Tests for the coefficients of effective generators computing function
    get_one_parameter_coeffs."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_omegas_jax(self, n, use_jit):
        """Test that the coefficients of the generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        fn = (
            jax.jit(get_one_parameter_coeffs, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_coeffs
        )
        omegas = fn(theta, n, "jax")
        assert omegas.shape == (d, d)
        assert jnp.allclose(omegas.real, 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = jnp.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "jax")
        assert jnp.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.tf
    def test_omegas_tf(self, n):
        """Test that the coefficients of the generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        omegas = get_one_parameter_coeffs(theta, n, "tf")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "tf")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.torch
    def test_omegas_torch(self, n):
        """Test that the coefficients of the generators are computed correctly in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        omegas = get_one_parameter_coeffs(theta, n, "torch")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "torch")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_omegas_autograd(self, n):
        """Test that the coefficients of the generators are computed correctly in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        omegas = get_one_parameter_coeffs(theta, n, "autograd")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "autograd")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)


def make_ones(interface):
    """Output a function that creates np.ones in an interface and import
    the required interface."""
    if interface == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        def ones(x):
            return jax.numpy.array(jax.numpy.ones(x))
    elif interface == "torch":
        import torch

        ones = partial(torch.ones, requires_grad=True)
    elif interface == "tensorflow":
        import tensorflow as tf

        def ones(x):
            return tf.Variable(np.ones(x))
    return ones


class TestInsertPauliRotTape:
    """Tests for the qfunc_transform insert_paulirot when applied to a tape directly."""

    @pytest.mark.parametrize(
        "ops",
        [
            [qml.RX(0, 0), qml.Hadamard("aux"), qml.RY(0.2, "SpecialUnitary")],
            [],
            [qml.RZ(np.array([0.4, 0.2, 0.1]), 9)],
        ],
    )
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.measurements.ExpectationMP(qml.PauliX(0))],
            [
                qml.measurements.ProbabilityMP(wires=[1]),
                qml.measurements.ExpectationMP(qml.PauliZ(4)),
            ],
        ],
    )
    def test_does_nothing_without_special_unitary(self, ops, measurements):
        """Tests that the transform does not modify a tape without
        SpecialUnitary operations in it."""
        tape = qml.tape.QuantumScript(ops, measurements)
        with qml.queuing.AnnotatedQueue() as q:
            insert_paulirot(tape)
        new_tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape) == len(new_tape)
        assert all(op.name == new_op.name for op, new_op in zip(tape, new_tape))
        assert all(qml.math.allclose(op.data, new_op.data) for op, new_op in zip(tape, new_tape))

    @pytest.mark.parametrize(
        "ops",
        [
            [qml.SpecialUnitary(np.ones(3), 0)],
            [qml.SpecialUnitary(np.ones(15), ["aux", 0])],
        ],
    )
    def test_does_nothing_with_numpy_special_unitary(self, ops):
        """Tests that the transform does not modify a tape with a
        SpecialUnitary that depends on a NumPy array in it."""
        tape = qml.tape.QuantumScript(ops)
        with qml.queuing.AnnotatedQueue() as q:
            insert_paulirot(tape)
        new_tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape) == len(new_tape)
        assert all(op.name == new_op.name for op, new_op in zip(tape, new_tape))
        assert all(qml.math.allclose(op.data, new_op.data) for op, new_op in zip(tape, new_tape))

    @pytest.mark.parametrize(
        "interface",
        [
            pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch),
            pytest.param("tensorflow", marks=pytest.mark.tf),
        ],
    )
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.measurements.ExpectationMP(qml.PauliX(0))],
            [
                qml.measurements.ProbabilityMP(wires=[1]),
                qml.measurements.ExpectationMP(qml.PauliZ(4)),
            ],
        ],
    )
    def test_inserts_correctly_for_special_unitary(self, measurements, interface):
        """Tests that the transform correctly inserts Pauli rotations into a tape with
        SpecialUnitary operations in it. Note that the insertion requires an auto-differentiation
        interface other than Autograd."""
        ones = make_ones(interface)

        ops = [qml.RX(0, 0), qml.Hadamard("aux"), qml.SpecialUnitary(ones(3), wires=[0])]
        tape = qml.tape.QuantumScript(ops, measurements)
        with qml.queuing.AnnotatedQueue() as q:
            insert_paulirot(tape)
        new_tape = qml.tape.QuantumScript.from_queue(q)
        assert len(new_tape) == len(tape) + 3
        assert new_tape.operations[0] == tape.operations[0]
        assert new_tape.operations[1] == tape.operations[1]
        assert qml.equal(
            new_tape.operations[-1],
            tape.operations[2],
            check_interface=False,
            check_trainability=False,
        )
        exp_pauli_rots = new_tape.operations[2:-1]
        assert all(op.name == "PauliRot" for op in exp_pauli_rots)
        assert all(
            op.hyperparameters == {"pauli_word": "XYZ"[i]} for i, op in enumerate(exp_pauli_rots)
        )
        assert all(qml.math.isclose(op.data[0], op.data[0] * 0) for op in exp_pauli_rots)

        ops = [qml.SpecialUnitary(ones(3), wires=[0])]
        tape = qml.tape.QuantumScript(ops, measurements)
        with qml.queuing.AnnotatedQueue() as q:
            insert_paulirot(tape)
        new_tape = qml.tape.QuantumScript.from_queue(q)
        assert len(new_tape) == len(tape) + 3
        assert qml.equal(
            new_tape.operations[-1],
            tape.operations[0],
            check_interface=False,
            check_trainability=False,
        )
        exp_pauli_rots = new_tape.operations[:3]
        assert all(op.name == "PauliRot" for op in exp_pauli_rots)
        assert all(
            op.hyperparameters == {"pauli_word": "XYZ"[i]} for i, op in enumerate(exp_pauli_rots)
        )
        assert all(qml.math.isclose(op.data[0], op.data[0] * 0) for op in exp_pauli_rots)

        ops = [
            qml.SpecialUnitary(ones(3), wires=[0]),
            qml.RZ(0.2, 1),
            qml.SpecialUnitary(ones(15), wires=[0, 2]),
        ]
        tape = qml.tape.QuantumScript(ops, measurements)
        with qml.queuing.AnnotatedQueue() as q:
            insert_paulirot(tape)
        new_tape = qml.tape.QuantumScript.from_queue(q)
        assert len(new_tape) == len(tape) + 3 + 15
        assert qml.equal(
            new_tape.operations[3],
            tape.operations[0],
            check_interface=False,
            check_trainability=False,
        )
        assert new_tape.operations[4] == tape.operations[1]
        assert qml.equal(
            new_tape.operations[-1],
            tape.operations[2],
            check_interface=False,
            check_trainability=False,
        )
        exp_pauli_rots = new_tape.operations[:3] + new_tape.operations[5:-1]
        assert all(op.name == "PauliRot" for op in exp_pauli_rots)
        assert all(
            op.hyperparameters == {"pauli_word": "XYZ"[i]}
            for i, op in enumerate(exp_pauli_rots[:3])
        )
        words = pauli_words(2)
        assert all(
            op.hyperparameters == {"pauli_word": words[i]}
            for i, op in enumerate(exp_pauli_rots[3:])
        )
        assert all(qml.math.isclose(op.data[0], op.data[0] * 0) for op in exp_pauli_rots)


def qfunc_without_SU(x):
    """A quantum function without SpecialUnitary."""
    qml.RX(x, 0)
    return qml.expval(qml.PauliZ(0))


def qfunc_with_single_SU(theta):
    """A quantum function with a single SpecialUnitary."""
    qml.RX(0.2, 0)
    qml.SpecialUnitary(theta, 0)
    return qml.expval(qml.PauliZ(0))


def qfunc_with_two_SUs(theta, phi):
    """A quantum function with two SpecialUnitary gates."""
    qml.RX(2.31, 0)
    qml.SpecialUnitary(theta, 0)
    qml.SpecialUnitary(phi, [1, 0])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


qfuncs = [qfunc_without_SU, qfunc_with_single_SU, qfunc_with_two_SUs]
arg_dims = [(1,), (3,), (3, 15)]


@pytest.mark.parametrize("qfunc, dims", list(zip(qfuncs, arg_dims)))
@pytest.mark.parametrize("new_diff_method", ["backprop", "parameter-shift"])
class TestInsertPauliRotQfunc:
    """Tests for the qfunc_transform insert_paulirot when applied to a qfunc."""

    def test_trainability_jax(self, qfunc, new_diff_method, dims):
        """Test trainability and that the parameter dependencies are correct, so that the Jacobian
        is correct in JAX. When new_diff_method="backprop", the automatic differentiation of the
        inserted PauliRots is checked. When new_diff_method="parameter-shift", the correct
        computation via parameter shifts is checked."""
        import jax

        jax.config.update("jax_enable_x64", True)
        new_qfunc = insert_paulirot(qfunc)

        args = [jax.numpy.array(0.2) if dim == 1 else jax.numpy.linspace(1, 2, dim) for dim in dims]
        # Test running the new function
        new_qfunc(*args)

        # Test QNode jacobians
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(new_qfunc, dev, interface="jax")
        new_qnode = qml.QNode(new_qfunc, dev, interface="jax", diff_method=new_diff_method)
        jac = jax.jacobian(qnode, argnums=list(range(len(dims))))(*args)
        new_jac = jax.jacobian(new_qnode, argnums=list(range(len(dims))))(*args)
        print(jac, new_jac)
        assert all(qml.math.shape(j) == qml.math.shape(nj) for j, nj in zip(jac, new_jac))
        assert all(qml.math.allclose(j, nj) for j, nj in zip(jac, new_jac))

    def test_trainability_torch(self, qfunc, new_diff_method, dims):
        """Test trainability and that the parameter dependencies are correct, so that the Jacobian
        is correct in Torch. When new_diff_method="backprop", the automatic differentiation of the
        inserted PauliRots is checked. When new_diff_method="parameter-shift", the correct
        computation via parameter shifts is checked."""
        import torch

        new_qfunc = insert_paulirot(qfunc)

        args = tuple(
            torch.tensor(0.2, requires_grad=True)
            if dim == 1
            else torch.linspace(1, 2, dim, requires_grad=True)
            for dim in dims
        )
        # Test running the new function
        new_qfunc(*args)

        # Test QNode jacobians
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(new_qfunc, dev, interface="torch")
        new_qnode = qml.QNode(new_qfunc, dev, interface="torch", diff_method=new_diff_method)
        jac = torch.autograd.functional.jacobian(qnode, args)
        new_jac = torch.autograd.functional.jacobian(new_qnode, args)
        assert all(qml.math.shape(j) == qml.math.shape(nj) for j, nj in zip(jac, new_jac))
        assert all(qml.math.allclose(j, nj) for j, nj in zip(jac, new_jac))

    def test_trainability_tf(self, qfunc, new_diff_method, dims):
        """Test trainability and that the parameter dependencies are correct, so that the Jacobian
        is correct in Tensorflow. When new_diff_method="backprop", the automatic differentiation of
        the inserted PauliRots is checked. When new_diff_method="parameter-shift", the correct
        computation via parameter shifts is checked."""
        import tensorflow as tf

        new_qfunc = insert_paulirot(qfunc)

        args = tuple(tf.Variable(0.2 if dim == 1 else np.linspace(1, 2, dim)) for dim in dims)
        # Test running the new function
        new_qfunc(*args)

        # Test QNode jacobians
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(new_qfunc, dev, interface="tensorflow")
        with tf.GradientTape() as t:
            out = qnode(*args)
        jac = t.jacobian(out, args)
        new_qnode = qml.QNode(new_qfunc, dev, interface="tensorflow", diff_method=new_diff_method)
        with tf.GradientTape() as t:
            out = new_qnode(*args)
        new_jac = t.jacobian(out, args)
        assert all(qml.math.shape(j) == qml.math.shape(nj) for j, nj in zip(jac, new_jac))
        assert all(qml.math.allclose(j, nj) for j, nj in zip(jac, new_jac))
