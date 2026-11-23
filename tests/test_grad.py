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
Unit tests for qp.grad
"""

import jax
import pytest

import pennylane as qp


@pytest.mark.parametrize("grad_fn", (qp.grad, qp.jacobian))
def test_kwarg_errors_without_qjit(grad_fn):
    """Test that errors are raised with method and h when qjit is not active."""

    def f(x):
        return x**2

    x = qp.numpy.array(0.5)

    with pytest.raises(ValueError, match="method = 'fd' unsupported without QJIT."):
        grad_fn(f, method="fd")(x)

    with pytest.raises(ValueError, match="unsupported without QJIT. "):
        grad_fn(f, h=1e-6)(0.5)


def test_grad_name():
    """Test that grad has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qp.grad(f).__name__ == "<grad: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qp.grad(A()).__name__ == "<grad: A>"


def test_jacobian_name():
    """Test that jacobian has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qp.jacobian(f).__name__ == "<jacobian: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qp.jacobian(A()).__name__ == "<jacobian: A>"


def test_value_and_grad_name():
    """Test that value_and_grad has name associated with it for the later mlir op."""

    def f(x):
        return x**2

    assert qp.value_and_grad(f).__name__ == "<value_and_grad: f>"

    class A:

        def __repr__(self):
            return "A"

        def __call__(self, x):
            return x**2

    assert qp.value_and_grad(A()).__name__ == "<value_and_grad: A>"


def test_vjp_without_qjit():
    """Test that an error is raised when using VJP without QJIT."""

    def vjp(params, cotangent):
        def f(x):
            y = [qp.math.sin(x[0]), x[1] ** 2, x[0] * x[1]]
            return qp.math.stack(y)

        return qp.vjp(f, [params], [cotangent])

    x = qp.numpy.array([0.1, 0.2])
    dy = qp.numpy.array([-0.5, 0.1, 0.3])

    with pytest.raises(
        qp.exceptions.CompileError,
        match="PennyLane does not support the VJP function without QJIT.",
    ):
        vjp(x, dy)


def test_jvp_without_qjit():
    """Test that an error is raised when using JVP without QJIT."""

    def jvp(params, dparams):
        def f(x):
            y = [qp.math.sin(x[0]), x[1] ** 2, x[0] * x[1]]
            return qp.math.stack(y)

        return qp.jvp(f, [params], [dparams])

    x = qp.numpy.array([0.1, 0.2])
    dy = qp.numpy.array([-0.5, 0.1])

    with pytest.raises(
        qp.exceptions.CompileError,
        match="PennyLane does not support the JVP function without QJIT.",
    ):
        jvp(x, dy)


def test_value_and_grad_without_qjit():
    """Test that an error is raised when using value_and_grad without QJIT."""

    def value_and_grad(params):
        def f(x):
            y = [qp.math.sin(x[0]), x[1] ** 2, x[0] * x[1]]
            return qp.math.stack(y)

        return qp.value_and_grad(f)(params)

    x = qp.numpy.array([0.1, 0.2])

    with pytest.raises(
        qp.exceptions.CompileError,
        match="PennyLane does not support the value_and_grad function without QJIT.",
    ):
        value_and_grad(x)


@pytest.mark.jax
def test_jax_grad_keywords_without_forward_cache():
    """JAX gradients respect keyword arguments without caching forward values."""

    def cost(first, second, *, scale):
        return first**2 + scale * first * second

    first, second = jax.numpy.array(2.0), jax.numpy.array(3.0)
    derivative = qp.grad(cost, argnums=0)
    assert derivative.forward is None
    assert qp.math.allclose(derivative(first, second, scale=4.0), 16.0)
    assert derivative.forward is None
    assert qp.math.allclose(jax.jit(derivative)(first, second, scale=4.0), 16.0)
    assert derivative.forward is None


@pytest.mark.jax
@pytest.mark.parametrize("argnums", [0, 1, [0], [1], (0, 1)])
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_jax_argnums(argnums, diff_function):
    """Explicit argument indices preserve scalar-versus-sequence result structure."""

    def cost(first, second, *, scale):
        value = scale * first * second
        if diff_function is qp.jacobian:
            return qp.math.stack([value, first**2])
        return value

    first, second = jax.numpy.array(2.0), jax.numpy.array(3.0)
    derivative = diff_function(cost, argnums=argnums)(first, second, scale=3.0)
    expected = [9.0, 6.0] if diff_function is qp.grad else [[9.0, 4.0], [6.0, 0.0]]
    if isinstance(argnums, int):
        assert not isinstance(derivative, tuple)
        assert qp.math.allclose(derivative, expected[argnums])
    else:
        assert isinstance(derivative, tuple)
        assert len(derivative) == len(argnums)
        for index, result in zip(argnums, derivative, strict=True):
            assert qp.math.allclose(result, expected[index])


@pytest.mark.jax
@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "adjoint"])
def test_jax_qnode_gradients(diff_method):
    """qp.grad and qp.jacobian differentiate scalar and vector JAX QNodes."""

    @qp.qnode(qp.device("default.qubit"), interface="jax", diff_method=diff_method)
    def scalar_circuit(angle):
        qp.RY(angle, 0)
        return qp.expval(qp.Z(0))

    @qp.qnode(qp.device("default.qubit"), interface="jax", diff_method=diff_method)
    def vector_circuit(angle):
        qp.RY(angle, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    angle = jax.numpy.array(0.3)
    derivative = qp.grad(scalar_circuit)
    assert qp.math.allclose(derivative(angle), -qp.math.sin(angle))
    jacobian = qp.jacobian(vector_circuit)(angle)
    expected = qp.math.stack([-qp.math.sin(angle), qp.math.cos(angle)])
    assert tuple(qp.math.shape(component) for component in jacobian) == ((), ())
    assert qp.math.allclose(qp.math.stack(jacobian), expected)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_jax_higher_derivatives(diff_function):
    """JAX derivative results can be differentiated again."""
    derivative = diff_function(lambda value: value**3)
    assert qp.math.allclose(qp.grad(derivative)(jax.numpy.array(2.0)), 12.0)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_jax_default_and_jit(diff_function):
    """JAX defaults to the first argument and remains compatible with JIT."""

    def cost(first, second):
        return first**2 + first * second

    first, second = jax.numpy.array(2.0), jax.numpy.array(3.0)
    derivative = diff_function(cost)
    assert qp.math.allclose(derivative(first, second), 7.0)
    assert qp.math.allclose(jax.jit(derivative)(first, second), 7.0)


@pytest.mark.jax
def test_jax_pytree_jacobian():
    """JAX Jacobians support structured inputs and outputs."""
    argument = {"value": jax.numpy.array(2.0)}
    result = qp.jacobian(lambda data: {"square": data["value"] ** 2})(argument)
    assert qp.math.allclose(result["square"]["value"], 4.0)


@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_autograd_with_mixed_dictionary_keys(diff_function):
    """Interface selection allows dictionaries with heterogeneous keys."""
    x = qp.numpy.array(2.0, requires_grad=False)
    offsets = {1: 2.0, "a": 3.0}
    derivative = diff_function(lambda offset, value: value**2 + sum(offset.values()))
    with pytest.warns(UserWarning, match="no trainable parameters"):
        assert derivative(offsets, x) == ()


def test_autograd_jacobian_with_iterator_argnums():
    """Backend selection and repeated calls reuse all indices from a one-shot iterator."""
    derivative = qp.jacobian(lambda x, y: x**2 + x * y, argnums=iter((0, 1)))
    x, y = qp.numpy.array(2.0), qp.numpy.array(3.0)
    assert qp.math.allclose(derivative(x, y), (7.0, 2.0))
    assert qp.math.allclose(derivative(x, y), (7.0, 2.0))


@pytest.mark.jax
def test_jax_jacobian_output_pytree_with_multiple_arguments():
    """The output pytree surrounds the tuple of argument derivatives."""
    derivative = qp.jacobian(lambda x, y: (x**2, x * y), argnums=(0, 1))
    result = derivative(jax.numpy.array(2.0), jax.numpy.array(3.0))
    expected = ((4.0, 0.0), (3.0, 2.0))
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(expected)
    assert qp.math.allclose(result, expected)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_autograd_with_jax_auxiliary_data(diff_function):
    """Auxiliary JAX data does not change inference of trainable NumPy arguments."""
    first, second = qp.numpy.array(2.0), qp.numpy.array(3.0)
    data = jax.numpy.array([1.0])
    derivative = diff_function(lambda x, y, aux: x**2 + y**2)
    assert qp.math.allclose(derivative(first, second, aux=data), (4.0, 6.0))
    assert qp.math.allclose(derivative(first, second, data), (4.0, 6.0))


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_dispatch_uses_explicit_argnums(diff_function):
    """Explicit argument selection excludes auxiliary JAX positional arguments."""
    derivative = diff_function(lambda data, value: value**2, argnums=1)
    value = qp.numpy.array(3.0)
    assert qp.math.allclose(derivative(jax.numpy.array([1.0]), value), 6.0)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_jax_auxiliary_data_before_trainable_parameter(diff_function):
    """Default Autograd inference also applies when auxiliary JAX data comes first."""
    derivative = diff_function(lambda data, value: value**2)
    assert qp.math.allclose(derivative(jax.numpy.array([1.0]), qp.numpy.array(3.0)), 6.0)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_registered_dictionary_subclass(diff_function):
    """Registered dictionary subclasses expose leaves through their custom flattening."""

    @jax.tree_util.register_pytree_node_class
    class Parameters(dict):
        """Store a parameter as an attribute rather than a dictionary entry."""

        def __init__(self, value):
            super().__init__()
            self.value = value

        def tree_flatten(self):
            """Return the differentiable parameter."""
            return (self.value,), None

        @classmethod
        def tree_unflatten(cls, _, children):
            """Rebuild the parameter container."""
            return cls(children[0])

    derivative = diff_function(lambda params: params.value**2)
    result = derivative(Parameters(jax.numpy.array(2.0)))
    assert isinstance(result, Parameters)
    assert qp.math.allclose(result.value, 4.0)
