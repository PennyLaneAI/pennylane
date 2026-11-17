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


@pytest.fixture(
    name="native_interface",
    params=[
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
    ],
)
def native_interface_fixture(request):
    """Select an installed native differentiation framework."""
    pytest.importorskip(request.param)
    return request.param


def test_native_grad_forward(native_interface):
    """Native gradients retain the forward value and respect keyword arguments."""

    def cost(first, second, *, scale):
        return first**2 + scale * first * second

    first = qp.math.asarray(2.0, like=native_interface, requires_grad=True)
    second = qp.math.asarray(3.0, like=native_interface, requires_grad=True)
    derivative = qp.grad(cost, argnums=0)
    assert derivative.forward is None
    assert qp.math.allclose(derivative(first, second, scale=4.0), 16.0)
    assert qp.math.allclose(derivative.forward, 28.0)
    assert qp.math.get_interface(derivative.forward) == native_interface


@pytest.mark.parametrize("argnums", [0, 1, [0], [1], (0, 1)])
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_native_argnums(native_interface, argnums, diff_function):
    """Explicit argument indices preserve scalar-versus-sequence result structure."""

    def cost(first, second, *, scale):
        value = scale * first * second
        if diff_function is qp.jacobian:
            return qp.math.stack([value, first**2])
        return value

    first = qp.math.asarray(2.0, like=native_interface, requires_grad=True)
    second = qp.math.asarray(3.0, like=native_interface, requires_grad=True)
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


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "adjoint"])
def test_native_qnode_gradients(native_interface, diff_method):
    """Public wrappers differentiate scalar and vector QNodes through native interfaces."""

    @qp.qnode(qp.device("default.qubit"), interface=native_interface, diff_method=diff_method)
    def scalar_circuit(angle):
        qp.RY(angle, 0)
        return qp.expval(qp.Z(0))

    @qp.qnode(qp.device("default.qubit"), interface=native_interface, diff_method=diff_method)
    def vector_circuit(angle):
        qp.RY(angle, 0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(0))

    angle = qp.math.asarray(0.3, like=native_interface, requires_grad=True)
    derivative = qp.grad(scalar_circuit)
    assert qp.math.allclose(derivative(angle), -qp.math.sin(angle))
    assert qp.math.allclose(derivative.forward, qp.math.cos(angle))
    jacobian = qp.jacobian(vector_circuit)(angle)
    expected = qp.math.stack([-qp.math.sin(angle), qp.math.cos(angle)])
    assert tuple(qp.math.shape(component) for component in jacobian) == ((), ())
    assert qp.math.allclose(qp.math.stack(jacobian), expected)


@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_native_higher_derivatives(native_interface, diff_function):
    """Native derivative results remain connected for nested differentiation."""
    argument = qp.math.asarray(2.0, like=native_interface, requires_grad=True)
    derivative = diff_function(lambda value: value**3)
    assert qp.math.allclose(qp.grad(derivative)(argument), 12.0)


@pytest.mark.jax
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_jax_default_and_jit(diff_function):
    """JAX defaults to the first argument and remains compatible with JIT."""
    jax = pytest.importorskip("jax")

    def cost(first, second):
        return first**2 + first * second

    first, second = jax.numpy.array(2.0), jax.numpy.array(3.0)
    derivative = diff_function(cost)
    assert qp.math.allclose(derivative(first, second), 7.0)
    assert qp.math.allclose(jax.jit(derivative)(first, second), 7.0)


@pytest.mark.jax
@pytest.mark.parametrize("eager_first", [False, True])
def test_jax_forward_remains_eager_value_after_jit(eager_first):
    """Compiled JAX executions cannot leak tracers through the cached forward value."""
    jax = pytest.importorskip("jax")
    derivative = qp.grad(lambda value: value**3)
    if eager_first:
        assert qp.math.allclose(derivative(jax.numpy.array(2.0)), 12.0)
    forward = derivative.forward
    compiled_derivative = jax.jit(derivative)
    assert qp.math.allclose(compiled_derivative(jax.numpy.array(3.0)), 27.0)
    assert qp.math.allclose(compiled_derivative(jax.numpy.array(4.0)), 48.0)
    assert derivative.forward is forward
    assert not isinstance(derivative.forward, jax.core.Tracer)
    if eager_first:
        assert qp.math.allclose(derivative.forward + 1.0, 9.0)
    else:
        assert derivative.forward is None


@pytest.mark.jax
def test_jax_pytree_jacobian():
    """JAX Jacobians support structured inputs and outputs."""
    jax = pytest.importorskip("jax")
    argument = {"value": jax.numpy.array(2.0)}
    result = qp.jacobian(lambda data: {"square": data["value"] ** 2})(argument)
    assert qp.math.allclose(result["square"]["value"], 4.0)


@pytest.mark.torch
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_torch_inferred_trainability(diff_function):
    """Torch re-infers trainable arguments on each call and unwraps a single result."""
    torch = pytest.importorskip("torch")
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(3.0)
    derivative = diff_function(lambda first, second: first * second)
    result = derivative(first, second)
    assert not isinstance(result, tuple)
    assert torch.allclose(result, torch.tensor(3.0))
    second.requires_grad_(True)
    result = derivative(first, second)
    assert isinstance(result, tuple)
    assert torch.allclose(result[0], torch.tensor(3.0))
    assert torch.allclose(result[1], torch.tensor(2.0))


@pytest.mark.torch
def test_torch_grad_does_not_accumulate():
    """Repeated gradient calls leave existing Torch gradient buffers intact."""
    torch = pytest.importorskip("torch")
    argument = torch.tensor(2.0, requires_grad=True)
    argument.grad = torch.tensor(7.0)
    derivative = qp.grad(lambda value: value**2)
    assert torch.allclose(derivative(argument), torch.tensor(4.0))
    assert torch.allclose(derivative(argument), torch.tensor(4.0))
    assert torch.allclose(argument.grad, torch.tensor(7.0))


@pytest.mark.torch
@pytest.mark.parametrize("relation", ["aliased", "connected"])
@pytest.mark.parametrize("argnums", [0, (0, 1)])
def test_torch_grad_positional_partials(relation, argnums):
    """Aliased and connected caller tensors still yield separate positional partials."""
    torch = pytest.importorskip("torch")
    first = torch.tensor(2.0, requires_grad=True)
    second = first if relation == "aliased" else first**2

    def cost(first, second):
        return first * second

    expected = torch.autograd.functional.jacobian(cost, (first, second), create_graph=True)
    derivative = qp.grad(cost, argnums=argnums)
    result = derivative(first, second)
    assert torch.allclose(derivative.forward, first * second)
    if isinstance(argnums, int):
        assert torch.allclose(result, expected[argnums])
        expected_second = torch.autograd.grad(expected[argnums], first)[0]

        def partial(value):
            return derivative(value, value if relation == "aliased" else value**2)

        assert torch.allclose(qp.grad(partial)(first), expected_second)
    else:
        for actual, reference in zip(result, expected, strict=True):
            assert torch.allclose(actual, reference)


@pytest.mark.torch
def test_torch_unused_and_constant_gradients():
    """Disconnected inputs and constant functions have zero derivatives."""
    torch = pytest.importorskip("torch")
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(3.0, requires_grad=True)
    result = qp.grad(lambda first, second: first**2)(first, second)
    assert torch.allclose(result[0], torch.tensor(4.0))
    assert torch.allclose(result[1], torch.tensor(0.0))
    constant_derivative = qp.grad(lambda value: torch.tensor(3.0))
    assert torch.allclose(constant_derivative(first), torch.tensor(0.0))
    assert torch.allclose(qp.grad(constant_derivative)(first), torch.tensor(0.0))


@pytest.mark.torch
@pytest.mark.parametrize("argnums", [0, [0], (0, 1)])
def test_torch_tuple_jacobian(argnums):
    """Torch preserves output and input tuple axes for vector Jacobians."""
    torch = pytest.importorskip("torch")
    first = torch.tensor([2.0, 3.0], requires_grad=True)
    second = torch.tensor(4.0, requires_grad=True)
    result = qp.jacobian(lambda first, second: (first**2, first * second), argnums=argnums)(
        first, second
    )
    expected_first = torch.diag(2 * first)
    expected_second = 4 * torch.eye(2)
    if isinstance(argnums, int):
        assert torch.allclose(result[0], expected_first)
        assert torch.allclose(result[1], expected_second)
    else:
        assert torch.allclose(result[0][0], expected_first)
        assert torch.allclose(result[1][0], expected_second)
        if len(argnums) > 1:
            assert torch.allclose(result[0][1], torch.zeros(2))
            assert torch.allclose(result[1][1], first)


@pytest.mark.jax
@pytest.mark.torch
@pytest.mark.parametrize("diff_function", [qp.grad, qp.jacobian])
def test_mixed_native_frameworks_rejected(diff_function):
    """A derivative call cannot mix JAX and Torch inputs."""
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Tensors contain mixed types"):
        diff_function(lambda first, second: first * second)(
            jax.numpy.array(2.0), torch.tensor(3.0, requires_grad=True)
        )


@pytest.mark.torch
@pytest.mark.parametrize("output_kind", ["python", "tuple", "vector", "complex"])
def test_torch_grad_invalid_output(output_kind):
    """Torch gradients require a real scalar Tensor output."""
    torch = pytest.importorskip("torch")
    argument = torch.tensor(2.0, requires_grad=True)
    outputs = {
        "python": 3.0,
        "tuple": (argument, argument),
        "vector": argument.reshape(1),
        "complex": argument.to(torch.complex64),
    }
    with pytest.raises(TypeError, match="Grad only applies to real scalar-output functions"):
        qp.grad(lambda value: outputs[output_kind])(argument)


@pytest.mark.torch
def test_torch_grad_requires_trainable_selected_input():
    """Explicitly selecting a non-trainable input cannot silently return a zero gradient."""
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="Selected Torch arguments must be tensors"):
        qp.grad(lambda value: value**2, argnums=0)(torch.tensor(2.0))
