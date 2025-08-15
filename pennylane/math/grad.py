# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This submodule defines grad and jacobian for differentiating circuits in an interface-independent way.
"""

from collections.abc import Callable, Sequence

from pennylane._grad import grad as _autograd_grad
from pennylane._grad import jacobian as _autograd_jacobian

from .interface_utils import get_interface


# pylint: disable=import-outside-toplevel
def grad(f: Callable, argnums: Sequence[int] | int = 0) -> Callable:
    """Compute the gradient in a jax-like manner for any interface.

    Args:
        f (Callable): a function with a single 0-D scalar output
        argnums (Sequence[int] | int ) = 0 : which arguments to differentiate

    Returns:
        Callable: a function with the same signature as ``f`` that returns the gradient.

    .. seealso:: :func:`pennylane.math.jacobian`

    Note that this function follows the same design as jax. By default, the function will return the gradient
    of the first argument, whether or not other arguments are trainable.

    >>> import jax, torch, tensorflow as tf
    >>> def f(x, y):
    ...     return  x * y
    >>> qml.math.grad(f)(qml.numpy.array(2.0), qml.numpy.array(3.0))
    tensor(3., requires_grad=True)
    >>> qml.math.grad(f)(jax.numpy.array(2.0), jax.numpy.array(3.0))
    Array(3., dtype=float32, weak_type=True)
    >>> qml.math.grad(f)(torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True))
    tensor(3.)
    >>> qml.math.grad(f)(tf.Variable(2.0), tf.Variable(3.0))
    <tf.Tensor: shape=(), dtype=float32, numpy=3.0>

    ``argnums`` can be provided to differentiate multiple arguments.

    >>> qml.math.grad(f, argnums=(0,1))(torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True))
    (tensor(3.), tensor(2.))

    Note that the selected arguments *must* be of an appropriately trainable datatype, or an error may occur.

    >>> qml.math.grad(f)(torch.tensor(1.0), torch.tensor(2.))
    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

    """

    argnums_integer = False
    if isinstance(argnums, int):
        argnums = (argnums,)
        argnums_integer = True

    def compute_grad(*args, **kwargs):
        interface = get_interface(*args)

        if interface == "autograd":
            g = _autograd_grad(f, argnum=argnums)(*args, **kwargs)
            return g[0] if argnums_integer else g

        if interface == "jax":
            import jax

            g = jax.grad(f, argnums=argnums)(*args, **kwargs)
            return g[0] if argnums_integer else g

        if interface == "torch":
            y = f(*args, **kwargs)
            y.backward()
            g = tuple(args[i].grad for i in argnums)
            return g[0] if argnums_integer else g

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            import tensorflow as tf

            with tf.GradientTape() as tape:
                y = f(*args, **kwargs)

            g = tape.gradient(y, tuple(args[i] for i in argnums))
            return g[0] if argnums_integer else g

        raise ValueError(f"Interface {interface} is not differentiable.")

    return compute_grad


# pylint: disable=import-outside-toplevel
def _torch_jac(f, argnums, args, kwargs):
    """Calculate a jacobian via torch."""
    from torch.autograd.functional import jacobian as _torch_jac

    argnums_torch = (argnums,) if isinstance(argnums, int) else argnums
    trainable_args = tuple(args[i] for i in argnums_torch)

    # keep track of output type to know how to unpack
    output_type_cache = []

    def partial_f(*_trainables):
        full_args = list(args)
        for argnum, value in zip(argnums_torch, _trainables, strict=True):
            full_args[argnum] = value
        result = f(*full_args, **kwargs)
        output_type_cache.append(type(result))
        return result

    jac = _torch_jac(partial_f, trainable_args)
    if output_type_cache[-1] is tuple:
        return tuple(j[0] for j in jac) if isinstance(argnums, int) else jac
    # else array
    return jac[0] if isinstance(argnums, int) else jac


# pylint: disable=import-outside-toplevel
def _tensorflow_jac(
    f, argnums, args, kwargs
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """Calculate a jacobian via tensorflow"""
    import tensorflow as tf

    with tf.GradientTape() as tape:
        y = f(*args, **kwargs)

    if get_interface(y) != "tensorflow":
        raise ValueError(
            f"qml.math.jacobian does not work with tensorflow and non-tensor outputs. Got {y} of type {type(y)}."
        )

    argnums_integer = False
    if isinstance(argnums, int):
        argnums_tf = (argnums,)
        argnums_integer = True
    else:
        argnums_tf = argnums

    g = tape.jacobian(y, tuple(args[i] for i in argnums_tf))
    return g[0] if argnums_integer else g


# pylint: disable=import-outside-toplevel
def jacobian(f: Callable, argnums: Sequence[int] | int = 0) -> Callable:
    """Compute the Jacobian in a jax-like manner for any interface.

    Args:
        f (Callable): a function with a vector valued output
        argnums (Sequence[int] | int ) = 0 : which arguments to differentiate

    Returns:
        Callable: a function with the same signature as ``f`` that returns the jacobian

    .. seealso:: :func:`pennylane.math.grad`

    Note that this function follows the same design as jax. By default, the function will return the gradient
    of the first argument, whether or not other arguments are trainable.

    >>> import jax, torch, tensorflow as tf
    >>> def f(x, y):
    ...     return  x * y
    >>> qml.math.jacobian(f)(qml.numpy.array([2.0, 3.0]), qml.numpy.array(3.0))
    array([[3., 0.],
              [0., 3.]])
    >>> qml.math.jacobian(f)(jax.numpy.array([2.0, 3.0]), jax.numpy.array(3.0))
    Array([[3., 0.],
               [0., 3.]], dtype=float32)
    >>> x_torch = torch.tensor([2.0, 3.0], requires_grad=True)
    >>> y_torch = torch.tensor(3.0, requires_grad=True)
    >>> qml.math.jacobian(f)(x_torch, y_torch)
    tensor([[3., 0.],
                [0., 3.]])
    >>> qml.math.jacobian(f)(tf.Variable([2.0, 3.0]), tf.Variable(3.0))
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[3., 0.],
              [0., 3.]], dtype=float32)>

    ``argnums`` can be provided to differentiate multiple arguments.

    >>> qml.math.jacobian(f, argnums=(0,1))(x_torch, y_torch)
    (tensor([[3., 0.],
            [0., 3.]]),
    tensor([2., 3.]))

    While jax can handle taking jacobians of outputs with any pytree shape:

    >>> def pytree_f(x):
    ...     return {"a": 2*x, "b": 3*x}
    >>> qml.math.jacobian(pytree_f)(jax.numpy.array(2.0))
    {'a': Array(2., dtype=float32, weak_type=True),
    'b': Array(3., dtype=float32, weak_type=True)}

    Torch can only differentiate arrays and tuples:

    >>> def tuple_f(x):
    ...     return x**2, x**3
    >>> qml.math.jacobian(tuple_f)(torch.tensor(2.0))
    (tensor(4.), tensor(12.))
    >>> qml.math.jacobian(pytree_f)(torch.tensor(2.0))
    TypeError: The outputs of the user-provided function given to jacobian must be
    either a Tensor or a tuple of Tensors but the given outputs of the user-provided
    function has type <class 'dict'>.


    But tensorflow and autograd can only handle array-valued outputs:

    >>> qml.math.jacobian(tuple_f)(qml.numpy.array(2.0))
    ValueError: autograd can only differentiate with respect to arrays, not <class 'tuple'>
    >>> qml.math.jacobian(tuple_f)(tf.Variable(2.0))
    ValueError: qml.math.jacobian does not work with tensorflow and non-tensor outputs.
    Got (<tf.Tensor: shape=(), dtype=float32, numpy=4.0>,
    <tf.Tensor: shape=(), dtype=float32, numpy=8.0>) of type <class 'tuple'>.

    """

    def compute_jacobian(*args, **kwargs):
        interface = get_interface(*args)

        if interface == "autograd":
            return _autograd_jacobian(f, argnum=argnums)(*args, **kwargs)

        if interface == "jax":
            import jax

            return jax.jacobian(f, argnums=argnums)(*args, **kwargs)

        if interface == "torch":
            return _torch_jac(f, argnums, args, kwargs)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            return _tensorflow_jac(f, argnums, args, kwargs)

        raise ValueError(f"Interface {interface} is not differentiable.")

    return compute_jacobian
