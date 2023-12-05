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
"""
This file contains the is_independent function that checks if
a function is independent of its arguments for the interfaces

* Autograd
* JAX
* TensorFlow
* PyTorch
"""
import warnings

import numpy as np
from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode

from pennylane import numpy as pnp


def _autograd_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using Autograd.

    Args:
        func (callable): Function to test for independence
        args (tuple): Arguments for the function with respect to which
            to test for independence
        kwargs (dict): Keyword arguments for the function at which
            (but not with respect to which) to test for independence

    Returns:
        bool: Whether the function seems to not depend on it ``args``
        analytically. That is, an output of ``True`` means that the
        ``args`` do *not* feed into the output.

    In Autograd, we test this by sending a ``Box`` through the function and
    testing whether the output is again a ``Box`` and on the same trace as
    the input ``Box``. This means that we can trace actual *independence*
    of the output from the input, not only whether the passed function is
    constant.
    The code is adapted from
    `autograd.tracer.py::trace
    <https://github.com/HIPS/autograd/blob/master/autograd/tracer.py#L7>`__.
    """
    # pylint: disable=protected-access
    node = VJPNode.new_root()
    with trace_stack.new_trace() as t:
        start_box = new_box(args, t, node)
        end_box = func(*start_box, **kwargs)

    if type(end_box) in [tuple, list]:
        if any(isbox(_end) and _end._trace == start_box._trace for _end in end_box):
            return False
    elif isinstance(end_box, np.ndarray):
        if end_box.ndim == 0:
            end_box = [end_box.item()]
        if any(isbox(_end) and _end._trace == start_box._trace for _end in end_box):
            return False
    else:
        if isbox(end_box) and end_box._trace == start_box._trace:
            return False
    return True


# pylint: disable=import-outside-toplevel,unnecessary-lambda-assignment,unnecessary-lambda
def _jax_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using JAX.

    Args:
        func (callable): Function to test for independence
        args (tuple): Arguments for the function with respect to which
            to test for independence
        kwargs (dict): Keyword arguments for the function at which
            (but not with respect to which) to test for independence

    Returns:
        bool: Whether the function seems to not depend on it ``args``
        analytically. That is, an output of ``True`` means that the
        ``args`` do *not* feed into the output.

    In JAX, we test this by constructing the VJP of the passed function
    and inspecting its signature.
    The first argument of the output of ``jax.vjp`` is a ``Partial``.
    If *any* processing happens to any input, the arguments of that
    ``Partial`` are unequal to ``((),)`.
    Functions that depend on the input in a trivial manner, i.e., without
    processing it, will go undetected by this. Therefore we also
    test the arguments of the *function* of the above ``Partial``.
    The first of these arguments is a list of tuples and if the
    first entry of the first tuple is not ``None``, the input arguments
    are detected to actually feed into the output.

    .. warning::

        This is an experimental function and unknown edge
        cases may exist to this two-stage test.
    """
    import jax

    mapped_func = lambda *_args: func(*_args, **kwargs)
    _vjp = jax.vjp(mapped_func, *args)[1]
    if _vjp.args[0].args != ((),):
        return False
    if _vjp.args[0].func.args[0][0][0] is not None:
        return False

    return True


def _tf_is_indep_analytic(func, *args, **kwargs):
    """Test analytically whether a function is independent of its arguments
    using TensorFlow.

    Args:
        func (callable): Function to test for independence
        args (tuple): Arguments for the function with respect to which
            to test for independence
        kwargs (dict): Keyword arguments for the function at which
            (but not with respect to which) to test for independence

    Returns:
        bool: Whether the function seems to not depend on it ``args``
        analytically. That is, an output of ``True`` means that the
        ``args`` do *not* feed into the output.

    In TensorFlow, we test this by computing the Jacobian of the output(s)
    with respect to the arguments. If the Jacobian is ``None``, the output(s)
    is/are independent.

    .. note::

        Of all interfaces, this is currently the most robust for the
        ``is_independent`` functionality.
    """
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    with tf.GradientTape(persistent=True) as tape:
        out = func(*args, **kwargs)

    if isinstance(out, tuple):
        jac = [tape.jacobian(_out, args) for _out in out]
        return all(all(__jac is None for __jac in _jac) for _jac in jac)

    jac = tape.jacobian(out, args)
    return all(_jac is None for _jac in jac)


def _get_random_args(args, interface, num, seed, bounds):
    r"""Generate random arguments of a given structure.

    Args:
        args (tuple): Original input arguments
        interface (str): Interface of the QNode into which the arguments will be fed
        num (int): Number of random argument sets to generate
        seed (int): Seed for random generation
        bounds (tuple[int]): Range within which to sample the random parameters.

    Returns:
        list[tuple]: List of length ``num`` with each entry being a random instance
        of arguments like ``args``.

    This function generates ``num`` many tuples of random arguments in the given range
    that have the same shapes as ``args``.
    """
    width = bounds[1] - bounds[0]
    if interface == "tf":
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tf.random.set_seed(seed)
        rnd_args = []
        for _ in range(num):
            _args = (tf.random.uniform(tf.shape(_arg)) * width + bounds[0] for _arg in args)
            _args = tuple(
                tf.Variable(_arg) if isinstance(arg, tf.Variable) else _arg
                for _arg, arg in zip(_args, args)
            )
            rnd_args.append(_args)
    elif interface == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        torch.random.manual_seed(seed)
        rnd_args = [
            tuple(torch.rand(np.shape(arg)) * width + bounds[0] for arg in args) for _ in range(num)
        ]
    else:
        rng = np.random.default_rng(seed)
        rnd_args = [
            tuple(rng.random(np.shape(arg)) * width + bounds[0] for arg in args) for _ in range(num)
        ]
        if interface == "autograd":
            # Mark the arguments as trainable with Autograd
            rnd_args = [tuple(pnp.array(a, requires_grad=True) for a in arg) for arg in rnd_args]

    return rnd_args


def _is_indep_numerical(func, interface, args, kwargs, num_pos, seed, atol, rtol, bounds):
    """Test whether a function returns the same output at random positions.

    Args:
        func (callable): Function to be tested
        interface (str): Interface used by ``func``
        args (tuple): Positional arguments with respect to which to test
        kwargs (dict): Keyword arguments for ``func`` at which to test;
            the ``kwargs`` are kept fixed in this test.
        num_pos (int): Number of random positions to test
        seed (int): Seed for random number generator
        atol (float): Absolute tolerance for comparing the outputs
        rtol (float): Relative tolerance for comparing the outputs
        bounds (tuple[int, int]): Limits of the range from which to sample

    Returns:
        bool: Whether ``func`` returns the same output at the randomly
        chosen points.
    """

    # pylint:disable=too-many-arguments

    rnd_args = _get_random_args(args, interface, num_pos, seed, bounds)
    original_output = func(*args, **kwargs)
    is_tuple_valued = isinstance(original_output, tuple)
    for _rnd_args in rnd_args:
        new_output = func(*_rnd_args, **kwargs)
        if is_tuple_valued:
            if not all(
                np.allclose(new, orig, atol=atol, rtol=rtol)
                for new, orig in zip(new_output, original_output)
            ):
                return False
        else:
            if not np.allclose(new_output, original_output, atol=atol, rtol=rtol):
                return False

    return True


def is_independent(
    func,
    interface,
    args,
    kwargs=None,
    num_pos=5,
    seed=9123,
    atol=1e-6,
    rtol=0,
    bounds=(-np.pi, np.pi),
):
    """Test whether a function is independent of its input arguments,
    both numerically and analytically.

    Args:
        func (callable): Function to be tested
        interface (str): Autodiff framework used by ``func``. Must correspond to one
            of the supported PennyLane interface strings, such as ``"autograd"``,
            ``"tf"``, ``"torch"``, ``"jax"``.
        args (tuple): Positional arguments with respect to which to test
        kwargs (dict): Keyword arguments for ``func`` at which to test;
            the keyword arguments are kept fixed in this test.
        num_pos (int): Number of random positions to test
        seed (int): Seed for the random number generator
        atol (float): Absolute tolerance for comparing the outputs
        rtol (float): Relative tolerance for comparing the outputs
        bounds (tuple[float]): 2-tuple containing limits of the range from which to sample

    Returns:
        bool: Whether ``func`` returns the same output at randomly
        chosen points and is numerically independent of its arguments.

    .. warning::

        This function is experimental.
        As such, it might yield wrong results and might behave
        slightly differently in distinct autodifferentiation frameworks
        for some edge cases.
        For example, a currently known edge case are piecewise
        functions that use classical control and simultaneously
        return (almost) constant output, such as

        .. code-block:: python

            def func(x):
                if abs(x) <1e-5:
                    return x
                else:
                    return 0. * x

    The analytic and numeric tests used are as follows.

    - The analytic test performed depends on the provided ``interface``,
      both in its method and its degree of reliability.

    - For the numeric test, the function is evaluated at a series of random positions,
      and the outputs numerically compared to verify that the output
      is constant.

    .. warning ::

        Currently, no analytic test is available for the PyTorch interface.
        When using PyTorch, a warning will be raised and only the
        numeric test is performed.

    .. note ::

        Due to the structure of ``is_independent``, it is possible that it
        errs on the side of reporting a dependent function to be independent
        (a false positive). However, reporting an independent function to be
        dependent (a false negative) is *highly* unlikely.

    **Example**

    Consider the (linear) function

    .. code-block:: python

        def lin(x, weights=None):
            return np.dot(x, weights)

    This function clearly depends on ``x``. We may check for this via

    .. code-block:: pycon

        >>> x = np.array([0.2, 9.1, -3.2], requires_grad=True)
        >>> weights = np.array([1.1, -0.7, 1.8], requires_grad=True)
        >>> qml.math.is_independent(lin, "autograd", (x,), {"weights": weights})
        False

    However, the Jacobian will not depend on ``x`` because ``lin`` is a
    linear function:

    .. code-block:: pycon

        >>> jac = qml.jacobian(lin)
        >>> qml.math.is_independent(jac, "autograd", (x,), {"weights": weights})
        True

    Note that a function ``f = lambda x: 0.0 * x`` will be counted as *dependent* on ``x``
    because it does depend on ``x`` *functionally*, even if the value is constant for all ``x``.
    This means that ``is_independent`` is a stronger test than simply verifying functions
    have constant output.
    """

    # pylint:disable=too-many-arguments

    if not interface in {"autograd", "jax", "tf", "torch", "tensorflow"}:
        raise ValueError(f"Unknown interface: {interface}")

    kwargs = kwargs or {}

    if interface == "autograd":
        if not _autograd_is_indep_analytic(func, *args, **kwargs):
            return False

    if interface == "jax":
        if not _jax_is_indep_analytic(func, *args, **kwargs):
            return False

    if interface in ("tf", "tensorflow"):
        if not _tf_is_indep_analytic(func, *args, **kwargs):
            return False

    if interface == "torch":
        warnings.warn(
            "The function is_independent is only available numerically for the PyTorch interface. "
            "Make sure that sampling positions and evaluating the function at these positions "
            "is a sufficient test, or change the interface."
        )

    return _is_indep_numerical(func, interface, args, kwargs, num_pos, seed, atol, rtol, bounds)
