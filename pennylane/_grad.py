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
This module contains the autograd wrappers :class:`grad` and :func:`jacobian`
"""
import numpy as onp

from pennylane import numpy as np
from functools import partial

from autograd.core import make_vjp as _make_vjp
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace
from autograd import jacobian as _jacobian

make_vjp = unary_to_nary(_make_vjp)


class grad:
    """Returns the gradient as a callable function of (functions of) QNodes.

    Function arguments with the property ``requires_grad`` set to ``False``
    will automatically be excluded from the gradient computation, unless
    the ``argnum`` keyword argument is passed.

    When the output gradient function is executed, both the forward pass
    *and* the backward pass will be performed in order to
    compute the gradient. The value of the forward pass is available via the
    :attr:`~.forward` property.

    Args:
        func (function): a plain QNode, or a Python function that contains
            a combination of quantum and classical nodes
        argnum (int, list(int), None): Which argument(s) to take the gradient
            with respect to. By default, the arguments themselves are used
            to determine differentiability, by examining the ``requires_grad``
            property.

    Returns:
        function: The function that returns the gradient of the input
        function with respect to the differentiable arguments, or, if specified,
        the arguments in ``argnum``.
    """

    def __init__(self, fun, argnum=None):
        self._forward = None
        self._grad_fn = None

        self._fun = fun
        self._argnum = argnum

        if self._argnum is not None:
            # If the differentiable argnum is provided, we can construct
            # the gradient function at once during initialization
            self._grad_fn = self._grad_with_forward(fun, argnum=argnum)

    def _get_grad_fn(self, args):
        """Get the required gradient function.

        * If the differentiable argnum was provided on initialization,
          this has been pre-computed and is available via self._grad_fn

        * Otherwise, we must dynamically construct the gradient function by
          inspecting as to which of the parameter arguments are marked
          as differentiable.
        """
        if self._grad_fn is not None:
            return self._grad_fn

        # Inspect the arguments for differentiability, and
        # compute the autograd gradient function with required argnums
        # dynamically.
        argnum = []

        for idx, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                argnum.append(idx)

        if len(argnum) == 1:
            argnum = argnum[0]

        return self._grad_with_forward(
            self._fun,
            argnum=argnum,
        )

    def __call__(self, *args, **kwargs):
        """Evaluates the gradient function, and saves the function value
        calculated during the forward pass in :attr:`.forward`."""
        grad_value, ans = self._get_grad_fn(args)(*args, **kwargs)
        self._forward = ans
        return grad_value

    @property
    def forward(self):
        """float: The result of the forward pass calculated while performing
        backpropagation. Will return ``None`` if the backpropagation has not yet
        been performed."""
        return self._forward

    @staticmethod
    @unary_to_nary
    def _grad_with_forward(fun, x):
        """This function is a replica of ``autograd.grad``, with the only
        difference being that it returns both the gradient *and* the forward pass
        value."""
        vjp, ans = _make_vjp(fun, x)

        if not vspace(ans).size == 1:
            raise TypeError(
                "Grad only applies to real scalar-output functions. "
                "Try jacobian, elementwise_grad or holomorphic_grad."
            )

        grad_value = vjp(vspace(ans).ones())
        return grad_value, ans


def jacobian(func, argnum=None):
    """Returns the Jacobian as a callable function of vector-valued
    (functions of) QNodes.

    This is a wrapper around the :mod:`autograd.jacobian` function.

    Args:
        func (function): A vector-valued Python function or QNode that contains
            a combination of quantum and classical nodes. The output of the computation
            must consist of a single NumPy array (if classical) or a tuple of
            expectation values (if a quantum node)
        argnum (int or Sequence[int]): Which argument to take the gradient
            with respect to. If a sequence is given, the Jacobian matrix
            corresponding to all input elements and all output elements is returned.

    Returns:
        function: the function that returns the Jacobian of the input
        function with respect to the arguments in argnum
    """
    # pylint: disable=no-value-for-parameter

    if argnum is not None:
        # for backwards compatibility with existing code
        # that manually specifies argnum
        if isinstance(argnum, int):
            return _jacobian(func, argnum)

        return lambda *args, **kwargs: np.stack(
            [_jacobian(func, arg)(*args, **kwargs) for arg in argnum]
        ).T

    def _jacobian_function(*args, **kwargs):
        """Inspect the arguments for differentiability, and
        compute the autograd gradient function with required argnums
        dynamically.

        This wrapper function is returned to the user instead of autograd.jacobian,
        so that we can take into account cases where the user computes the
        jacobian function once, but then calls it with arguments that change
        in differentiability.
        """
        argnum = []

        for idx, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                argnum.append(idx)

        if not argnum:
            return tuple()

        if len(argnum) == 1:
            return _jacobian(func, argnum[0])(*args, **kwargs)

        jacobians = [_jacobian(func, arg)(*args, **kwargs) for arg in argnum]

        try:
            return np.stack(jacobians).T
        except ValueError:
            # The Jacobian of each argument is a different shape and cannot
            # be stacked; simply return the tuple of argument Jacobians.
            return tuple(jacobians)

    return _jacobian_function


def _fd_first_order_centered(f, argnum, delta, *args, idx=None, **kwargs):

    r"""Uses a central finite difference approximation to compute the gradient
    of the function ``f`` with respect to the argument ``argnum``.

    Args:
        f (function): function with signature ``f(*args, **kwargs)``
        argnum (int): the argument with respect to which the gradient is taken
        delta (float): step size used to evaluate the finite difference
        idx (list[int]): If argument ``args[argnum]`` is an array, ``idx`` can
            be used to specify the indices of the arguments to differentiate.
            For example, for function ``f(x, y, z)``, ``argnum=1``, ``idx=[3, 2]``
            the function will differentiate ``f`` with respect to elements
            ``3`` and ``2`` of argument ``y``.

    Returns:
        (float or array): the gradient of the input function with respect
        to the arguments in ``argnum``
    """

    if argnum > len(args) - 1:
        raise ValueError(
            "The value of 'argnum' has to be between 0 and {}; got {}".format(len(args) - 1, argnum)
        )

    x = onp.array(args[argnum])
    gradient = onp.zeros_like(x, dtype="O")

    if x.ndim == 0 and idx is not None:
        raise ValueError(
            "Argument {} is not an array, 'idx' should be set to 'None';"
            " got {}".format(argnum, idx)
        )

    if idx is None:
        idx = list(onp.ndindex(*x.shape))

    for i in idx:
        shift = onp.zeros_like(x)
        shift[i] += 0.5 * delta
        gradient[i] = (
            f(*args[:argnum], x + shift, *args[argnum + 1 :], **kwargs)
            - f(*args[:argnum], x - shift, *args[argnum + 1 :], **kwargs)
        ) * delta ** -1

    return gradient


def _fd_second_order_centered(f, argnum, delta, *args, idx=None, **kwargs):
    r"""Uses a central finite difference approximation to compute the second-order
    derivative :math:`\frac{\partial^2 f(x)}{\partial x_i \partial x_j}` of the function ``f``
    with respect to the argument ``argnum``.

    Args:
        f (function): function with signature ``f(*args, **kwargs)``
        argnum (int): the argument with respect to which the gradient is taken
        delta (float): step size used to evaluate the finite difference
        idx (list[int]): If argument ``args[argnum]`` is an array, `idx`` specifies
            the indices ``i, j`` of the arguments to differentiate.
            For example, for function ``f(x, y, z)``, ``argnum=1``, ``idx=[3, 2]``,
            the function will calculate the second-order derivative of ``f`` with
            respect to elements ``3`` and ``2`` of argument ``y``.

    Returns:
        (float or array): the second-order derivative of the input function with respect
        to the arguments in ``argnum``
    """

    if argnum > len(args) - 1:
        raise ValueError(
            "The value of 'argnum' has to be between 0 and {}; got {}".format(len(args) - 1, argnum)
        )

    x = onp.array(args[argnum])

    if x.ndim == 0 and idx is not None:
        raise ValueError(
            "Argument {} is not an array, 'idx' should be set to 'None';"
            " got {}".format(argnum, idx)
        )

    if idx is None:
        if x.ndim != 0:
            raise ValueError(
                "Argument {} is an array, 'idx' should contain the indices of the arguments"
                " to differentiate; got idx = {}".format(argnum, idx)
            )
        idx = [(), ()]
    else:
        if len(idx) > 2:
            raise ValueError(
                "The number of indices given in 'idx' can not be greater than two; got {} indices".format(
                    len(idx)
                )
            )

    i, j = idx

    # diagonal
    if i == j:
        shift = onp.zeros_like(x)
        shift[i] += delta
        deriv2 = (
            f(*args[:argnum], x + shift, *args[argnum + 1 :], **kwargs)
            - 2 * f(*args[:argnum], x, *args[argnum + 1 :], **kwargs)
            + f(*args[:argnum], x - shift, *args[argnum + 1 :], **kwargs)
        ) * delta ** -2

    # off-diagonal
    if i != j:
        shift_i = onp.zeros_like(x)
        shift_i[i] += 0.5 * delta

        shift_j = onp.zeros_like(x)
        shift_j[j] += 0.5 * delta

        deriv2 = (
            f(*args[:argnum], x + shift_i + shift_j, *args[argnum + 1 :], **kwargs)
            - f(*args[:argnum], x - shift_i + shift_j, *args[argnum + 1 :], **kwargs)
            - f(*args[:argnum], x + shift_i - shift_j, *args[argnum + 1 :], **kwargs)
            + f(*args[:argnum], x - shift_i - shift_j, *args[argnum + 1 :], **kwargs)
        ) * delta ** -2

    return deriv2


def finite_diff(f, N=1, argnum=0, idx=None, delta=0.01):
    r"""Returns a function that can be evaluated to compute the gradient or the
    second-order derivative of the callable function ``f`` using a centered finite
    difference approximation.

    The first-order derivatives :math:`\frac{\partial f(x)}{\partial x_i}` entering
    the gradient of the input function are given by,

    .. math::

        \frac{\partial f(x)}{\partial x_i} \approx \frac{f(x_i + \delta/2)
        - f(x_i - \delta/2)}{\delta}

    On the other hand, the second-order derivative
    :math:`\frac{\partial^2 f(x)}{\partial x_i \partial x_j}` are evaluated using the
    following expressions:

    For :math:`i = j`:

    .. math::
        \frac{\partial^2 f(x)}{\partial x_i^2} \approx
        \frac{f(x_i + \delta) - 2 f(x) + f(x_i - \delta)}{\delta^2},

    and for :math:`i \neq j`:

    .. math::
        \frac{\partial^2 f(x)}{\partial x_i \partial x_j} \approx
        \frac{f(x_i + \delta/2, x_j + \delta/2) - f(x_i - \delta/2, x_j + \delta/2)
        - f(x_i + \delta/2, x_j - \delta/2) + f(x_i - \delta/2, x_j - \delta/2)}
        {\delta^2}.

    Args:
        f (function): function with signature ``f(*args, **kwargs)``
        N (int): specifies the order of the finite difference approximation
        argnum (int): the argument of function ``f`` to differentiate
        idx (list[int]): If argument ``args[argnum]`` is an array, ``idx`` can be used
            to specify the indices of the argument ``argnum`` to differentiate.
            For ``N=1`` it can be given to specify the gradient components to be computed.
            For example, for the function ``f(x, y, z)``, ``argnum=1``, ``idx=[3, 2]``
            the returned function will differentiate ``f`` with respect to elements
            ``3`` and ``2`` of argument ``y``. For ``N=2``, it specifies the indices
            ``i, j`` of the variables involved in the second-order derivative
            :math:`\frac{\partial^2 f(x, y, z)}{\partial y_i \partial y_j}`.
        delta (float): step size used to evaluate the finite differences

    Returns:
        function: the function to compute the gradient (``N=1``) or the
        second-order derivative (``N=2``) of the input function ``f`` with respect
        to the arguments in ``argnum``

    **Examples**

    >>> def f(x, y):
    ...     return np.sin(y[0])*np.sin(y[1]) - x**-3

    >>> (x, y) = (0.376, np.array([1.975, 0.33, -0.4]))

    >>> # We compute the gradient with respect to 'y' as
    >>> gradient = qml.finite_diff(f, argnum=1)
    >>> print(gradient(x, y))
    [-0.12744129189670161 0.8698027233702277]

    >>> # We can also compute the derivative with respect to 'y[1]'
    >>> derivative = qml.finite_diff(f, argnum=1, idx=[1])
    >>> print(derivative(x, y)[1])
    0.8698027233702277

    >>> # and the second derivative with respect to 'y[0], y[1]'
    >>> second_derivative = qml.finite_diff(f, N=2, argnum=1, idx=[0, 1])
    >>> print(second_derivative(x, y))
    -0.372062798810191
    """

    if not callable(f):
        error_message = "{} object is not callable. \n" "'f' should be a callable function".format(
            type(f)
        )
        raise TypeError(error_message)

    if delta <= 0.0:
        raise ValueError(
            "The value of the step size 'delta' has to be greater than 0; got {}".format(delta)
        )

    if N == 1:
        return partial(_fd_first_order_centered, f, argnum, delta, idx=idx)

    if N == 2:
        return partial(_fd_second_order_centered, f, argnum, delta, idx=idx)

    raise ValueError(
        "At present, finite-difference approximations are supported up to second-order."
        " The value of 'N' can be 1 or 2; got {}".format(N)
    )
