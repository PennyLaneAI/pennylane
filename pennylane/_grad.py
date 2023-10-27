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
import warnings


from autograd import jacobian as _jacobian
from autograd.core import make_vjp as _make_vjp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary

from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError

make_vjp = unary_to_nary(_make_vjp)


class grad:
    """Returns the gradient as a callable function of (functions of) QNodes.

    By default, gradients are computed for arguments which contain the property
    ``requires_grad=True``. Alternatively, the ``argnum`` keyword argument can
    be specified to compute gradients for function arguments without this property,
    such as scalars, lists, tuples, dicts, or vanilla NumPy arrays. Setting
    ``argnum`` to the index of an argument with ``requires_grad=False`` will raise
    a ``NonDifferentiableError``.

    When the output gradient function is executed, both the forward pass
    *and* the backward pass will be performed in order to
    compute the gradient. The value of the forward pass is available via the
    :attr:`~.forward` property.

    .. warning::
        ``grad`` is intended to be used with the Autograd interface only.

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
            # the gradient function at once during initialization.
            # Known pylint issue with function signatures and decorators:
            # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
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
            return self._grad_fn, self._argnum

        # Inspect the arguments for differentiability, and
        # compute the autograd gradient function with required argnums
        # dynamically.
        argnum = []

        for idx, arg in enumerate(args):
            trainable = getattr(arg, "requires_grad", None) or isinstance(arg, ArrayBox)
            if trainable:
                argnum.append(idx)

        if len(argnum) == 1:
            argnum = argnum[0]

        # Known pylint issue with function signatures and decorators:
        # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
        return self._grad_with_forward(self._fun, argnum=argnum), argnum

    def __call__(self, *args, **kwargs):
        """Evaluates the gradient function, and saves the function value
        calculated during the forward pass in :attr:`.forward`."""
        grad_fn, argnum = self._get_grad_fn(args)

        if not isinstance(argnum, int) and not argnum:
            warnings.warn(
                "Attempted to differentiate a function with no trainable parameters. "
                "If this is unintended, please add trainable parameters via the "
                "'requires_grad' attribute or 'argnum' keyword."
            )
            self._forward = self._fun(*args, **kwargs)
            return ()

        grad_value, ans = grad_fn(*args, **kwargs)  # pylint: disable=not-callable
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
        vjp, ans = _make_vjp(fun, x)  # pylint: disable=redefined-outer-name

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
            with respect to. If a sequence is given, the Jacobian corresponding
            to all marked inputs and all output elements is returned.

    Returns:
        function: the function that returns the Jacobian of the input
        function with respect to the arguments in argnum

    .. note::
        Due to a limitation in Autograd, this function can only differentiate built-in scalar
        or NumPy array arguments.

    For ``argnum=None``, the trainable arguments are inferred dynamically from the arguments
    passed to the function. The returned function takes the same arguments as the original
    function and outputs a ``tuple``. The ``i`` th entry of the ``tuple`` has shape
    ``(*output shape, *shape of args[argnum[i]])``.

    If a single trainable argument is inferred, or if a single integer
    is provided as ``argnum``, the tuple is unpacked and its only entry is returned instead.

    **Example**

    Consider the QNode

    .. code-block::

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.RX(weights[0, 0, 0], wires=0)
            qml.RY(weights[0, 0, 1], wires=1)
            qml.RZ(weights[1, 0, 2], wires=0)
            return tuple(qml.expval(qml.PauliZ(w)) for w in dev.wires)

        weights = np.array(
            [[[0.2, 0.9, -1.4]], [[0.5, 0.2, 0.1]]], requires_grad=True
        )

    It has a single array-valued QNode argument with shape ``(2, 1, 3)`` and outputs
    a tuple of two expectation values. Therefore, the Jacobian of this QNode
    will be a single array with shape ``(2, 2, 1, 3)``:

    >>> qml.jacobian(circuit)(weights).shape
    (2, 2, 1, 3)

    On the other hand, consider the following QNode for the same circuit
    structure:

    .. code-block::

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=0)
            return tuple(qml.expval(qml.PauliZ(w)) for w in dev.wires)

        x = np.array(0.2, requires_grad=True)
        y = np.array(0.9, requires_grad=True)
        z = np.array(-1.4, requires_grad=True)

    It has three scalar QNode arguments and outputs a tuple of two expectation
    values. Consequently, its Jacobian will be a three-tuple of arrays with the
    shape ``(2,)``:

    >>> jac = qml.jacobian(circuit)(x, y, z)
    >>> type(jac)
    tuple
    >>> for sub_jac in jac:
    ...     print(sub_jac.shape)
    (2,)
    (2,)
    (2,)

    For a more advanced setting of QNode arguments, consider the QNode

    .. code-block::

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x[0], wires=0)
            qml.RY(y[0, 3], wires=1)
            qml.RX(x[1], wires=2)
            return [qml.expval(qml.PauliZ(w)) for w in [0, 1, 2]]

        x = np.array([0.1, 0.5], requires_grad=True)
        y = np.array([[-0.3, 1.2, 0.1, 0.9], [-0.2, -3.1, 0.5, -0.7]], requires_grad=True)

    If we do not provide ``argnum``, ``qml.jacobian`` will correctly identify both,
    ``x`` and ``y``, as trainable function arguments:

    >>> jac = qml.jacobian(circuit)(x, y)
    >>> print(type(jac), len(jac))
    <class 'tuple'> 2
    >>> qml.math.shape(jac[0])
    (3, 2)
    >>> qml.math.shape(jac[1])
    (3, 2, 4)

    As we can see, there are two entries in the output, one Jacobian for each
    QNode argument. The shape ``(3, 2)`` of the first Jacobian is the combination
    of the QNode output shape (``(3,)``) and the shape of ``x`` (``(2,)``).
    Similarly, the shape ``(2, 4)`` of ``y`` leads to a Jacobian shape ``(3, 2, 4)``.

    Instead we may choose the output to contain only one of the two
    entries by providing an iterable as ``argnum``:

    >>> jac = qml.jacobian(circuit, argnum=[1])(x, y)
    >>> print(type(jac), len(jac))
    <class 'tuple'> 1
    >>> qml.math.shape(jac)
    (1, 3, 2, 4)

    Here we included the size of the tuple in the shape analysis, corresponding to the
    first dimension of size ``1``.

    Finally, we may want to receive the single entry above directly, not as a tuple
    with a single entry. This is done by providing a single integer as ``argnum``

    >>> jac = qml.jacobian(circuit, argnum=1)(x, y)
    >>> print(type(jac), len(jac))
    <class 'numpy.ndarray'> 3
    >>> qml.math.shape(jac)
    (3, 2, 4)

    As expected, the tuple was unpacked and we directly received the Jacobian of the
    QNode with respect to ``y``.
    """
    # pylint: disable=no-value-for-parameter

    def _get_argnum(args):
        """Inspect the arguments for differentiability and return the
        corresponding indices."""
        argnum = []

        for idx, arg in enumerate(args):
            trainable = getattr(arg, "requires_grad", None) or isinstance(arg, ArrayBox)
            if trainable:
                argnum.append(idx)

        return argnum

    def _jacobian_function(*args, **kwargs):
        """Compute the autograd Jacobian.

        This wrapper function is returned to the user instead of autograd.jacobian,
        so that we can take into account cases where the user computes the
        jacobian function once, but then calls it with arguments that change
        in differentiability.
        """
        if argnum is None:
            # Infer which arguments to consider trainable
            _argnum = _get_argnum(args)
            # Infer whether to unpack from the inferred argnum
            unpack = len(_argnum) == 1
        else:
            # For a single integer as argnum, unpack the Jacobian tuple
            unpack = isinstance(argnum, int)
            _argnum = [argnum] if unpack else argnum

        if not _argnum:
            warnings.warn(
                "Attempted to differentiate a function with no trainable parameters. "
                "If this is unintended, please add trainable parameters via the "
                "'requires_grad' attribute or 'argnum' keyword."
            )
        jac = tuple(_jacobian(func, arg)(*args, **kwargs) for arg in _argnum)

        return jac[0] if unpack else jac

    return _jacobian_function


# pylint: disable=too-many-arguments
def vjp(f, params, cotangents, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product of PennyLane programs.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    .. warning::

        ``vjp`` is intended to be used with :func:`~.qjit` only.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler;
        see :func:`catalyst.vjp` for more details.
        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        f(Callable): Function-like object to calculate VJP for
        params(List[Array]): List (or a tuple) of f's arguments specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnum`` parameter.
        cotangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size
                                 and shapes must match the size and shape of ``f`` outputs.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns (Tuple[Array]):
        Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example**

    .. code-block:: python

        @qml.qjit
        def vjp(params, cotangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return qml.vjp(f, [params], [cotangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> dy = jnp.array([-0.5, 0.1, 0.3])
    >>> vjp(x, dy)
    [array([0.09983342, 0.04      , 0.02      ]),
    array([-0.43750208,  0.07000001])]
    """
    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.vjp(f, params, cotangents, method=method, h=h, argnum=argnum)

    raise CompileError("Pennylane does not support the VJP function without QJIT.")


# pylint: disable=too-many-arguments
def jvp(f, params, tangents, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian-vector product of PennyLane programs.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program.

    .. warning::

        ``jvp`` is intended to be used with :func:`~.qjit` only.

    .. note::

        When used with :func:`~.qjit`, this function only supports the Catalyst compiler;
        see :func:`catalyst.jvp` for more details.
        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    Args:
        f (Callable): Function-like object to calculate JVP for
        params (List[Array]): List (or a tuple) of the function arguments specifying the point
                              to calculate JVP at. A subset of these parameters are declared as
                              differentiable by listing their indices in the ``argnum`` parameter.
        tangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size and
                               shapes must match the ones of differentiable params.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns (Tuple[Array]):
        Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example 1 (basic usage)**

    .. code-block:: python

        @qml.qjit
        def jvp(params, tangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return qml.jvp(f, [params], [tangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> tangent = jnp.array([0.3, 0.6])
    >>> jvp(x, tangent)
    [array([0.09983342, 0.04      , 0.02      ]),
    array([0.29850125, 0.24000006, 0.12      ])]

    **Example 2 (argnum usage)**

    Here we show how to use ``argnum`` to ignore the non-differentiable parameter ``n`` of the
    target function. Note that the length and shapes of tangents must match the length and shape of
    primal parameters which we mark as differentiable by passing their indices to ``argnum``.

    .. code-block:: python

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(n, params):
            qml.RX(params[n, 0], wires=n)
            qml.RY(params[n, 1], wires=n)
            return qml.expval(qml.PauliZ(1))

        @qml.qjit
        def workflow(primals, tangents):
            return qml.jvp(circuit, [1, primals], [tangents], argnum=[1])

    >>> params = jnp.array([[0.54, 0.3154], [0.654, 0.123]])
    >>> dy = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    >>> workflow(params, dy)
    [array(0.78766064), array(-0.7011436)]
    """

    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        return ops_loader.jvp(f, params, tangents, method=method, h=h, argnum=argnum)

    raise CompileError("Pennylane does not support the JVP function without QJIT.")
