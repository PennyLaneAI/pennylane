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
"""Multiple dispatch functions"""
# pylint: disable=import-outside-toplevel,too-many-return-statements
import warnings
from collections.abc import Sequence
import functools

from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray

from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, get_interface, requires_grad


def _multi_dispatch(values):
    """Determines the correct framework to dispatch to given a
    sequence of tensor-like objects.

    Args:
        values (Sequence[tensor_like]): a sequence of tensor like objects

    Returns:
        str: the name of the interface

    To determine the framework to dispatch to, the following rules
    are applied:

    * Tensors that are incompatible (such as Torch and TensorFlow tensors)
      cannot both be present.

    * Autograd tensors *may* be present alongside Torch and TensorFlow tensors,
      but Torch and TensorFlow take precendence; the autograd arrays will
      be treated as non-differentiable NumPy arrays. A warning will be raised
      suggesting that vanilla NumPy be used instead.

    * Vanilla NumPy arrays and SciPy sparse matrices can be used alongside other tensor objects;
      they will always be treated as non-differentiable constants.
    """
    if "resource_variable" in getattr(values, "__module__", tuple()):
        values = np.asarray(values)

    interfaces = {get_interface(v) for v in values}

    if len(set(interfaces) - {"numpy", "scipy", "autograd"}) > 1:
        # contains multiple non-autograd interfaces
        raise ValueError("Tensors contain mixed types; cannot determine dispatch library")

    non_numpy_scipy_interfaces = set(interfaces) - {"numpy", "scipy"}

    if len(non_numpy_scipy_interfaces) > 1:
        # contains autograd and another interface
        warnings.warn(
            f"Contains tensors of types {non_numpy_scipy_interfaces}; dispatch will prioritize "
            "TensorFlow and PyTorch over autograd. Consider replacing Autograd with vanilla NumPy.",
            UserWarning,
        )

    if "tensorflow" in interfaces:
        return "tensorflow"

    if "torch" in interfaces:
        return "torch"

    if "autograd" in interfaces:
        return "autograd"

    if "jax" in interfaces:
        return "jax"

    return "numpy"


def multi_dispatch(argnum=None, tensor_list=None):
    r"""Decorater to dispatch arguments handled by the interface.

    This helps simplify definitions of new functions inside PennyLane. We can
    decorate the function, indicating the arguments that are tensors handled
    by the interface:


    >>> @qml.math.multi_dispatch(argnum=[0, 1])
    ... def some_function(tensor1, tensor2, option, like):
    ...     # the interface string is stored in `like`.
    ...     ...


    Args:
        argnum (list[int]): A list of integers indicating indicating the indices
            to dispatch (i.e., the arguments that are tensors handled by an interface).
            If ``None``, dispatch over all arguments.
        tensor_lists (list[int]): a list of integers indicating which indices
            in ``argnum`` are expected to be lists of tensors. If an argument
            marked as tensor list is not a ``tuple`` or ``list``, it is treated
            as if it was not marked as tensor list. If ``None``, this option is ignored.

    Returns:
        func: A wrapped version of the function, which will automatically attempt
        to dispatch to the correct autodifferentiation framework for the requested
        arguments. Note that the ``like`` argument will be optional, but can be provided
        if an explicit override is needed.

    .. seealso:: :func:`pennylane.math.multi_dispatch._multi_dispatch`

    .. note::
        This decorator makes the interface argument "like" optional as it utilizes
        the utility function `_multi_dispatch` to automatically detect the appropriate
        interface based on the tensor types.

    **Examples**

    We can redefine external functions to be suitable for PennyLane. Here, we
    redefine Autoray's ``stack`` function.

    >>> stack = multi_dispatch(argnum=0, tensor_list=0)(autoray.numpy.stack)

    We can also use the ``multi_dispatch`` decorator to dispatch
    arguments of more more elaborate custom functions. Here is an example
    of a ``custom_function`` that
    computes :math:`c \\sum_i (v_i)^T v_i`, where :math:`v_i` are vectors in ``values`` and
    :math:`c` is a fixed ``coefficient``. Note how ``argnum=0`` only points to the first argument ``values``,
    how ``tensor_list=0`` indicates that said first argument is a list of vectors, and that ``coefficient`` is not
    dispatched.

    >>> @math.multi_dispatch(argnum=0, tensor_list=0)
    >>> def custom_function(values, like, coefficient=10):
    >>>     # values is a list of vectors
    >>>     # like can force the interface (optional)
    >>>     if like == "tensorflow":
    >>>         # add interface-specific handling if necessary
    >>>     return coefficient * np.sum([math.dot(v,v) for v in values])

    We can then run

    >>> values = [np.array([1, 2, 3]) for _ in range(5)]
    >>> custom_function(values)
    700

    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            argnums = argnum if argnum is not None else list(range(len(args)))
            tensor_lists = tensor_list if tensor_list is not None else []

            if not isinstance(argnums, Sequence):
                argnums = [argnums]
            if not isinstance(tensor_lists, Sequence):
                tensor_lists = [tensor_lists]

            dispatch_args = []

            for a in argnums:
                # Only use extend if the marked argument really
                # is a (native) python Sequence
                if a in tensor_lists and isinstance(args[a], (list, tuple)):
                    dispatch_args.extend(args[a])
                else:
                    dispatch_args.append(args[a])

            interface = kwargs.pop("like", None)
            interface = interface or _multi_dispatch(dispatch_args)
            kwargs["like"] = interface

            return fn(*args, **kwargs)

        return wrapper

    return decorator


@multi_dispatch(argnum=[0], tensor_list=[0])
def block_diag(values, like=None):
    """Combine a sequence of 2D tensors to form a block diagonal tensor.

    Args:
        values (Sequence[tensor_like]): Sequence of 2D arrays/tensors to form
            the block diagonal tensor.

    Returns:
        tensor_like: the block diagonal tensor

    **Example**

    >>> t = [
    ...     np.array([[1, 2], [3, 4]]),
    ...     torch.tensor([[1, 2, 3], [-1, -6, -3]]),
    ...     torch.tensor(5)
    ... ]
    >>> qml.math.block_diag(t)
    tensor([[ 1,  2,  0,  0,  0,  0],
            [ 3,  4,  0,  0,  0,  0],
            [ 0,  0,  1,  2,  3,  0],
            [ 0,  0, -1, -6, -3,  0],
            [ 0,  0,  0,  0,  0,  5]])
    """
    values = np.coerce(values, like=like)
    return np.block_diag(values, like=like)


@multi_dispatch(argnum=[0], tensor_list=[0])
def concatenate(values, axis=0, like=None):
    """Concatenate a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            concatenate. The objects must have the same shape, except in the dimension corresponding
            to axis (the first, by default).
        axis (int): The axis along which the input tensors are concatenated. If axis is None,
            tensors are flattened before use. Default is 0.

    Returns:
        tensor_like: The concatenated tensor.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> concatenate([x, y, z])
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([6.00e-01, 1.00e-01, 6.00e-01, 1.00e-01, 2.00e-01, 3.00e-01, 5.00e+00, 8.00e+00, 1.01e+02], dtype=float32)>
    """
    if like == "torch":
        import torch

        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            values = [np.flatten(torch.as_tensor(t)) for t in values]
            axis = 0
        else:
            values = [torch.as_tensor(t) for t in values]

    if like == "tensorflow" and axis is None:
        # flatten and then concatenate zero'th dimension
        # to reproduce numpy's behaviour
        values = [np.flatten(np.array(t)) for t in values]
        axis = 0

    return np.concatenate(values, axis=axis, like=like)


@multi_dispatch(argnum=[0], tensor_list=[0])
def diag(values, k=0, like=None):
    """Construct a diagonal tensor from a list of scalars.

    Args:
        values (tensor_like or Sequence[scalar]): sequence of numeric values that
            make up the diagonal
        k (int): The diagonal in question. ``k=0`` corresponds to the main diagonal.
            Use ``k>0`` for diagonals above the main diagonal, and ``k<0`` for
            diagonals below the main diagonal.

    Returns:
        tensor_like: the 2D diagonal tensor

    **Example**

    >>> x = [1., 2., tf.Variable(3.)]
    >>> qml.math.diag(x)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]], dtype=float32)>
    >>> y = tf.Variable([0.65, 0.2, 0.1])
    >>> qml.math.diag(y, k=-1)
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.65, 0.  , 0.  , 0.  ],
           [0.  , 0.2 , 0.  , 0.  ],
           [0.  , 0.  , 0.1 , 0.  ]], dtype=float32)>
    >>> z = torch.tensor([0.1, 0.2])
    >>> qml.math.diag(z, k=1)
    tensor([[0.0000, 0.1000, 0.0000],
            [0.0000, 0.0000, 0.2000],
            [0.0000, 0.0000, 0.0000]])
    """
    if isinstance(values, (list, tuple)):
        values = np.stack(np.coerce(values, like=like), like=like)

    return np.diag(values, k=k, like=like)


@multi_dispatch(argnum=[0, 1])
def dot(tensor1, tensor2, like=None):
    """Returns the matrix or dot product of two tensors.

    * If both tensors are 0-dimensional, elementwise multiplication
      is performed and a 0-dimensional scalar returned.

    * If both tensors are 1-dimensional, the dot product is returned.

    * If the first array is 2-dimensional and the second array 1-dimensional,
      the matrix-vector product is returned.

    * If both tensors are 2-dimensional, the matrix product is returned.

    * Finally, if the the first array is N-dimensional and the second array
      M-dimensional, a sum product over the last dimension of the first array,
      and the second-to-last dimension of the second array is returned.

    Args:
        tensor1 (tensor_like): input tensor
        tensor2 (tensor_like): input tensor

    Returns:
        tensor_like: the matrix or dot product of two tensors
    """
    x, y = np.coerce([tensor1, tensor2], like=like)

    if like == "torch":
        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    if like == "tensorflow":
        if len(np.shape(x)) == 0 and len(np.shape(y)) == 0:
            return x * y

        if len(np.shape(y)) == 1:
            return np.tensordot(x, y, axes=[[-1], [0]], like=like)

        if len(np.shape(x)) == 2 and len(np.shape(y)) == 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    return np.dot(x, y, like=like)


@multi_dispatch(argnum=[0, 1])
def tensordot(tensor1, tensor2, axes=None, like=None):
    """Returns the tensor product of two tensors.
    In general ``axes`` specifies either the set of axes for both
    tensors that are contracted (with the first/second entry of ``axes``
    giving all axis indices for the first/second tensor) or --- if it is
    an integer --- the number of last/first axes of the first/second
    tensor to contract over.
    There are some non-obvious special cases:

    * If both tensors are 0-dimensional, ``axes`` must be 0.
      and a 0-dimensional scalar is returned containing the simple product.

    * If both tensors are 1-dimensional and ``axes=0``, the outer product
      is returned.

    * Products between a non-0-dimensional and a 0-dimensional tensor are not
      supported in all interfaces.

    Args:
        tensor1 (tensor_like): input tensor
        tensor2 (tensor_like): input tensor
        axes (int or list[list[int]]): Axes to contract over, see detail description.

    Returns:
        tensor_like: the tensor product of the two input tensors
    """
    tensor1, tensor2 = np.coerce([tensor1, tensor2], like=like)
    return np.tensordot(tensor1, tensor2, axes=axes, like=like)


@multi_dispatch(argnum=[0], tensor_list=[0])
def get_trainable_indices(values, like=None):
    """Returns a set containing the trainable indices of a sequence of
    values.

    Args:
        values (Iterable[tensor_like]): Sequence of tensor-like objects to inspect

    Returns:
        set[int]: Set containing the indices of the trainable tensor-like objects
        within the input sequence.

    **Example**

    >>> def cost_fn(params):
    ...     print("Trainable:", qml.math.get_trainable_indices(params))
    ...     return np.sum(np.sin(params[0] * params[1]))
    >>> values = [np.array([0.1, 0.2], requires_grad=True),
    ... np.array([0.5, 0.2], requires_grad=False)]
    >>> cost_fn(values)
    Trainable: {0}
    tensor(0.0899685, requires_grad=True)
    """
    trainable = requires_grad
    trainable_params = set()

    if like == "jax":
        import jax

        if not any(isinstance(v, jax.core.Tracer) for v in values):
            # No JAX tracing is occuring; treat all `DeviceArray` objects as trainable.
            trainable = lambda p, **kwargs: isinstance(p, jax.numpy.DeviceArray)
        else:
            # JAX tracing is occuring; use the default behaviour (only traced arrays
            # are treated as trainable). This is required to ensure that `jax.grad(func, argnums=...)
            # works correctly, as the argnums argnument determines which parameters are
            # traced arrays.
            trainable = requires_grad

    for idx, p in enumerate(values):
        if trainable(p, interface=like):
            trainable_params.add(idx)

    return trainable_params


def ones_like(tensor, dtype=None):
    """Returns a tensor of all ones with the same shape and dtype
    as the input tensor.

    Args:
        tensor (tensor_like): input tensor
        dtype (str, np.dtype, None): The desired output datatype of the array. If not provided, the dtype of
            ``tensor`` is used. This argument can be any supported NumPy dtype representation, including
            a string (``"float64"``), a ``np.dtype`` object (``np.dtype("float64")``), or
            a dtype class (``np.float64``). If ``tensor`` is not a NumPy array, the
            **equivalent** dtype in the dispatched framework is used.

    Returns:
        tensor_like: an all-ones tensor with the same shape and
        size as ``tensor``

    **Example**

    >>> x = torch.tensor([1., 2.])
    >>> ones_like(x)
    tensor([1, 1])
    >>> y = tf.Variable([[0], [5]])
    >>> ones_like(y, dtype=np.complex128)
    <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[1.+0.j],
           [1.+0.j]])>
    """
    if dtype is not None:
        return cast(np.ones_like(tensor), dtype)

    return np.ones_like(tensor)


def safe_squeeze(tensor, axis=None, exclude_axis=None):
    """Squeeze a tensor either along all axes, specified axes or all
    but a set of excluded axes. For selective squeezing, catch errors
    and do nothing if the selected axes do not have size 1.

    Args:
        tensor (tensor_like): input tensor
        axis (int or Sequence[int]): Axis/axes to squeeze
        exclude_axis (int or Sequence[int]): Axis/axes not to squeeze

    Return:
        tensor_like: The input tensor with those axes removed that were specified
        or not excluded and that have size 1. If no axes are specified or excluded,
        all axes are attempted to be squeezed.
    """
    if get_interface(tensor) == "tensorflow":
        from tensorflow.python.framework.errors_impl import InvalidArgumentError

        exception = InvalidArgumentError
    else:
        exception = ValueError

    num_axes = len(np.shape(tensor))
    if axis is None:
        if exclude_axis is None:
            return np.squeeze(tensor)
        if np.isscalar(exclude_axis):
            exclude_axis = [exclude_axis if exclude_axis >= 0 else num_axes + exclude_axis]
        else:
            exclude_axis = [(i if i >= 0 else num_axes + i) for i in exclude_axis]
        axis = [i for i in range(num_axes) if not i in exclude_axis]
    elif np.isscalar(axis):
        axis = [axis if axis >= 0 else num_axes + axis]
    else:
        axis = [(i if i >= 0 else num_axes + i) for i in axis]

    for ax in sorted(set(axis)):
        # Modify the axis index to squeeze by the number of squeezes already
        # performed - this works because we squeeze in increasing axis index order.
        # If the axis index is negative, no modification is needed.
        ax -= num_axes - len(np.shape(tensor))
        try:
            tensor = np.squeeze(tensor, axis=ax)
        except exception:
            pass
    return tensor


@multi_dispatch(argnum=[0], tensor_list=[0])
def stack(values, axis=0, like=None):
    """Stack a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            stack. Each object in the sequence must have the same size in the given axis.
        axis (int): The axis along which the input tensors are stacked. ``axis=0`` corresponds
            to vertical stacking.

    Returns:
        tensor_like: The stacked array. The stacked array will have one additional dimension
        compared to the unstacked tensors.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> stack([x, y, z])
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[6.00e-01, 1.00e-01, 6.00e-01],
           [1.00e-01, 2.00e-01, 3.00e-01],
           [5.00e+00, 8.00e+00, 1.01e+02]], dtype=float32)>
    """
    values = np.coerce(values, like=like)
    return np.stack(values, axis=axis, like=like)


def where(condition, x=None, y=None):
    """Returns elements chosen from x or y depending on a boolean tensor condition,
    or the indices of entries satisfying the condition.

    The input tensors ``condition``, ``x``, and ``y`` must all be broadcastable to the same shape.

    Args:
        condition (tensor_like[bool]): A boolean tensor. Where ``True`` , elements from
            ``x`` will be chosen, otherwise ``y``. If ``x`` and ``y`` are ``None`` the
            indices where ``condition==True`` holds will be returned.
        x (tensor_like): values from which to choose if the condition evaluates to ``True``
        y (tensor_like): values from which to choose if the condition evaluates to ``False``

    Returns:
        tensor_like or tuple[tensor_like]: If ``x is None`` and ``y is None``, a tensor
        or tuple of tensors with the indices where ``condition`` is ``True`` .
        Else, a tensor with elements from ``x`` where the ``condition`` is ``True``,
        and ``y`` otherwise. In this case, the output tensor has the same shape as
        the input tensors.

    **Example with three arguments**

    >>> a = torch.tensor([0.6, 0.23, 0.7, 1.5, 1.7], requires_grad=True)
    >>> b = torch.tensor([-1., -2., -3., -4., -5.], requires_grad=True)
    >>> math.where(a < 1, a, b)
    tensor([ 0.6000,  0.2300,  0.7000, -4.0000, -5.0000], grad_fn=<SWhereBackward>)

    .. warning::

        The output format for ``x=None`` and ``y=None`` follows the respective
        interface and differs between TensorFlow and all other interfaces:
        For TensorFlow, the output is a tensor with shape
        ``(num_true, len(condition.shape))`` where ``num_true`` is the number
        of entries in ``condition`` that are ``True`` .
        The entry at position ``(i, j)`` is the ``j`` th entry of the ``i`` th
        index.
        For all other interfaces, the output is a tuple of tensor-like objects,
        with the ``j`` th object indicating the ``j`` th entries of all indices.
        Also see the examples below.

    **Example with single argument**

    For Torch, Autograd, JAX and NumPy, the output formatting is as follows:

    >>> a = [[0.6, 0.23, 1.7],[1.5, 0.7, -0.2]]
    >>> math.where(torch.tensor(a) < 1)
    (tensor([0, 0, 1, 1]), tensor([0, 1, 1, 2]))

    This is not a single tensor-like object but corresponds to the shape
    ``(2, 4)`` . For TensorFlow, on the other hand:

    >>> math.where(tf.constant(a) < 1)
    tf.Tensor(
    [[0 0]
     [0 1]
     [1 1]
     [1 2]], shape=(4, 2), dtype=int64)

    As we can see, the dimensions are swapped and the output is a single Tensor.
    Note that the number of dimensions of the output does *not* depend on the input
    shape, it is always two-dimensional.

    """
    if x is None and y is None:
        interface = _multi_dispatch([condition])
        return np.where(condition, like=interface)

    return np.where(condition, x, y, like=_multi_dispatch([condition, x, y]))


@multi_dispatch(argnum=[0, 1])
def frobenius_inner_product(A, B, normalize=False, like=None):
    r"""Frobenius inner product between two matrices.

    .. math::

        \langle A, B \rangle_F = \sum_{i,j=1}^n A_{ij} B_{ij} = \operatorname{tr} (A^T B)

    The Frobenius inner product is equivalent to the Hilbert-Schmidt inner product for
    matrices with real-valued entries.

    Args:
        A (tensor_like[float]): First matrix, assumed to be a square array.
        B (tensor_like[float]): Second matrix, assumed to be a square array.
        normalize (bool): If True, divide the inner product by the Frobenius norms of A and B.

    Returns:
        float: Frobenius inner product of A and B

    **Example**

    >>> A = np.random.random((3,3))
    >>> B = np.random.random((3,3))
    >>> qml.math.frobenius_inner_product(A, B)
    3.091948202943376
    """
    A, B = np.coerce([A, B], like=like)

    inner_product = np.sum(A * B)

    if normalize:
        norm = np.sqrt(np.sum(A * A) * np.sum(B * B))
        inner_product = inner_product / norm

    return inner_product


@multi_dispatch(argnum=[0, 2])
def scatter_element_add(tensor, index, value, like=None):
    """In-place addition of a multidimensional value over various
    indices of a tensor.

    Args:
        tensor (tensor_like[float]): Tensor to add the value to
        index (tuple or list[tuple]): Indices to which to add the value
        value (float or tensor_like[float]): Value to add to ``tensor``
        like (str): Manually chosen interface to dispatch to.
    Returns:
        tensor_like[float]: The tensor with the value added at the given indices.

    **Example**

    >>> tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> index = (1, 2)
    >>> value = -3.1
    >>> qml.math.scatter_element_add(tensor, index, value)
    tensor([[ 0.1000,  0.2000,  0.3000],
            [ 0.4000,  0.5000, -2.5000]])

    If multiple indices are given, in the form of a list of tuples, the
    ``k`` th tuple is interpreted to contain the ``k`` th entry of all indices:

    >>> indices = [(1, 0), (2, 1)] # This will modify the entries (1, 2) and (0, 1)
    >>> values = torch.tensor([10, 20])
    >>> qml.math.scatter_element_add(tensor, indices, values)
    tensor([[ 0.1000, 20.2000,  0.3000],
            [ 0.4000,  0.5000, 10.6000]])
    """
    if len(np.shape(tensor)) == 0 and index == ():
        return tensor + value

    return np.scatter_element_add(tensor, index, value, like=like)


def unwrap(values, max_depth=None):
    """Unwrap a sequence of objects to NumPy arrays.

    Note that tensors on GPUs will automatically be copied
    to the CPU.

    Args:
        values (Sequence[tensor_like]): sequence of tensor-like objects to unwrap
        max_depth (int): Positive integer indicating the depth of unwrapping to perform
            for nested tensor-objects. This argument only applies when unwrapping
            Autograd ``ArrayBox`` objects.

    **Example**

    >>> values = [np.array([0.1, 0.2]), torch.tensor(0.1, dtype=torch.float64), torch.tensor([0.5, 0.2])]
    >>> math.unwrap(values)
    [array([0.1, 0.2]), 0.1, array([0.5, 0.2], dtype=float32)]

    This function will continue to work during backpropagation:

    >>> def cost_fn(params):
    ...     unwrapped_params = math.unwrap(params)
    ...     print("Unwrapped:", [(i, type(i)) for i in unwrapped_params])
    ...     return np.sum(np.sin(params))
    >>> params = np.array([0.1, 0.2, 0.3])
    >>> grad = autograd.grad(cost_fn)(params)
    Unwrapped: [(0.1, <class 'float'>), (0.2, <class 'float'>), (0.3, <class 'float'>)]
    >>> print(grad)
    [0.99500417 0.98006658 0.95533649]
    """
    res = []

    for t in values:
        if isinstance(t, ArrayBox):
            a = np.to_numpy(t, max_depth=max_depth)
        else:
            a = np.to_numpy(t)

        if isinstance(a, ndarray) and not a.shape:
            # if NumPy array is scalar, convert to a Python float
            res.append(a.tolist())
        else:
            res.append(a)

    return res
