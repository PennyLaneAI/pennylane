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
import functools
from collections.abc import Sequence
from operator import attrgetter

# pylint: disable=wrong-import-order
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray

from . import single_dispatch  # pylint:disable=unused-import
from .interface_utils import get_interface
from .utils import cast, cast_like, requires_grad


# pylint:disable=redefined-outer-name
def array(*args, like=None, **kwargs):
    """Creates an array or tensor object of the target framework.

    If the PyTorch interface is specified, this method preserves the Torch device used.
    If the JAX interface is specified, this method uses JAX numpy arrays, which do not cause issues with jit tracers.

    Returns:
        tensor_like: the tensor_like object of the framework
    """
    res = np.array(*args, like=like, **kwargs)
    if like is not None and get_interface(like) == "torch":
        res = res.to(device=like.device)
    return res


def eye(*args, like=None, **kwargs):
    """Creates an identity array or tensor object of the target framework.

    This method preserves the Torch device used.

    Returns:
        tensor_like: the tensor_like object of the framework
    """
    res = np.eye(*args, like=like, **kwargs)
    if like is not None and get_interface(like) == "torch":
        res = res.to(device=like.device)
    return res


def multi_dispatch(argnum=None, tensor_list=None):
    r"""Decorator to dispatch arguments handled by the interface.

    This helps simplify definitions of new functions inside PennyLane. We can
    decorate the function, indicating the arguments that are tensors handled
    by the interface:


    >>> @qml.math.multi_dispatch(argnum=[0, 1])
    ... def some_function(tensor1, tensor2, option, like):
    ...     # the interface string is stored in `like`.
    ...     ...


    Args:
        argnum (list[int]): A list of integers indicating the indices
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
            interface = interface or get_interface(*dispatch_args)
            kwargs["like"] = interface

            return fn(*args, **kwargs)

        return wrapper

    return decorator


@multi_dispatch(argnum=[0, 1])
def kron(*args, like=None, **kwargs):
    """The kronecker/tensor product of args."""
    if like == "scipy":
        return onp.kron(*args, **kwargs)  # Dispatch scipy kron to numpy backed specifically.

    if like == "torch":
        # Extract all the devices for the incoming tensors
        devs = set(map(attrgetter("device"), args))
        devs = list(devs)
        # If multiple devices found, choose the non-CPU device as the default
        if len(devs) > 1:  # Assuming "cpu" and non-"cpu" are the only options
            dev = devs[0] if getattr(devs[0], "type", str(devs[0])) != "cpu" else devs[1]
        else:
            dev = devs[0]
        # Migrate the tensors to all be on the chosen device, if necessary
        mats = [np.asarray(arg, like="torch", device=dev) for arg in args]
        return np.kron(*mats)

    return np.kron(*args, like=like, **kwargs)


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
    ...     torch.tensor([[5]])
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
    <tf.Tensor: shape=(9,), dtype=float32, numpy=
    array([6.00e-01, 1.00e-01, 6.00e-01, 1.00e-01, 2.00e-01, 3.00e-01,
           5.00e+00, 8.00e+00, 1.01e+02], dtype=float32)>
    """

    if like == "torch":
        import torch

        device_set = set()
        dev_indices = set()
        torch_device = None
        for t in values:
            if isinstance(t, torch.Tensor):
                device_set.add(t.device.type)
                dev_indices.add(t.device.index)

        # TODO: Remove the no-cover pragma once we are able to test with multiple GPUs on CI.
        if device_set:  # pragma: no cover
            # If data exists on two separate GPUs, outright fail
            if len(dev_indices) > 1:
                device_names = ", ".join(str(d) for d in device_set)

                raise RuntimeError(
                    f"Expected all tensors to be on the same device, but found at least two devices, {device_names}!"
                )

            device = device_set.pop()
            dev_id = dev_indices.pop() if dev_indices else None
            torch_device = torch.device(f"{device}:{dev_id}" if dev_id is not None else device)

        else:  # pragma: no cover
            torch_device = torch.device("cpu")

        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            values = [
                np.flatten(torch.as_tensor(t, device=torch_device))  # pragma: no cover
                for t in values
            ]
            axis = 0
        else:
            values = [torch.as_tensor(t, device=torch_device) for t in values]  # pragma: no cover

    if (
        like == "tensorflow" and axis is None
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
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
def matmul(tensor1, tensor2, like=None):
    """Returns the matrix product of two tensors."""
    if like == "torch":
        if get_interface(tensor1) != "torch":
            tensor1 = np.asarray(tensor1, like="torch")
        if get_interface(tensor2) != "torch":
            tensor2 = np.asarray(tensor2, like="torch")
        tensor2 = cast_like(tensor2, tensor1)  # pylint: disable=arguments-out-of-order
    return np.matmul(tensor1, tensor2, like=like)


@multi_dispatch(argnum=[0, 1])
def dot(tensor1, tensor2, like=None):
    """Returns the matrix or dot product of two tensors.

    * If either tensor is 0-dimensional, elementwise multiplication
      is performed and a 0-dimensional scalar or a tensor with the
      same dimensions as the other tensor is returned.

    * If both tensors are 1-dimensional, the dot product is returned.

    * If the first array is 2-dimensional and the second array 1-dimensional,
      the matrix-vector product is returned.

    * If both tensors are 2-dimensional, the matrix product is returned.

    * Finally, if the first array is N-dimensional and the second array
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

        if x.ndim == 0 or y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    if like in {"tensorflow", "autograd"}:

        ndim_y = len(np.shape(y))
        ndim_x = len(np.shape(x))

        if ndim_x == 0 or ndim_y == 0:
            return x * y

        if ndim_y == 1:
            return np.tensordot(x, y, axes=[[-1], [0]], like=like)

        if ndim_x == 2 and ndim_y == 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    if like == "scipy":
        # See https://github.com/scipy/scipy/issues/18938 for the issue
        # with scipy sparse and np dot product

        # Avoid the case when one is a scalar - using a robust check for scalars
        if onp.isscalar(x) or onp.isscalar(y):
            return x * y
        return x.dot(y)
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

    >>> from pennylane import numpy as np
    >>> def cost_fn(params):
    ...     print("Trainable:", qml.math.get_trainable_indices(params))
    ...     return np.sum(np.sin(params[0] * params[1]))
    >>> values = [np.array([0.1, 0.2], requires_grad=True),
    ... np.array([0.5, 0.2], requires_grad=False)]
    >>> cost_fn(values)
    Trainable: {0}
    tensor(0.0899685, requires_grad=True)
    """
    trainable_params = set()

    for idx, p in enumerate(values):
        if requires_grad(p, interface=like):
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
    tensor([1., 1.])
    >>> y = tf.Variable([[0], [5]])
    >>> ones_like(y, dtype=np.complex128)
    <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[1.+0.j],
           [1.+0.j]])>
    """
    if dtype is not None:
        return cast(np.ones_like(tensor), dtype)

    return np.ones_like(tensor)


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


def einsum(indices, *operands, like=None, optimize=None):
    """Evaluates the Einstein summation convention on the operands.

    Args:
        indices (str): Specifies the subscripts for summation as comma separated list of
            subscript labels. An implicit (classical Einstein summation) calculation is
            performed unless the explicit indicator ‘->’ is included as well as subscript
            labels of the precise output form.
        *operands (tuple[tensor_like]): The tensors for the operation.

    Returns:
        tensor_like: The calculation based on the Einstein summation convention.

    **Examples**

    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> qml.math.einsum('ii', a)
    60

    Extract the diagonal (requires explicit form):

    >>> qml.math.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> qml.math.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    """
    if like is None:
        like = get_interface(*operands)
    operands = np.coerce(operands, like=like)
    if optimize is None or like == "torch":
        # torch einsum doesn't support the optimize keyword argument
        return np.einsum(indices, *operands, like=like)
    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        # Unpacking and casting necessary for higher order derivatives,
        # and avoiding implicit fp32 down-conversions.
        op1, op2 = operands
        op1 = array(op1, like=op1[0], dtype=op1[0].dtype)
        op2 = array(op2, like=op2[0], dtype=op2[0].dtype)
        return np.einsum(indices, op1, op2, like=like)
    return np.einsum(indices, *operands, like=like, optimize=optimize)


def where(condition, x=None, y=None):
    r"""Returns elements chosen from x or y depending on a boolean tensor condition,
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
        ``(len(condition.shape), num_true)`` where ``num_true`` is the number
        of entries in ``condition`` that are ``True`` .
        For all other interfaces, the output is a tuple of tensor-like objects,
        with the ``j``\ th object indicating the ``j``\ th entries of all indices.
        Also see the examples below.

    **Example with single argument**

    For Torch, Autograd, JAX and NumPy, the output formatting is as follows:

    >>> a = [[0.6, 0.23, 1.7],[1.5, 0.7, -0.2]]
    >>> math.where(torch.tensor(a) < 1)
    (tensor([0, 0, 1, 1]), tensor([0, 1, 1, 2]))

    This is not a single tensor-like object but corresponds to the shape
    ``(2, 4)`` . For TensorFlow, on the other hand:

    >>> math.where(tf.constant(a) < 1)
    <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
    array([[0, 0, 1, 1],
           [0, 1, 1, 2]])>

    Note that the number of dimensions of the output does *not* depend on the input
    shape, it is always two-dimensional.

    """
    if x is None and y is None:
        interface = get_interface(condition)
        res = np.where(condition, like=interface)

        if (
            interface == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            return np.transpose(np.stack(res))

        return res

    interface = get_interface(condition, x, y)
    res = np.where(condition, x, y, like=interface)

    return res


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


@multi_dispatch(argnum=[1])
def scatter(indices, array, new_dims, like=None):
    """Scatters an array into a tensor of shape new_dims according to indices.

    This operation is similar to scatter_element_add, except that the tensor
    is zero-initialized. Calling scatter(indices, array, new_dims) is identical
    to calling scatter_element_add(np.zeros(new_dims), indices, array)

    Args:
        indices (tensor_like[int]): Indices to update
        array (tensor_like[float]): Values to assign to the new tensor
        new_dims (int or tuple[int]): The shape of the new tensor
        like (str): Manually chosen interface to dispatch to.
    Returns:
        tensor_like[float]: The tensor with the values modified the given indices.

    **Example**

    >>> indices = np.array([4, 3, 1, 7])
    >>> updates = np.array([9, 10, 11, 12])
    >>> shape = 8
    >>> qml.math.scatter(indices, updates, shape)
    array([ 0, 11,  0, 10,  9,  0,  0, 12])
    """
    return np.scatter(indices, array, new_dims, like=like)


# pylint: disable=too-many-arguments
@multi_dispatch(argnum=[0, 2])
def scatter_element_add(
    tensor, index, value, like=None, *, indices_are_sorted=False, unique_indices=False
):
    """In-place addition of a multidimensional value over various
    indices of a tensor.

    Args:
        tensor (tensor_like[float]): Tensor to add the value to
        index (tuple or list[tuple]): Indices to which to add the value
        value (float or tensor_like[float]): Value to add to ``tensor``
        like (str): Manually chosen interface to dispatch to.

    Keyword Args:
        indices_are_sorted=False (bool): If ``True``, jax will assume that the indices are in
            ascending order. Required to be ``True`` with catalyst.
        unique_indices=False (bool): If ``True``, jax will assume each index is unique.
            Required to be ``True`` with catalyst.

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

    return np.scatter_element_add(
        tensor,
        index,
        value,
        like=like,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )


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
    Unwrapped: [(0.1, <class 'numpy.float64'>), (0.2, <class 'numpy.float64'>), (0.3, <class 'numpy.float64'>)]
    >>> print(grad)
    [0.99500417 0.98006658 0.95533649]
    """

    def convert(val):
        if isinstance(val, (tuple, list)):
            return unwrap(val)
        new_val = (
            np.to_numpy(val, max_depth=max_depth) if isinstance(val, ArrayBox) else np.to_numpy(val)
        )
        return new_val.tolist() if isinstance(new_val, ndarray) and not new_val.shape else new_val

    if isinstance(values, (tuple, list)):
        return type(values)(convert(val) for val in values)
    return (
        np.to_numpy(values, max_depth=max_depth)
        if isinstance(values, ArrayBox)
        else np.to_numpy(values)
    )


@multi_dispatch(argnum=[0, 1])
def add(*args, like=None, **kwargs):
    """Add arguments element-wise."""
    if like == "scipy":
        return onp.add(*args, **kwargs)  # Dispatch scipy add to numpy backed specifically.

    arg_interfaces = {get_interface(args[0]), get_interface(args[1])}

    # case of one torch tensor and one vanilla numpy array
    if like == "torch" and len(arg_interfaces) == 2:
        # In autoray 0.6.5, np.add dispatches to torch instead of
        # numpy if one parameter is a torch tensor and the other is
        # a numpy array. torch.add raises an Exception if one of the
        # arguments is a numpy array, so here we cast both arguments
        # to be tensors.
        dev = getattr(args[0], "device", None) or getattr(args[1], "device")
        arg0 = np.asarray(args[0], device=dev, like=like)
        arg1 = np.asarray(args[1], device=dev, like=like)
        return np.add(arg0, arg1, *args[2:], **kwargs)

    return np.add(*args, **kwargs, like=like)


@multi_dispatch()
def iscomplex(tensor, like=None):
    """Return True if the tensor has a non-zero complex component."""
    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        imag_tensor = tf.math.imag(tensor)
        return tf.math.count_nonzero(imag_tensor) > 0

    if like == "torch":
        import torch

        if torch.is_complex(tensor):
            imag_tensor = torch.imag(tensor)
            return torch.count_nonzero(imag_tensor) > 0
        return False

    return np.iscomplex(tensor)


@multi_dispatch()
def expm(tensor, like=None):
    """Compute the matrix exponential of an array :math:`e^{X}`.

    ..note::
        This function is not differentiable with Autograd, as it
        relies on the scipy implementation.
    """
    if like == "torch":
        return tensor.matrix_exp()
    if like == "jax":
        from jax.scipy.linalg import expm as jax_expm

        return jax_expm(tensor)
    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        return tf.linalg.expm(tensor)
    from scipy.linalg import expm as scipy_expm

    return scipy_expm(tensor)


@multi_dispatch()
def norm(tensor, like=None, **kwargs):
    """Compute the norm of a tensor in each interface."""
    if like == "jax":
        from jax.numpy.linalg import norm

    elif (
        like == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        from tensorflow import norm

    elif like == "torch":
        from torch.linalg import norm

        if "axis" in kwargs:
            axis_val = kwargs.pop("axis")
            kwargs["dim"] = axis_val

    elif (
        like == "autograd" and kwargs.get("ord", None) is None and kwargs.get("axis", None) is None
    ):
        norm = _flat_autograd_norm

    elif like == "scipy":
        from scipy.sparse.linalg import norm

    else:
        from scipy.linalg import norm

    return norm(tensor, **kwargs)


@multi_dispatch(argnum=[0])
def svd(tensor, like=None, **kwargs):
    r"""Compute the singular value decomposition of a tensor in each interface.

    The singular value decomposition for a matrix :math:`A` consist of three matrices :math:`S`,
    :math:`U` and :math:`V_h`, such that:

    .. math::

        A = U \cdot Diag(S) \cdot V_h

    Args:
        tensor (tensor_like): input tensor
        compute_uv (bool):  if ``True``, the full decomposition is returned


    Returns:
        :math:`S`, :math:`U` and :math:`V_h` or :math:`S`: full decomposition
        if ``compute_uv`` is ``True`` or ``None``, or only the singular values
        if ``compute_uv`` is ``False``
    """
    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        from tensorflow.linalg import adjoint, svd

        # Tensorflow results need some post-processing to keep it similar to other frameworks.

        if kwargs.get("compute_uv", True):
            S, U, V = svd(tensor, **kwargs)
            return U, S, adjoint(V)
        return svd(tensor, **kwargs)

    if like == "jax":
        from jax.numpy.linalg import svd

    elif like == "torch":
        # Torch is deprecating torch.svd() in favour of torch.linalg.svd().
        # The new UI is slightly different and breaks the logic for the multi dispatching.
        # This small workaround restores the compute_uv control argument.
        if kwargs.get("compute_uv", True) is False:
            from torch.linalg import svdvals as svd
        else:
            from torch.linalg import svd
        if kwargs.get("compute_uv", None) is not None:
            kwargs.pop("compute_uv")

    else:
        from numpy.linalg import svd

    return svd(tensor, **kwargs)


def _flat_autograd_norm(tensor, **kwargs):  # pylint: disable=unused-argument
    """Helper function for computing the norm of an autograd tensor when the order or axes are not
    specified. This is used for differentiability."""
    x = np.ravel(tensor)
    sq_norm = np.dot(x, np.conj(x))
    return np.real(np.sqrt(sq_norm))


@multi_dispatch(argnum=[1])
def gammainc(m, t, like=None):
    r"""Return the lower incomplete Gamma function.

    The lower incomplete Gamma function is defined in scipy as

    .. math::

        \gamma(m, t) = \frac{1}{\Gamma(m)} \int_{0}^{t} x^{m-1} e^{-x} dx,

    where :math:`\Gamma` denotes the Gamma function.

     Args:
        m (float): exponent of the incomplete Gamma function
        t (array[float]): upper limit of the incomplete Gamma function

    Returns:
        (array[float]): value of the incomplete Gamma function
    """
    if like == "jax":
        from jax.scipy.special import gammainc

        return gammainc(m, t)

    if like == "autograd":
        from autograd.scipy.special import gammainc

        return gammainc(m, t)

    import scipy

    return scipy.special.gammainc(m, t)


@multi_dispatch()
def detach(tensor, like=None):
    """Detach a tensor from its trace and return just its numerical values.

    Args:
        tensor (tensor_like): Tensor to detach
        like (str): Manually chosen interface to dispatch to.

    Returns:
        tensor_like: A tensor in the same interface as the input tensor but
        with a stopped gradient.
    """
    if like == "jax":
        import jax

        return jax.lax.stop_gradient(tensor)

    if like == "torch":
        return tensor.detach()

    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        return tf.stop_gradient(tensor)

    if like == "autograd":
        return np.to_numpy(tensor)

    return tensor


@multi_dispatch(tensor_list=[1])
def set_index(array, idx, val, like=None):
    """Set the value at a specified index in an array.
    Calls ``array[idx]=val`` and returns the updated array unless JAX or Tensorflow.

    Args:
        array (tensor_like): array to be modified
        idx (int, tuple): index to modify
        val (int, float): value to set

    Returns:
        a new copy of the array with the specified index updated to ``val``.

    Whether the original array is modified is interface-dependent.
    """
    if like == "jax":
        from jax import numpy as jnp

        # ensure array is jax array (interface may be jax because of idx or val and not array)
        jax_array = jnp.array(array)
        return jax_array.at[idx].set(val)

    if like == "tensorflow":  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        return tf.concat([array[:idx], val[None], array[idx + 1 :]], 0)

    array[idx] = val
    return array
