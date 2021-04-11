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
import itertools
import warnings

import autoray as ar
from autoray import numpy as np
import numpy as _np


from . import registrations


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

    * Vanilla NumPy arrays can be used alongside other tensor objects; they will
      always be treated as non-differentiable constants.
    """
    if "resource_variable" in getattr(values, "__module__", tuple()):
        values = np.asarray(values)

    interfaces = [get_interface(v) for v in values]

    if len(set(interfaces) - {"numpy", "autograd"}) > 1:
        # contains multiple non-autograd interfaces
        raise ValueError("Tensors contain mixed types; cannot determine dispatch library")

    non_numpy_interfaces = set(interfaces) - {"numpy"}

    if len(non_numpy_interfaces) > 1:
        # contains autograd and another interface
        warnings.warn(
            f"Contains tensors of types {non_numpy_interfaces}; dispatch will prioritize "
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


def allequal(tensor1, tensor2, **kwargs):
    """Returns True if two tensors are element-wise equal along a given axis.

    This function is equivalent to calling ``np.all(tensor1 == tensor2, **kwargs)``,
    but allows for ``tensor1`` and ``tensor2`` to differ in type.

    Args:
        tensor1 (tensor_like): tensor to compare
        tensor2 (tensor_like): tensor to compare
        **kwargs: Accepts any keyword argument that is accepted by ``np.all``,
            such as ``axis``, ``out``, and ``keepdims``. See the `NumPy documentation
            <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`__ for
            more details.

    Returns:
        ndarray, bool: If ``axis=None``, a logical AND reduction is applied to all elements
        and a boolean will be returned, indicating if all elements evaluate to True. Otherwise,
        a boolean NumPy array will be returned.

    **Example**

    >>> a = torch.tensor([1, 2])
    >>> b = np.array([1, 2])
    >>> allequal(a, b)
    True
    """
    t1 = ar.to_numpy(tensor1)
    t2 = ar.to_numpy(tensor2)
    return np.all(t1 == t2, **kwargs)


def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    """Wrapper around np.allclose, allowing tensors ``a`` and ``b``
    to differ in type"""
    t1 = ar.to_numpy(a)
    t2 = ar.to_numpy(b)
    return np.allclose(t1, t2, rtol=rtol, atol=atol, **kwargs)


allclose.__doc__ = np.allclose.__doc__


def block_diag(values):
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
    interface = _multi_dispatch(values)

    if isinstance(values, (list, tuple)):
        values = np.coerce(values, like=interface)
        return np.block_diag(values, like=interface)

    return np.block_diag(values, like=interface)


def cast(tensor, dtype):
    """Casts the given tensor to a new type.

    Args:
        tensor (tensor_like): tensor to cast
        dtype (str, np.dtype): Any supported NumPy dtype representation; this can be
            a string (``"float64"``), a ``np.dtype`` object (``np.dtype("float64")``), or
            a dtype class (``np.float64``). If ``tensor`` is not a NumPy array, the
            **equivalent** dtype in the dispatched framework is used.

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor`` and the
        same dtype as ``dtype``

    **Example**

    We can use NumPy dtype specifiers:

    >>> x = torch.tensor([1, 2])
    >>> cast(x, np.float64)
    tensor([1., 2.], dtype=torch.float64)

    We can also use strings:

    >>> x = tf.Variable([1, 2])
    >>> cast(x, "complex128")
    <tf.Tensor: shape=(2,), dtype=complex128, numpy=array([1.+0.j, 2.+0.j])>
    """
    if isinstance(tensor, (list, tuple)):
        tensor = np.asarray(tensor)

    if not isinstance(dtype, str):
        try:
            dtype = np.dtype(dtype).name
        except (AttributeError, TypeError):
            try:
                dtype = dtype.name
            except AttributeError:
                pass

    return ar.astype(tensor, ar.to_backend_dtype(dtype, like=ar.infer_backend(tensor)))


def concatenate(values, axis=0):
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
    interface = _multi_dispatch(values)

    if interface == "torch":
        import torch

        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            values = [np.flatten(torch.as_tensor(t)) for t in values]
            axis = 0
        else:
            values = [torch.as_tensor(t) for t in values]

    if interface == "tensorflow" and axis is None:
        # flatten and then concatenate zero'th dimension
        # to reproduce numpy's behaviour
        values = [np.flatten(np.array(t)) for t in values]
        axis = 0

    return np.concatenate(values, axis=axis, like=interface)


def cast_like(tensor1, tensor2):
    """Casts a tensor to the same dtype as another.

    Args:
        tensor1 (tensor_like): tensor to cast
        tensor2 (tensor_like): tensor with corresponding dtype to cast to

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor1`` and the
        same dtype as ``tensor2``

    **Example**

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3., 4.])
    >>> cast(x, y)
    tensor([1., 2.])
    """
    dtype = ar.to_numpy(tensor2).dtype.type
    return cast(tensor1, dtype)


def cov_matrix(prob, obs, wires=None, diag_approx=False):
    """Calculate the covariance matrix of a list of commuting observables, given
    the joint probability distribution of the system in the shared eigenbasis.

    .. note::
        This method only works for **commuting observables.**
        If the probability distribution is the result of a quantum circuit,
        the quantum state must be rotated into the shared
        eigenbasis of the list of observables before measurement.

    Args:
        prob (tensor_like): probability distribution
        obs (list[.Observable]): a list of observables for which
            to compute the covariance matrix for
        diag_approx (bool): if True, return the diagonal approximation
        wires (.Wires): The wire register of the system. If not provided,
            it is assumed that the wires are labelled with consecutive integers.

    Returns:
        tensor_like: the covariance matrix of size ``(len(obs), len(obs))``

    **Example**

    Consider the following ansatz and observable list:

    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(2)]
    >>> ansatz = qml.templates.StronglyEntanglingLayers

    We can construct a QNode to output the probability distribution in the shared eigenbasis of the
    observables:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            ansatz(weights, wires=[0, 1, 2])
            # rotate into the basis of the observables
            for o in obs_list:
                o.diagonalizing_gates()
            return qml.probs(wires=[0, 1, 2])

    We can now compute the covariance matrix:

    >>> weights = qml.init.strong_ent_layers_normal(n_layers=2, n_wires=3)
    >>> cov = qml.math.cov_matrix(circuit(weights), obs_list)
    >>> cov
    array([[0.98707611, 0.03665537],
         [0.03665537, 0.99998377]])

    Autodifferentiation is fully supported using all interfaces.
    Here we use autograd:

    >>> cost_fn = lambda weights: qml.math.cov_matrix(circuit(weights), obs_list)[0, 1]
    >>> qml.grad(cost_fn)(weights)[0]
    array([[[ 4.94240914e-17, -2.33786398e-01, -1.54193959e-01],
            [-3.05414996e-17,  8.40072236e-04,  5.57884080e-04],
            [ 3.01859411e-17,  8.60411436e-03,  6.15745204e-04]],
           [[ 6.80309533e-04, -1.23162742e-03,  1.08729813e-03],
            [-1.53863193e-01, -1.38700657e-02, -1.36243323e-01],
            [-1.54665054e-01, -1.89018172e-02, -1.56415558e-01]]])
    """
    variances = []

    # diagonal variances
    for i, o in enumerate(obs):
        l = cast(o.eigvals, dtype=_np.float64)
        w = o.wires.labels if wires is None else wires.indices(o.wires)
        p = marginal_prob(prob, w)

        res = dot(l ** 2, p) - (dot(l, p)) ** 2
        variances.append(res)

    cov = diag(variances)

    if diag_approx:
        return cov

    for i, j in itertools.combinations(range(len(obs)), r=2):
        o1 = obs[i]
        o2 = obs[j]

        o1wires = o1.wires.labels if wires is None else wires.indices(o1.wires)
        o2wires = o2.wires.labels if wires is None else wires.indices(o2.wires)
        shared_wires = set(o1wires + o2wires)

        l1 = cast(o1.eigvals, dtype=_np.float64)
        l2 = cast(o2.eigvals, dtype=_np.float64)
        l12 = cast(_np.kron(l1, l2), dtype=_np.float64)

        p1 = marginal_prob(prob, o1wires)
        p2 = marginal_prob(prob, o2wires)
        p12 = marginal_prob(prob, shared_wires)

        res = dot(l12, p12) - dot(l1, p1) * dot(l2, p2)

        cov = np.scatter_element_add(cov, [i, j], res)
        cov = np.scatter_element_add(cov, [j, i], res)

    return cov


def convert_like(tensor1, tensor2):
    """Convert a tensor to the same type as another.

    Args:
        tensor1 (tensor_like): tensor to convert
        tensor2 (tensor_like): tensor with corresponding type to convert to

    Returns:
        tensor_like: a tensor with the same shape, values, and dtype as ``tensor1`` and the
        same type as ``tensor2``.

    **Example**

    >>> x = np.array([1, 2])
    >>> y = tf.Variable([3, 4])
    >>> cast(x, y)
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 2])>
    """
    return np.asarray(tensor1, like=get_interface(tensor2))


def diag(values, k=0):
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
    >>> diag(x)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]], dtype=float32)>
    >>> y = tf.Variable([0.65, 0.2, 0.1])
    >>> diag(y, k=-1)
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
    interface = _multi_dispatch(values)

    if isinstance(values, (list, tuple)):
        values = np.stack(np.coerce(values, like=interface), like=interface)

    return np.diag(values, k=k, like=interface)


def dot(tensor1, tensor2):
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
    """
    interface = _multi_dispatch([tensor1, tensor2])
    x, y = np.coerce([tensor1, tensor2], like=interface)

    if interface == "torch":
        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return np.tensordot(x, y, dims=[[-1], [-2]], like=interface)

    if interface == "tensorflow":
        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if y.ndim == 1:
            return np.tensordot(x, y, axes=[[-1], [0]], like=interface)

        if x.ndim == 2 and y.ndim == 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=interface)

    return np.dot(x, y, like=interface)


def get_interface(tensor):
    """Returns the name of the package that any array/tensor manipulations
    will dispatch to. The returned strings correspond to those used for PennyLane
    :doc:`interfaces </introduction/interfaces>`.

    Args:
        tensor (tensor_like): tensor input

    Returns:
        str: name of the interface

    **Example**

    >>> x = torch.tensor([1., 2.])
    >>> get_interface(x)
    'torch'
    >>> from pennylane import numpy as np
    >>> x = np.array([4, 5], requires_grad=True)
    >>> get_interface(x)
    'autograd'
    """
    namespace = tensor.__class__.__module__.split(".")[0]

    if namespace in ("pennylane", "autograd"):
        return "autograd"

    res = ar.infer_backend(tensor)

    if res == "builtins":
        return "numpy"

    return res


def marginal_prob(prob, axis):
    """Compute the marginal probability given a joint probability distribution expressed as a tensor.
    Each random variable corresponds to a dimension.

    If the distribution arises from a quantum circuit measured in computational basis, each dimension
    corresponds to a wire. For example, for a 2-qubit quantum circuit `prob[0, 1]` is the probability of measuring the
    first qubit in state 0 and the second in state 1.

    Args:
        prob (tensor_like): 1D tensor of probabilities. This tensor should of size
            ``(2**N,)`` for some integer value ``N``.
        axis (list[int]): the axis for which to calculate the marginal
            probability distribution

    Returns:
        tensor_like: the marginal probabilities, of
        size ``(2**len(axis),)``

    **Example**

    >>> x = tf.Variable([1, 0, 0, 1.], dtype=tf.float64) / np.sqrt(2)
    >>> marginal_prob(x, axis=[0, 1])
    <tf.Tensor: shape=(4,), dtype=float64, numpy=array([0.70710678, 0.        , 0.        , 0.70710678])>
    >>> marginal_prob(x, axis=[0])
    <tf.Tensor: shape=(2,), dtype=float64, numpy=array([0.70710678, 0.70710678])>
    """
    prob = np.flatten(prob)
    num_wires = int(np.log2(len(prob)))

    if num_wires == len(axis):
        return prob

    inactive_wires = tuple(set(range(num_wires)) - set(axis))
    prob = np.reshape(prob, [2] * num_wires)
    prob = np.sum(prob, axis=inactive_wires)
    return np.flatten(prob)


def ones_like(tensor, dtype=None):
    """Returns a tensor of all ones with the same shape and dtype
    as the input tensor.

    Args:
        tensor (tensor_like): input tensor
        dtype (str, np.dtype): The desired output datatype of the array. If not provided, the dtype of

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


def requires_grad(tensor):
    """Returns True if the tensor is considered trainable.

    .. warning::

        The implemetation depends on the contained tensor type, and
        may be context dependent.

        For example, Torch tensors and PennyLane tensors track trainability
        as a property of the tensor itself. TensorFlow, on the other hand,

        only tracks trainability if being watched by a gradient tape.

    Args:
        tensor (tensor_like): input tensor

    **Example**

    Calling this function on a PennyLane NumPy array:

    >>> x = np.array([1., 5.], requires_grad=True)
    >>> requires_grad(x)
    True
    >>> x.requires_grad = False
    >>> requires_grad(x)
    False

    PyTorch has similar behaviour.

    With TensorFlow, the output is dependent on whether the tensor
    is currently being watched by a gradient tape:

    >>> x = tf.Variable([0.6, 0.1])
    >>> requires_grad(x)
    False
    >>> with tf.GradientTape() as tape:
    ...     print(requires_grad(x))
    True

    While TensorFlow constants are by default not trainable, they can be
    manually watched by the gradient tape:

    >>> x = tf.constant([0.6, 0.1])
    >>> with tf.GradientTape() as tape:
    ...     print(requires_grad(x))
    False
    >>> with tf.GradientTape() as tape:
    ...     tape.watch([x])
    ...     print(requires_grad(x))
    True
    """
    interface = get_interface(tensor)

    if interface == "tensorflow":
        import tensorflow as tf

        try:
            from tensorflow.python.eager.tape import should_record_backprop
        except ImportError:  # pragma: no cover
            from tensorflow.python.eager.tape import should_record as should_record_backprop

        return should_record_backprop([tf.convert_to_tensor(tensor)])

    if interface in ("torch", "autograd"):
        return tensor.requires_grad

    if interface == "numpy":
        return False

    if interface == "jax":
        return True

    return False


def stack(values, axis=0):
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
    interface = _multi_dispatch(values)
    values = np.coerce(values, like=interface)
    return np.stack(values, axis=axis, like=interface)


def where(condition, x, y):
    """Returns elements chosen from x or y depending on a boolean tensor condition.

    The input tensors ``condition``, ``x``, and ``y`` must all be broadcastable to the same shape.

    Args:
        condition (tensor_like[bool]): A boolean tensor. Where True, elements from
            ``x`` will be chosen, otherwise ``y``.
        x (tensor_like): values from which to choose if the condition evaluates to True
        y (tensor_like): values from which to choose if the condition evaluates to False

    Returns:
        tensor_like: A tensor with elements from ``x`` where the condition is True, and
        ``y`` otherwise. The output tensor has the same shape as the input tensors.

    **Example**

    >>> a = torch.tensor([0.6, 0.23, 0.7, 1.5, 1.7], requires_grad=True)
    >>> b = torch.tensor([-1., -2., -3., -4., -5.], requires_grad=True)
    >>> math.where(a < 1, a, b)
    tensor([ 0.6000,  0.2300,  0.7000, -4.0000, -5.0000], grad_fn=<SWhereBackward>)
    """
    return np.where(condition, x, y, like=_multi_dispatch([x, y]))
