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
"""Function wrappers for the TensorBox API"""
# pylint:disable=abstract-class-instantiated,unexpected-keyword-arg
from collections.abc import Sequence
import itertools
import warnings

import numpy as np

from .tensorbox import TensorBox


def _get_multi_tensorbox(values):
    """Determines the correct framework to dispatch to given a
    sequence of tensor-like objects.

    Args:
        values (Sequence[tensor_like]): a sequence of tensor like objects

    Returns:
        .TensorBox: A TensorBox that will dispatch to the correct framework
        given the rules of precedence. This TensorBox will contain the *first*
        tensor-like object in ``values`` that corresponds to the highest-priority
        framework.

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

    if "tf" in interfaces:
        return TensorBox(values[interfaces.index("tf")])

    if "torch" in interfaces:
        return TensorBox(values[interfaces.index("torch")])

    if "autograd" in interfaces:
        return TensorBox(values[interfaces.index("autograd")])

    if "jax" in interfaces:
        return TensorBox(values[interfaces.index("jax")])

    return TensorBox(values[interfaces.index("numpy")])


def abs_(tensor):
    """Returns the element-wise absolute value.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        tensor_like:

    **Example**

    >>> a = torch.tensor([1., -2.], requires_grad=True)
    >>> abs(a)
    tensor([1., 2.], grad_fn=<AbsBackward>)
    """
    return TensorBox(tensor).abs(wrap_output=False)


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
    t1 = toarray(tensor1)
    t2 = toarray(tensor2)
    return np.all(t1 == t2, **kwargs)


def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    """Wrapper around np.allclose, allowing tensors ``a`` and ``b``
    to differ in type"""
    t1 = toarray(a)
    t2 = toarray(b)
    return np.allclose(t1, t2, rtol=rtol, atol=atol, **kwargs)


allclose.__doc__ = np.allclose.__doc__


def angle(tensor):
    """Returns the element-wise angle of a complex tensor.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        tensor_like:

    **Example**

    >>> a = torch.tensor([1.0, 1.0j, 1+1j], requires_grad=True)
    >>> angle(a)
    tensor([0.0000, 1.5708, 0.7854], grad_fn=<AngleBackward>)
    """
    return TensorBox(tensor).angle(wrap_output=False)


def arcsin(tensor):
    """Returns the element-wise inverse sine of the tensor"""
    return TensorBox(tensor).arcsin(wrap_output=False)


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
    return _get_multi_tensorbox(values).block_diag(values, wrap_output=False)


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
    return TensorBox(tensor).cast(dtype, wrap_output=False)


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
    dtype = toarray(tensor2).dtype.type
    return cast(tensor1, dtype)


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
    TensorBox: <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([6.00e-01, 1.00e-01, 6.00e-01, 1.00e-01, 2.00e-01, 3.00e-01, 5.00e+00, 8.00e+00, 1.01e+02], dtype=float32)>
    """
    return _get_multi_tensorbox(values).concatenate(values, axis=axis, wrap_output=False)


def conj(tensor):
    """Conjugate a tensor. Negate the imaginary part of a complex value.

    Args:
        tensor (tensor_like): A tensor-like object to conjugate.

    Returns:
        tensor_like: The conjugated tensor.

    **Example**

    >>> x = tf.constant([0.6 + 0.1j, 0.1 - 0.3j, 0.6])
    >>> conj(x)
    <tf.Tensor: shape=(3,), dtype=complex64, numpy=array([6.00e-01 + 1.00e-1j, 1.00e-01 + 3.00e-1j, 6.00e-01 + 0.00j], dtype=complex64)>
    """
    return TensorBox(tensor).conj(wrap_output=False)


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
        l = cast(o.eigvals, dtype=np.float64)
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

        l1 = cast(o1.eigvals, dtype=np.float64)
        l2 = cast(o2.eigvals, dtype=np.float64)
        l12 = cast(np.kron(l1, l2), dtype=np.float64)

        p1 = marginal_prob(prob, o1wires)
        p2 = marginal_prob(prob, o2wires)
        p12 = marginal_prob(prob, shared_wires)

        res = dot(l12, p12) - dot(l1, p1) * dot(l2, p2)

        cov = scatter_element_add(cov, [i, j], res)
        cov = scatter_element_add(cov, [j, i], res)

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
    return TensorBox(tensor2).astensor(tensor1)


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
    >>> qml.diag(z, k=1)
    >>> qml.math.diag(z, k=1)
    tensor([[0.0000, 0.1000, 0.0000],
            [0.0000, 0.0000, 0.2000],
            [0.0000, 0.0000, 0.0000]])
    """
    if isinstance(values, Sequence):
        return _get_multi_tensorbox(values).diag(values, k=k, wrap_output=False)

    return TensorBox(values).diag(values, k=k, wrap_output=False)


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
    return _get_multi_tensorbox([tensor1, tensor2]).dot(tensor1, tensor2, wrap_output=False)


def expand_dims(tensor, axis):
    """Expand the shape of an array by adding a new dimension of size 1
    at the specified axis location.

    .. warning::

        This function differs from ``np.expand_dims``.

    Args:
        tensor (tensor_like): tensor to expand
        axis (int): location in the axes to place the new dimension

    Returns:
        tensor_like: a tensor with the expanded shape

    **Example**

    >>> x = tf.Variable([3, 4])
    >>> expand_dims(x, axis=1)
    <tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    array([[3],
           [4]], dtype=int32)>
    """
    return TensorBox(tensor).expand_dims(axis, wrap_output=False)


def flatten(tensor):
    """Flattens an N-dimensional tensor to a 1-dimensional tensor.

    Args:
        tensor (tensor_like): tensor to flatten

    Returns:
        tensor_like: the flattened tensor

    **Example**

    >>> x = tf.Variable([[1, 3], [2, 4]])
    >>> flatten(x)
    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 3, 2, 4], dtype=int32)>
    """
    return reshape(tensor, (-1,))


def gather(tensor, indices):
    """Gather tensor values given a tuple of indices.

    This is equivalent to the following NumPy fancy indexing:

    ..code-block:: python

        tensor[array(indices)]

    Args:
        tensor (tensor_like): tensor to gather from
        indices (Sequence[int]): the indices of the values to extract

    Returns:

        tensor_like: the gathered tensor values

    .. seealso::

        :func:`~.take`
    """
    return TensorBox(tensor).gather(np.array(indices), wrap_output=False)


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
    return TensorBox(tensor).interface


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
    prob = flatten(prob)
    num_wires = int(np.log2(len(prob)))

    if num_wires == len(axis):
        return prob

    inactive_wires = tuple(set(range(num_wires)) - set(axis))
    prob = reshape(prob, [2] * num_wires)
    prob = sum_(prob, axis=inactive_wires)
    return flatten(prob)


def toarray(tensor):
    """Returns the tensor as a NumPy ``ndarray``. No copying
    is performed; the tensor and the returned array share the
    same storage.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        array: a ``ndarray`` view into the same data

    **Example**

    >>> x = torch.tensor([1., 2.])
    >>> toarray(x)
    array([1, 2])
    """
    return TensorBox(tensor).numpy()


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
        return TensorBox(tensor).ones_like().cast(dtype, wrap_output=False)

    return TensorBox(tensor).ones_like(wrap_output=False)


def reshape(tensor, shape):  # pylint: disable=redefined-outer-name
    """Gives a new shape to a tensor without changing its data.

    Args:
        tensor (tensor_like): input tensor
        shape (tuple[int]): The new shape. The special value of -1 indicates
            that the size of that dimension is computed so that the total size
            remains constant. A dimension of -1 can only be specified once.

    Returns:
        tensor_like: a new view into the input tensor with
        shape ``shape``

    **Example**

    >>> a = tf.range(4.)
    >>> reshape(a, (2, 2))
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0., 1.],
           [2., 3.]], dtype=float32)>
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([0, 1, 2, 3])
    """
    return TensorBox(tensor).reshape(shape, wrap_output=False)


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
    return TensorBox(tensor).requires_grad


def scatter_element_add(tensor, index, value):
    """Adds a scalar value to a specific index of a tensor.

    This is a pure equivalent of ``tensor[index] += value``.

    Args:
        tensor (tensor_like): the input tensor to be updated
        index (tuple[int]): the index of the input tensor to update
        value (scalar): the scalar value to add to the tensor element

    Returns:
        tensor_like: the output tensor

    **Example**

    >>> x = torch.ones((2, 3))
    >>> qml.math.scatter_element_add(x, [1, 2], 3)
    tensor([[1., 1., 1.],
            [1., 1., 4.]])
    """
    value = convert_like(value, tensor)
    return TensorBox(tensor).scatter_element_add(index, value, wrap_output=False)


def shape(tensor):
    """Returns the shape of the tensor.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        tuple[int]: shape of the tensor

    **Example**

    >>> x = tf.constant([[0.6, 0.1, 0.6], [1., 2., 3.]])
    >>> shape(x)
    (2, 3)
    """
    return TensorBox(tensor).shape


def sqrt(tensor):
    """Returns the element-wise square root.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        tensor_like:

    **Example**

    >>> a = torch.tensor([4., 9.], requires_grad=True)
    >>> sqrt(a)
    tensor([2., 3.], grad_fn=<SqrtBackward>)
    """
    return TensorBox(tensor).sqrt(wrap_output=False)


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
    return _get_multi_tensorbox(values).stack(values, axis=axis, wrap_output=False)


def squeeze(tensor):
    """Remove single-dimensional entries from the shape of an array.

    Args:
        tensor (tensor_like): A tensor-like object.

    Returns:
        The input array, but with all or a subset of the dimensions of length 1 removed.
        This is always a itself or a view into a. Note that if all axes are squeezed,
        the result is a 0d array and not a scalar.

    **Example**

    >>> x = torch.ones((2, 1, 3, 4, 1))
    >>> y = squeeze(x)
    >>> y.shape
    (2, 3, 4)
    """
    return TensorBox(tensor).squeeze(wrap_output=False)


def sum_(tensor, axis=None, keepdims=False):
    """TensorBox: Returns the sum of the tensor elements across the specified dimensions.

    Args:
        tensor (tensor_like): input tensor
        axis (int or tuple[int]): The axis or axes along which to perform the sum.
            If not specified, all elements of the tensor across all dimensions
            will be summed, returning a tensor.
        keepdims (bool): If True, retains all summed dimensions.

    Returns:
        tensor_like: The tensor with specified dimensions summed over. Note that
        if all elements are summed, then a 0-dimensional tensor is returned, rather
        than a Python scalar.

    **Example**

    Summing over all dimensions:

    >>> x = tf.Variable([[1., 2.], [3., 4.]])
    >>> sum(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=10.0>

    Summing over specified dimensions:

    >>> x = np.array([[[1, 1], [5, 3]], [[1, 4], [-6, -1]]])
    >>> x.shape
    (2, 2, 2)
    >>> sum(x, axis=(0, 2))
    tensor([7, 1], requires_grad=True)
    >>> sum(x, axis=(0, 2), keepdims=True)
    tensor([[[7],
             [1]]], requires_grad=True)
    """
    return TensorBox(tensor).sum(axis=axis, keepdims=keepdims, wrap_output=False)


def T(tensor):
    """Returns the transpose of the tensor by reversing the order
    of the axes. For a 2D tensor, this corresponds to the matrix transpose.

    Args:
        tensor (tensor_like): input tensor

    Returns:
        tensor_like: input tensor with axes reversed

    **Example**

    >>> x = tf.Variable([[1, 2], [3, 4]])
    >>> T(x)
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 3],
           [2, 4]], dtype=int32)>
    """
    return TensorBox(tensor).T(wrap_output=False)


def take(tensor, indices, axis=None):
    """Gather elements from a tensor.

    Note that ``take(indices, axis=3)`` is equivalent
    to ``tensor[:, :, :, indices, ...]`` for frameworks that support
    NumPy-like fancy indexing.

    This function is roughly equivalent to ``np.take`` and ``tf.gather``.
    In the case of a 1-dimensional set of indices, it is roughly equivalent
    to ``torch.index_select``, but deviates for multi-dimensional indices.

    Args:
        tensor (tensor_like): input tensor
        indices (Sequence[int]): the indices of the values to extract
        axis: The axis over which to select the values. If not provided,
            the tensor is flattened before value extraction.

    **Example**

    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> take(y, indices=[[0, 0], [1, 0]], axis=1)
    tensor([[[1, 1],
             [2, 1]],

            [[3, 3],
             [4, 3]]])
    """
    return TensorBox(tensor).take(indices, axis=axis, wrap_output=False)


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
    return _get_multi_tensorbox([x, y]).where(condition, x, y, wrap_output=False)
