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
"""Utility functions"""

# pylint: disable=wrong-import-order
import autoray as ar
import numpy as _np
import scipy as sp

# pylint: disable=import-outside-toplevel
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np

from pennylane import math


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
        and a boolean will be returned, indicating if all elements evaluate to ``True``. Otherwise,
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


def _allclose_sparse(a, b, rtol=1e-05, atol=1e-08):
    """Compare two sparse matrices for approximate equality.

    Args:
        a, b: scipy sparse matrices to compare
        rtol (float): relative tolerance
        atol (float): absolute tolerance

    Returns:
        bool: True if matrices are approximately equal
    """
    if (a != b).nnz == 0:
        return True

    diff = abs(a - b)

    # Handle cases where the matrix might be empty
    max_diff = diff.data.max() if diff.nnz > 0 else 0
    max_b = abs(b).data.max() if b.nnz > 0 else 0

    return max_diff <= atol + rtol * max_b


def _mixed_shape_match(a, b):
    """Check if the shapes of two matrices of mixed types are compatible for comparison.

    Args:
        a, b: matrices to compare

    Returns:
        bool: True if the shapes are compatible
    """
    a_shapes = a.shape
    b_shapes = b.shape
    # Take the product, if inequal then false
    if np.prod(a_shapes) != np.prod(b_shapes):
        return False
    # Make the sets of shapes, and ignore '1'
    a_shape_set = set(a_shapes) - {1}
    b_shape_set = set(b_shapes) - {1}
    if len(a_shape_set) == len(b_shape_set) == 1:
        return True  # For intrinsic one-dimensional arrays
    return a_shapes == b_shapes


def _allclose_mixed(a, b, rtol=1e-05, atol=1e-08, b_is_sparse=True):
    """Helper function for comparing dense and sparse matrices with correct tolerance reference.

    Args:
        a: first matrix (dense or sparse)
        b: second matrix (sparse or dense)
        rtol: relative tolerance
        atol: absolute tolerance
        b_is_sparse: True if b is sparse matrix, False if a is sparse matrix

    Returns:
        bool: True if matrices are approximately equal
    """
    sparse = b if b_is_sparse else a
    dense = a if b_is_sparse else b

    if sparse.nnz == 0:
        return np.allclose(dense, 0, rtol=rtol, atol=atol)

    if not _mixed_shape_match(dense, sparse):
        return False

    SIZE_THRESHOLD = 10000
    if np.prod(dense.shape) < SIZE_THRESHOLD:
        # Use dense comparison but maintain b as reference
        if b_is_sparse:
            return np.allclose(a, sparse.toarray(), rtol=rtol, atol=atol)
        return np.allclose(sparse.toarray(), b, rtol=rtol, atol=atol)

    dense_coords = dense.nonzero()
    sparse_coords = sparse.nonzero()

    coord_diff = set(zip(*dense_coords, strict=True)) ^ set(zip(*sparse_coords, strict=True))
    if coord_diff:
        return False

    # Maintain asymmetric comparison with correct reference
    if b_is_sparse:
        a_data = dense[dense_coords]
        b_data = sparse.data
    else:
        a_data = sparse.data
        b_data = dense[sparse_coords]
    return np.allclose(a_data, b_data, rtol=rtol, atol=atol)


def _allclose_sparse_scalar(sparse_mat, scalar, rtol=1e-05, atol=1e-08):
    """Compare a sparse matrix to a scalar value.

    This function checks if a sparse matrix is approximately equal to a scalar value
    by considering three cases:
    1. Empty/zero sparse matrix compared to any scalar -> scalar must be close to zero
    2. Sparse matrix with non-zero values compared to a scalar -> all non-zero values must match
    3. Following case 2, if scalar is zero, any sparsity pattern is allowed
    4. If the scalar is non-zero, all elements must be non-zero, e.g. size match nnz


    Args:
        sparse_mat: A scipy sparse matrix
        scalar: A scalar value to compare against
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if the sparse matrix is approximately equal to the scalar
    """
    # Case 1: Empty sparse matrix - only close to zero scalar
    if sparse_mat.nnz == 0:
        return np.isclose(scalar, 0, rtol=rtol, atol=atol)

    # Case 2: Sparse matrix with all non-zero values matching scalar
    # Check if all non-zeros in the sparse matrix match the scalar
    if not np.allclose(sparse_mat.data, scalar, rtol=rtol, atol=atol):
        return False

    # Note that from this step all the data already close to scalar
    # Case 3: Special handling for scalar = 0 or fully populated sparse matrix
    # If scalar is approximately zero, allow any sparsity pattern
    if np.isclose(scalar, 0, rtol=rtol, atol=atol):
        return True

    # If scalar is non-zero, all elements must be non-zero
    # Use size property for efficiency with very large matrices
    return sparse_mat.nnz == np.prod(sparse_mat.shape)


def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    """Wrapper around np.allclose, allowing tensors ``a`` and ``b``
    to differ in type"""
    try:
        # Some frameworks may provide their own allclose implementation.
        # Try and use it if available.
        if sp.sparse.issparse(a) and sp.sparse.issparse(b):
            return _allclose_sparse(a, b, rtol=rtol, atol=atol)
        # Add handling for sparse matrix compared with scalar
        if sp.sparse.issparse(a) and np.isscalar(b):
            return _allclose_sparse_scalar(a, b, rtol=rtol, atol=atol)
        if sp.sparse.issparse(b) and np.isscalar(a):
            return _allclose_sparse_scalar(b, a, rtol=rtol, atol=atol)

        if sp.sparse.issparse(a):

            return _allclose_mixed(a, b, rtol=rtol, atol=atol, b_is_sparse=False)
        if sp.sparse.issparse(b):
            return _allclose_mixed(a, b, rtol=rtol, atol=atol, b_is_sparse=True)
        res = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
    except (TypeError, AttributeError, ImportError, RuntimeError):
        # Otherwise, convert the input to NumPy arrays.
        #
        # TODO: replace this with a bespoke, framework agnostic
        # low-level implementation to avoid the NumPy conversion:
        #
        #    np.abs(a - b) <= atol + rtol * np.abs(b)
        #
        t1 = ar.to_numpy(a)
        t2 = ar.to_numpy(b)
        res = np.allclose(t1, t2, rtol=rtol, atol=atol, **kwargs)

    return res


allclose.__doc__ = _np.allclose.__doc__


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
    if isinstance(tensor, (list, tuple, int, float, complex)):
        tensor = np.asarray(tensor)

    if not isinstance(dtype, str):
        try:
            dtype = np.dtype(dtype).name
        except (AttributeError, TypeError, ImportError):
            dtype = getattr(dtype, "name", dtype)

    return ar.astype(tensor, ar.to_backend_dtype(dtype, like=ar.infer_backend(tensor)))


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
    >>> cast_like(x, y)
    tensor([1., 2.])
    """
    if isinstance(tensor2, tuple) and len(tensor2) > 0:
        tensor2 = tensor2[0]
    if isinstance(tensor2, ArrayBox):
        dtype = ar.to_numpy(tensor2._value).dtype.type  # pylint: disable=protected-access
    elif not is_abstract(tensor2):
        dtype = ar.to_numpy(tensor2).dtype.type
    else:
        dtype = tensor2.dtype
    return cast(tensor1, dtype)


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
    >>> convert_like(x, y)
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 2])>
    """
    interface = math.get_interface(tensor2)

    if interface == "torch":
        dev = tensor2.device
        return np.asarray(tensor1, device=dev, like=interface)

    if interface == "scipy":
        return sp.sparse.csr_matrix(tensor1)

    return np.asarray(tensor1, like=interface)


def is_abstract(tensor, like=None):
    """Returns True if the tensor is considered abstract.

    Abstract arrays have no internal value, and are used primarily when
    tracing Python functions, for example, in order to perform just-in-time
    (JIT) compilation.

    Abstract tensors most commonly occur within a function that has been
    decorated using ``@tf.function`` or ``@jax.jit``.

    .. note::

        Currently Autograd tensors and Torch tensors will always return ``False``.
        This is because:

        - Autograd does not provide JIT compilation, and

        - ``@torch.jit.script`` is not currently compatible with QNodes.

    Args:
        tensor (tensor_like): input tensor
        like (str): The name of the interface. Will be determined automatically
            if not provided.

    Returns:
        bool: whether the tensor is abstract or not

    **Example**

    Consider the following JAX function:

    .. code-block:: python

        import jax
        from jax import numpy as jnp

        def function(x):
            print("Value:", x)
            print("Abstract:", qml.math.is_abstract(x))
            return jnp.sum(x ** 2)

    When we execute it, we see that the tensor is not abstract; it has known value:

    >>> x = jnp.array([0.5, 0.1])
    >>> function(x)
    Value: [0.5, 0.1]
    Abstract: False
    Array(0.26, dtype=float32)

    However, if we use the ``@jax.jit`` decorator, the tensor will now be abstract:

    >>> x = jnp.array([0.5, 0.1])
    >>> jax.jit(function)(x)
    Value: Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace(level=0/1)>
    Abstract: True
    Array(0.26, dtype=float32)

    Note that JAX uses an abstract *shaped* array, so although we won't be able to
    include conditionals within our function that depend on the value of the tensor,
    we *can* include conditionals that depend on the shape of the tensor.

    Similarly, consider the following TensorFlow function:

    .. code-block:: python

        import tensorflow as tf

        def function(x):
            print("Value:", x)
            print("Abstract:", qml.math.is_abstract(x))
            return tf.reduce_sum(x ** 2)

    >>> x = tf.Variable([0.5, 0.1])
    >>> function(x)
    Value: <tf.Variable 'Variable:0' shape=(2,) dtype=float32, numpy=array([0.5, 0.1], dtype=float32)>
    Abstract: False
    <tf.Tensor: shape=(), dtype=float32, numpy=0.26>

    If we apply the ``@tf.function`` decorator, the tensor will now be abstract:

    >>> tf.function(function)(x)
    Value: <tf.Variable 'Variable:0' shape=(2,) dtype=float32>
    Abstract: True
    <tf.Tensor: shape=(), dtype=float32, numpy=0.26>
    """
    interface = like or math.get_interface(tensor)

    if interface == "jax":
        import jax
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        if isinstance(
            tensor,
            (
                jax.interpreters.ad.JVPTracer,
                jax.interpreters.batching.BatchTracer,
                jax.interpreters.partial_eval.JaxprTracer,
            ),
        ):
            # Tracer objects will be used when computing gradients or applying transforms.
            # If the value of the tracer is known, jax.core.is_concrete will return True.
            # Otherwise, it will be abstract.
            return not jax.core.is_concrete(tensor)

        return isinstance(tensor, DynamicJaxprTracer)

    if (
        interface == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf
        from tensorflow.python.framework.ops import EagerTensor

        return not isinstance(tf.convert_to_tensor(tensor), EagerTensor)

    # Autograd does not have a JIT

    # QNodes do not currently support TorchScript:
    #   NotSupportedError: Compiled functions can't take variable number of arguments or
    #   use keyword-only arguments with defaults.
    return False


def import_should_record_backprop():  # pragma: no cover
    """Return should_record_backprop or an equivalent function from TensorFlow."""
    import tensorflow.python as tfpy

    if hasattr(tfpy.eager.tape, "should_record_backprop"):
        from tensorflow.python.eager.tape import should_record_backprop
    elif hasattr(tfpy.eager.tape, "should_record"):
        from tensorflow.python.eager.tape import should_record as should_record_backprop
    elif hasattr(tfpy.eager.record, "should_record_backprop"):
        from tensorflow.python.eager.record import should_record_backprop
    else:
        raise ImportError("Cannot import should_record_backprop from TensorFlow.")

    return should_record_backprop


def requires_grad(tensor, interface=None):
    """Returns True if the tensor is considered trainable.

    .. warning::

        The implementation depends on the contained tensor type, and
        may be context dependent.

        For example, Torch tensors and PennyLane tensors track trainability
        as a property of the tensor itself. TensorFlow, on the other hand,
        only tracks trainability if being watched by a gradient tape.

    Args:
        tensor (tensor_like): input tensor
        interface (str): The name of the interface. Will be determined automatically
            if not provided.

    Returns:
        bool: whether the tensor is trainable or not.

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
    interface = interface or math.get_interface(tensor)

    if (
        interface == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        should_record_backprop = import_should_record_backprop()
        return should_record_backprop([tf.convert_to_tensor(tensor)])

    if interface == "autograd":
        if isinstance(tensor, ArrayBox):
            return True

        return getattr(tensor, "requires_grad", False)

    if interface == "torch":
        return getattr(tensor, "requires_grad", False)

    if interface in {"numpy", "scipy"}:
        return False

    if interface == "jax":
        import jax

        return isinstance(tensor, jax.core.Tracer)

    raise ValueError(f"Argument {tensor} is an unknown object")


def in_backprop(tensor, interface=None):
    """Returns True if the tensor is considered to be in a backpropagation environment, it works for Autograd,
    TensorFlow and Jax. It is not only checking the differentiability of the tensor like :func:`~.requires_grad`, but
    rather checking if the gradient is actually calculated.

    Args:
        tensor (tensor_like): input tensor
        interface (str): The name of the interface. Will be determined automatically
            if not provided.

    Returns:
        bool: whether the tensor is in a backpropagation environment or not.

    **Example**

    >>> x = tf.Variable([0.6, 0.1])
    >>> requires_grad(x)
    False
    >>> with tf.GradientTape() as tape:
    ...     print(requires_grad(x))
    True

    .. seealso:: :func:`~.requires_grad`
    """
    interface = interface or math.get_interface(tensor)

    if (
        interface == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        import tensorflow as tf

        should_record_backprop = import_should_record_backprop()
        return should_record_backprop([tf.convert_to_tensor(tensor)])

    if interface == "autograd":
        return isinstance(tensor, ArrayBox)

    if interface == "jax":
        import jax

        return isinstance(tensor, jax.core.Tracer)

    if interface in {"numpy", "scipy"}:
        return False

    raise ValueError(f"Cannot determine if {tensor} is in backpropagation.")


def binary_finite_reduced_row_echelon(binary_matrix):
    r"""Returns the reduced row echelon form (RREF) of a matrix in a binary finite field :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix representation of a Hamiltonian
    Returns:
        array[int]: reduced row-echelon form of the given `binary_matrix`

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                           [1, 0, 1, 0, 0, 0, 1, 0],
    ...                           [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> qml.math.binary_finite_reduced_row_echelon(binary_matrix)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    rref_mat = binary_matrix.copy()
    shape = rref_mat.shape
    icol = 0

    for irow in range(shape[0]):
        while icol < shape[1] and not rref_mat[irow][icol]:
            # get the nonzero indices in the remainder of column icol
            non_zero_idx = rref_mat[irow:, icol].nonzero()[0]

            if len(non_zero_idx) == 0:  # if remainder of column icol is all zero
                icol += 1
            else:
                # find value and index of largest element in remainder of column icol
                krow = irow + non_zero_idx[0]

                # swap rows krow and irow
                rref_mat[irow, icol:], rref_mat[krow, icol:] = (
                    rref_mat[krow, icol:].copy(),
                    rref_mat[irow, icol:].copy(),
                )
        if icol < shape[1] and rref_mat[irow][icol]:
            # store remainder right hand side columns of the pivot row irow
            rpvt_cols = rref_mat[irow, icol:].copy()

            # get the column icol and set its irow element to 0 to avoid XORing pivot row with itself
            currcol = rref_mat[:, icol].copy()
            currcol[irow] = 0

            # XOR the right hand side of the pivot row irow with all of the other rows
            rref_mat[:, icol:] ^= np.outer(currcol, rpvt_cols)
            icol += 1

    return rref_mat.astype(int)
