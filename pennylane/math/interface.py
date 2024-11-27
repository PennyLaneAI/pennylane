# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions related to interfaces"""

import warnings
from enum import Enum
from typing import Literal

# pylint: disable=wrong-import-order
import autoray as ar

# pylint: disable=import-outside-toplevel
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np

import pennylane as qml


class Interface(Enum):
    """Standard set of interfaces."""

    AUTOGRAD = "autograd"
    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"
    JAX_JIT = "jax-jit"
    TF = "tf"
    TF_AUTOGRAPH = "tf-autograph"
    AUTO = "auto"


INTERFACE_MAP = {
    None: Interface.NUMPY,
    "auto": Interface.AUTO,
    "autograd": Interface.AUTOGRAD,
    "numpy": Interface.NUMPY,
    "scipy": Interface.NUMPY,
    "jax": Interface.JAX,
    "jax-jit": Interface.JAX_JIT,
    "jax-python": Interface.JAX,
    "JAX": Interface.JAX,
    "torch": Interface.TORCH,
    "pytorch": Interface.TORCH,
    "tf": Interface.TF,
    "tensorflow": Interface.TF,
    "tensorflow-autograph": Interface.TF_AUTOGRAPH,
    "tf-autograph": Interface.TF_AUTOGRAPH,
}
"""dict[str, str]: maps an allowed interface specification to its canonical name."""

SupportedInterfaceUserInput = Literal[tuple(INTERFACE_MAP.keys())]

SUPPORTED_INTERFACE_NAMES = list(Interface)
"""list[str]: allowed interface strings"""

jpc_interfaces = {
    Interface.AUTOGRAD,
    Interface.NUMPY,
    Interface.TORCH,
    Interface.JAX,
    Interface.JAX_JIT,
    Interface.TF,
}


def get_canonical_interface(user_input: str | None) -> Interface:
    """Retrieve the canonical Interface based on user input."""
    try:
        if user_input in SUPPORTED_INTERFACE_NAMES:
            return user_input
        return INTERFACE_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Unknown interface {user_input}. Interface must be one of {SUPPORTED_INTERFACE_NAMES}."
        ) from exc


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
    interface = get_interface(tensor2)

    if interface == "torch":
        dev = tensor2.device
        return np.asarray(tensor1, device=dev, like=interface)

    return np.asarray(tensor1, like=interface)


def get_interface(*values):
    """Determines the correct framework to dispatch to given a tensor-like object or a
    sequence of tensor-like objects.

    Args:
        *values (tensor_like): variable length argument list with single tensor-like objects

    Returns:
        str: the name of the interface

    To determine the framework to dispatch to, the following rules
    are applied:

    * Tensors that are incompatible (such as Torch, TensorFlow and Jax tensors)
      cannot both be present.

    * Autograd tensors *may* be present alongside Torch, TensorFlow and Jax tensors,
      but Torch, TensorFlow and Jax take precedence; the autograd arrays will
      be treated as non-differentiable NumPy arrays. A warning will be raised
      suggesting that vanilla NumPy be used instead.

    * Vanilla NumPy arrays and SciPy sparse matrices can be used alongside other tensor objects;
      they will always be treated as non-differentiable constants.

    .. warning::
        ``get_interface`` defaults to ``"numpy"`` whenever Python built-in objects are passed.
        I.e. a list or tuple of ``torch`` tensors will be identified as ``"numpy"``:

        >>> get_interface([torch.tensor([1]), torch.tensor([1])])
        "numpy"

        The correct usage in that case is to unpack the arguments ``get_interface(*[torch.tensor([1]), torch.tensor([1])])``.

    """

    if len(values) == 1:
        return _get_interface_of_single_tensor(values[0])

    interfaces = {_get_interface_of_single_tensor(v) for v in values}

    if len(interfaces - {"numpy", "scipy", "autograd"}) > 1:
        # contains multiple non-autograd interfaces
        raise ValueError("Tensors contain mixed types; cannot determine dispatch library")

    non_numpy_scipy_interfaces = set(interfaces) - {"numpy", "scipy"}

    if len(non_numpy_scipy_interfaces) > 1:
        # contains autograd and another interface
        warnings.warn(
            f"Contains tensors of types {non_numpy_scipy_interfaces}; dispatch will prioritize "
            "TensorFlow, PyTorch, and Jax over Autograd. Consider replacing Autograd with vanilla NumPy.",
            UserWarning,
        )

    if "tensorflow" in interfaces:
        return "tensorflow"

    if "torch" in interfaces:
        return "torch"

    if "jax" in interfaces:
        return "jax"

    if "autograd" in interfaces:
        return "autograd"

    return "numpy"


def _get_interface_of_single_tensor(tensor):
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


def get_deep_interface(value):
    """
    Given a deep data structure with interface-specific scalars at the bottom, return their
    interface name.

    Args:
        value (list, tuple): A deep list-of-lists, tuple-of-tuples, or combination with
            interface-specific data hidden within it

    Returns:
        str: The name of the interface deep within the value

    **Example**

    >>> x = [[jax.numpy.array(1), jax.numpy.array(2)], [jax.numpy.array(3), jax.numpy.array(4)]]
    >>> get_deep_interface(x)
    'jax'

    This can be especially useful when converting to the appropriate interface:

    >>> qml.math.asarray(x, like=qml.math.get_deep_interface(x))
    Array([[1, 2],
           [3, 4]], dtype=int64)

    """
    itr = value
    while isinstance(itr, (list, tuple)):
        if len(itr) == 0:
            return "numpy"
        itr = itr[0]
    return _get_interface_of_single_tensor(itr)


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
    interface = like or get_interface(tensor)

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
            # If the value of the tracer is known, it will contain a ConcreteArray.
            # Otherwise, it will be abstract.
            return not isinstance(tensor.aval, jax.core.ConcreteArray)

        return isinstance(tensor, DynamicJaxprTracer)

    if interface == "tensorflow":
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
    interface = interface or get_interface(tensor)

    if interface == "tensorflow":
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
    interface = interface or get_interface(tensor)

    if interface == "tensorflow":
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


# pylint: disable=import-outside-toplevel
def _use_tensorflow_autograph():
    """Checks if TensorFlow is in graph mode, allowing Autograph for optimized execution"""
    try:  # pragma: no cover
        import tensorflow as tf
    except ImportError as e:  # pragma: no cover
        raise qml.QuantumFunctionError(  # pragma: no cover
            "tensorflow not found. Please install the latest "  # pragma: no cover
            "version of tensorflow supported by Pennylane "  # pragma: no cover
            "to enable the 'tensorflow' interface."  # pragma: no cover
        ) from e  # pragma: no cover

    return not tf.executing_eagerly()


def _get_interface_name(tapes, interface):
    """Helper function to get the interface name of a list of tapes

    Args:
        tapes (list[.QuantumScript]): Quantum tapes
        interface (Optional[str]): Original interface to use as reference.

    Returns:
        str: Interface name"""

    interface = get_canonical_interface(interface)

    if interface == Interface.AUTO:
        params = []
        for tape in tapes:
            params.extend(tape.get_parameters(trainable_only=False))
        interface = get_interface(*params)
        if interface != Interface.NUMPY:
            interface = INTERFACE_MAP.get(interface, None)
    if interface == Interface.TF and _use_tensorflow_autograph():
        interface = Interface.TF_AUTOGRAPH
    if interface == Interface.JAX:
        interface = get_jax_interface_name(tapes)

    return interface


def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t:
            # Unwrap the observable from a MeasurementProcess
            if not isinstance(op, qml.ops.Prod):
                op = op.obs if hasattr(op, "obs") else op
            if op is not None:
                # Some MeasurementProcess objects have op.obs=None
                for param in op.data:
                    if is_abstract(param):
                        return Interface.JAX_JIT

    return Interface.JAX
