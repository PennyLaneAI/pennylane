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
from typing import Literal, Union

import autoray as ar

import pennylane as qml


class Interface(Enum):
    """Canonical set of interfaces supported."""

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
"""list[str]: allowed interface names that the user can input"""

SUPPORTED_INTERFACE_NAMES = list(Interface)
"""list[Interface]: allowed interface names"""

jpc_interfaces = {
    Interface.AUTOGRAD,
    Interface.NUMPY,
    Interface.TORCH,
    Interface.JAX,
    Interface.JAX_JIT,
    Interface.TF,
}


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


def _get_jax_interface_name(tapes: "qml.tape.QuantumScriptBatch") -> Interface:
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
        Interface: name of JAX interface that fits the tape parameters
    """
    for t in tapes:
        for op in t:
            # Unwrap the observable from a MeasurementProcess
            if not isinstance(op, qml.ops.Prod):
                op = getattr(op, "obs", op)
            if op is not None:
                # Some MeasurementProcess objects have op.obs=None
                if any(qml.math.is_abstract(param) for param in op.data):
                    return Interface.JAX_JIT

    return Interface.JAX


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


def _get_canonical_interface_name(user_input: Union[str, Interface]) -> Interface:
    """Helper function to get the canonical interface.

    Args:
        interface (str, Interface): reference interface

    Raises:
        ValueError: key does not exist in the interface map

    Returns:
        Interface: canonical interface
    """

    try:
        if user_input in SUPPORTED_INTERFACE_NAMES:
            return user_input
        return INTERFACE_MAP[user_input]
    except KeyError as exc:
        raise ValueError(
            f"Unknown interface {user_input}. Interface must be one of {SUPPORTED_INTERFACE_NAMES}."
        ) from exc


def _resolve_interface(interface: Interface, tapes: "qml.tape.QuantumScriptBatch") -> Interface:
    """Helper function to resolve the interface name based on a batch of tapes.

    Args:
        interface (str, Interface): original interface to use as reference
        tapes (list[.QuantumScript]): quantum tapes

    Returns:
        Interface: resolved interface"""

    interface = _get_canonical_interface_name(interface)

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
        # pylint: disable=unused-import
        try:  # pragma: no cover
            import jax
        except ImportError as e:  # pragma: no cover
            raise qml.QuantumFunctionError(  # pragma: no cover
                "jax not found. Please install the latest "  # pragma: no cover
                "version of jax to enable the 'jax' interface."  # pragma: no cover
            ) from e  # pragma: no cover

        interface = _get_jax_interface_name(tapes)

    return interface
