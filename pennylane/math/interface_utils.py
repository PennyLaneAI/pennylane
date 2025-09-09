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

import autoray as ar


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

    @classmethod
    def _missing_(cls, value) -> "Interface":
        """Custom lookup to allow users to pass in None or common
        variants of the interface names."""
        match value:
            case None | "scipy":
                return cls.NUMPY
            case "jax-python" | "JAX":
                return cls.JAX
            case "pytorch":
                return cls.TORCH
            case (
                "tensorflow"
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                return cls.TF
            case (
                "tensorflow-autograph"
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                return cls.TF_AUTOGRAPH

        supported_values = [item.value for item in cls]

        standard_msg = f"'{value}' is not a valid {cls.__name__}."
        custom_addition = f"Please use one of the supported interfaces: {supported_values}."
        raise ValueError(f"{standard_msg} {custom_addition}")

    def get_like(self) -> str | None:
        """Maps canonical set of interfaces to those known by autoray."""
        mapping = {
            Interface.AUTOGRAD: "autograd",
            Interface.NUMPY: "numpy",
            Interface.TORCH: "torch",
            Interface.JAX: "jax",
            Interface.JAX_JIT: "jax",
            Interface.TF: "tensorflow",
            Interface.TF_AUTOGRAPH: "tensorflow",
            Interface.AUTO: None,
        }
        return mapping[self]

    def __eq__(self, interface) -> bool:
        if isinstance(interface, str):
            raise TypeError("Cannot compare Interface with str")
        return super().__eq__(interface)

    def __hash__(self) -> int:
        return super().__hash__()


SUPPORTED_INTERFACE_NAMES = list(Interface)
"""list[Interface]: allowed interface names"""


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

    if (
        len(interfaces - {"numpy", "scipy", "autograd"}) > 1
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)# pragma: no cover (TensorFlow tests were disabled during deprecation)
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

    priority_queue = ("tensorflow", "torch", "jax", "autograd", "scipy")
    for target_interface in priority_queue:
        if target_interface in interfaces:
            return target_interface

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
