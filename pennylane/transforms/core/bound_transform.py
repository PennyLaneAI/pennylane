# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This module defines the data structure that encapsulates a 'BoundTransform'.
"""

import warnings
from collections.abc import Callable

from pennylane.exceptions import PennyLaneDeprecationWarning, TransformError

from .compile_pipeline import CompilePipeline
from .transform import Transform


class BoundTransform:  # pylint: disable=too-many-instance-attributes
    """A transform with bound inputs.

    Args:
        transform: Any transform.
        args (Sequence[Any]): The positional arguments to use with the transform.
        kwargs (Dict | None): The keyword arguments for use with the transform.

    Keyword Args:
        use_argnum (bool): An advanced option used in conjunction with calculating
            classical cotransforms of jax workflows.

    .. seealso:: :func:`~.pennylane.transform`

    >>> bound_t = BoundTransform(qml.transforms.merge_rotations, (), {"atol": 1e-4})
    >>> bound_t
    <merge_rotations(atol=0.0001)>

    The class can also be created by directly calling the transform with its inputs:

    >>> qml.transforms.merge_rotations(atol=1e-4)
    <merge_rotations(atol=0.0001)>

    These objects can now directly applied to anything individual transforms can apply to:

    .. code-block:: python

        @bound_t
        @qml.qnode(qml.device('null.qubit', wires=2))
        def c(x):
            qml.RX(x, 0)
            qml.RX(-x + 1e-6, 0)
            qml.RY(x, 1)
            qml.RY(-x + 1e-2, 1)
            return qml.probs(wires=(0,1))

    If we draw this circuit, we can see that the ``merge_rotations`` transforms was applied with a
    tolerance of ``1e-4``.  The ``RX`` gates sufficiently close to zero disappear, while the ``RY`` gates
    that are further from zero remain.

    >>> print(qml.draw(c)(1.0))
    0: ───────────┤ ╭Probs
    1: ──RY(0.01)─┤ ╰Probs

    Repeated versions of the bound transform can be created with multiplication:

    >>> print(bound_t * 3)
    CompilePipeline(
      [1] merge_rotations(atol=0.0001),
      [2] merge_rotations(atol=0.0001),
      [3] merge_rotations(atol=0.0001)
    )

    And it can be used in conjunction with both individual transforms, bound transforms, and
    compile pipelines.

    >>> print(bound_t + qml.transforms.cancel_inverses)
    CompilePipeline(
      [1] merge_rotations(atol=0.0001),
      [2] cancel_inverses()
    )
    >>> print(bound_t + qml.transforms.cancel_inverses + bound_t)
    CompilePipeline(
      [1] merge_rotations(atol=0.0001),
      [2] cancel_inverses(),
      [3] merge_rotations(atol=0.0001)
    )

    """

    def __hash__(self):
        hashable_dict = tuple((key, value) for key, value in self.kwargs.items())
        return hash((self.tape_transform, self.pass_name, self.args, hashable_dict))

    def __init__(
        self,
        transform: Transform,
        args: tuple | list = (),
        kwargs: None | dict = None,
        *,
        use_argnum: bool = False,
        **transform_config,
    ):
        if not isinstance(transform, Transform):
            transform = Transform(transform, **transform_config)
        elif transform_config:
            raise ValueError(
                f"transform_config kwargs {transform_config} cannot be passed if a transform is provided."
            )
        self._transform = transform
        self._args = tuple(args)
        self._kwargs = kwargs or {}
        self._use_argnum = use_argnum

    def __repr__(self):
        name = self.tape_transform.__name__ if self.tape_transform else self.pass_name
        arg_str = ", ".join(repr(a) for a in self._args) if self._args else ""
        kwarg_str = (
            ", ".join(f"{key}={value}" for key, value in self._kwargs.items())
            if self._kwargs
            else ""
        )
        if arg_str and kwarg_str:
            total_str = ", ".join([arg_str, kwarg_str])
        elif arg_str:
            total_str = arg_str
        else:
            total_str = kwarg_str
        return f"<{name}({total_str})>"

    def __call__(self, obj):
        return self._transform(obj, *self.args, **self.kwargs)

    def __iter__(self):
        return iter(
            (
                self._transform.tape_transform,
                self._args,
                self._kwargs,
                self._transform.classical_cotransform,
                self._transform.plxpr_transform,
                self._transform.is_informative,
                self._transform.is_final_transform,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundTransform):
            return False
        return (
            self.args == other.args
            and self.tape_transform == other.tape_transform
            and self.pass_name == other.pass_name
            and self.kwargs == other.kwargs
            and self.classical_cotransform == other.classical_cotransform
            and self.is_informative == other.is_informative
            and self.is_final_transform == other.is_final_transform
        )

    @property
    def tape_transform(self) -> Callable | None:
        """The raw tape transform definition for the transform."""
        return self._transform.tape_transform

    @property
    def transform(self) -> Callable | None:
        """The raw tape transform definition of the transform.

        .. warning::
            This property is deprecated and will be removed in v0.46.
            Please use :attr:`~.BoundTransform.tape_transform` instead.

        """
        warnings.warn(
            "The 'BoundTransform.transform' property is deprecated and will be removed in v0.46. "
            "Please use 'BoundTransform.tape_transform' instead.",
            PennyLaneDeprecationWarning,
            stacklevel=2,
        )
        return self.tape_transform

    @property
    def expand_transform(self) -> BoundTransform | None:
        """The expand_transform associated with this transform."""
        if not self._transform.expand_transform:
            return None
        return BoundTransform(
            self._transform.expand_transform,
            args=self.args,
            kwargs=self.kwargs,
            use_argnum=self._transform._use_argnum_in_expand,  # pylint:disable=protected-access
        )

    @property
    def pass_name(self) -> None | str:
        """The name of the corresponding Catalyst pass, if it exists."""
        return self._transform.pass_name

    @property
    def args(self) -> tuple:
        """The stored quantum transform's ``args``."""
        return self._args

    @property
    def kwargs(self) -> dict:
        """The stored quantum transform's ``kwargs``."""
        return self._kwargs

    @property
    def classical_cotransform(self) -> None | Callable:
        """The stored quantum transform's classical co-transform."""
        return self._transform.classical_cotransform

    @property
    def plxpr_transform(self) -> None | Callable:
        """The stored quantum transform's PLxPR transform.

        **UNMAINTAINED AND EXPERIMENTAL**
        """
        return self._transform.plxpr_transform

    @property
    def is_informative(self) -> bool:
        """Whether or not a transform is informative. If true the transform is queued at the end
        of the transform program and the tapes or qnode aren't executed.

        This property is rare, but used by such transforms as ``qml.transforms.commutation_dag``.
        """
        return self._transform.is_informative

    @property
    def is_final_transform(self) -> bool:
        """Whether or not the transform must be the last one to be executed
        in a ``CompilePipeline``.

        This property is ``True`` for most gradient transforms.
        """
        return self._transform.is_final_transform

    def __add__(self, other):
        """Add two transforms to create a CompilePipeline."""

        if not isinstance(other, (Transform, BoundTransform)):
            return NotImplemented

        if self.is_final_transform and other.is_final_transform:
            raise TransformError(
                f"Both {self} and {other} are final transforms and cannot be combined."
            )

        return CompilePipeline(self, other)

    def __mul__(self, n):
        """Multiply by an integer to create a pipeline with this transform repeated."""

        if not isinstance(n, int):
            return NotImplemented

        if n < 0:
            raise ValueError("Cannot multiply transform container by negative integer")

        if self.is_final_transform and n > 1:
            raise TransformError(
                f"{self} is a final transform and cannot be applied more than once."
            )

        return CompilePipeline(self) * n

    __rmul__ = __mul__


TransformContainer = BoundTransform
