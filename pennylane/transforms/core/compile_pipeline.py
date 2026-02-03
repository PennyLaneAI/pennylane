# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains the ``CompilePipeline`` class.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, overload

from pennylane.exceptions import TransformError
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import BatchPostprocessingFn, PostprocessingFn, ResultBatch

from .cotransform_cache import CotransformCache
from .transform import BoundTransform, Transform

if TYPE_CHECKING:
    import jax

    import pennylane as qml


def _batch_postprocessing(
    results: ResultBatch,
    individual_fns: list[PostprocessingFn],
    slices: list[slice] | list[int],
) -> ResultBatch:
    """Broadcast individual post processing functions onto their respective tapes.

    Args:
        results (ResultBatch): The numeric outcome from executing a batch of :class:`~.QuantumTape`

    Keyword Args:
        individual_fns (List[Callable]): postprocessing functions converting a batch of results into a single result
            corresponding to only a single :class:`~.QuantumTape`.
        slices (List[slice]): the indices for the results that correspond to each individual post processing function.

    >>> results = (1.0, 2.0, 3.0, 4.0)
    >>> def postprocessing1(results):
    ...     return results[0] + results[1]
    >>> def postprocessing2(results):
    ...     return results[0]+0.5
    >>> def postprocessing3(results):
    ...     return results[0]*2
    >>> slices = [slice(0,2), slice(2,3), slice(3,4)]
    >>> individual_fns = [postprocessing1, postprocessing2, postprocessing3]
    >>> _batch_postprocessing(results, individual_fns, slices)
    (3.0, 3.5, 8.0)

    """
    return tuple(fn(results[sl]) for fn, sl in zip(individual_fns, slices, strict=True))


def _apply_postprocessing_stack(
    results: ResultBatch,
    postprocessing_stack: list[BatchPostprocessingFn],
) -> ResultBatch:
    """Applies the postprocessing and cotransform postprocessing functions in a Last-In-First-Out LIFO manner.

    Args:
        results (ResultBatch): The numeric outcome from executing a batch of :class:`~.QuantumTape`

    Keyword Args:
        postprocessing_stack (List(BatchPostProcessingFn)): a LIFO stack of post processing functions.

    Returns:
        ResultBatch: the post processed results.

    >>> results = (1.0, 2.0, 3.0, 4.0)
    >>> def postprocessing1(results):
    ...     return (results[0] + results[1], results[2] + results[3])
    >>> def postprocessing2(results):
    ...     return (results[0] + 1, results[1] + 2)
    >>> _apply_postprocessing_stack(results, [postprocessing1])
    (3.0, 7.0)
    >>> _apply_postprocessing_stack(results, [postprocessing2, postprocessing1])
    (4.0, 9.0)

    """
    for postprocessing in reversed(postprocessing_stack):
        results = postprocessing(results)
    return results


def null_postprocessing(results: ResultBatch) -> ResultBatch:
    """An empty postprocessing function that simply returns its input.

    Args:
        results (ResultBatch): Results from executing a batch of :class:`~.QuantumTape`.

    Returns:
        ResultBatch: the input to the function.

    """
    return results


class CompilePipeline:
    """A sequence of transforms to be applied to a quantum function or a :class:`~pennylane.QNode`.

    Args:
        *transforms (Optional[Sequence[Transform | BoundTransform]]): A sequence of
            transforms with which to initialize the program.
        cotransform_cache (Optional[CotransformCache]): A named tuple containing the ``qnode``,
            ``args``, and ``kwargs`` required to compute classical cotransforms.

    **Example:**

    The ``CompilePipeline`` class allows you to chain together multiple quantum function transforms
    to create custom circuit optimization pipelines.

    For example, consider if you wanted to apply the following optimizations to a quantum circuit:

    - pushing all commuting single-qubit gates as far right as possible
      (:func:`~pennylane.transforms.commute_controlled`)
    - cancellation of adjacent inverse gates
      (:func:`~pennylane.transforms.cancel_inverses`)
    - merging adjacent rotations of the same type
      (:func:`~pennylane.transforms.merge_rotations`)

    You can specify a transform program (``pipeline``) by passing these transforms to the ``CompilePipeline``
    class. By applying the created ``pipeline`` directly on a quantum function as a decorator, the circuit will
    be transformed with each pass within the pipeline sequentially:

    .. code-block:: python

        pipeline = qml.CompilePipeline(
            qml.transforms.commute_controlled,
            qml.transforms.cancel_inverses(recursive=True),
            qml.transforms.merge_rotations,
        )

        @pipeline
        @qml.qnode(qml.device("default.qubit"))
        def circuit(x, y):
            qml.CNOT([1, 0])
            qml.X(0)
            qml.CNOT([1, 0])
            qml.H(0)
            qml.H(0)
            qml.X(0)
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            return qml.expval(qml.Z(1))

    >>> print(qml.draw(circuit)(0.1, 0.2))
    0: ──RX(0.30)─┤
    1: ───────────┤  <Z>

    Alternatively, the transform program can be constructed intuitively by combining multiple transforms. For
    example, the transforms can be added together with ``+``:

    >>> pipeline = qml.transforms.merge_rotations + qml.transforms.cancel_inverses(recursive=True)
    >>> print(pipeline)
    CompilePipeline(
      [0] merge_rotations(),
      [1] cancel_inverses(recursive=True)
    )

    Or multiplied by a scalar via ``*``:

    >>> pipeline += 2 * qml.transforms.commute_controlled
    >>> print(pipeline)
    CompilePipeline(
      [0] merge_rotations(),
      [1] cancel_inverses(recursive=True),
      [2] commute_controlled(),
      [3] commute_controlled()
    )

    A compilation pipeline can also be easily modified using operations similar to Python lists, including
    ``insert``, ``append``, ``extend`` and ``pop``:

    >>> pipeline.insert(0, qml.transforms.remove_barrier)
    >>> print(pipeline)
    CompilePipeline(
      [0] remove_barrier(),
      [1] merge_rotations(),
      [2] cancel_inverses(recursive=True),
      [3] commute_controlled(),
      [4] commute_controlled()
    )

    Additionally, multiple compilation pipelines can be concatenated:

    >>> another_pipeline = qml.decompose(gate_set={qml.RX, qml.RZ, qml.CNOT}) + qml.transforms.combine_global_phases
    >>> print(another_pipeline + pipeline)
    CompilePipeline(
      [0] decompose(gate_set=...),
      [1] combine_global_phases(),
      [2] remove_barrier(),
      [3] merge_rotations(),
      [4] cancel_inverses(recursive=True),
      [5] commute_controlled(),
      [6] commute_controlled()
    )

    We can create a new pipeline that will do multiple passes of the original with multiplication:

    >>> original = qml.transforms.merge_rotations + qml.transforms.cancel_inverses
    >>> print(2 * original)
    CompilePipeline(
      [0] merge_rotations(),
      [1] cancel_inverses(),
      [2] merge_rotations(),
      [3] cancel_inverses()
    )

    """

    @overload
    def __init__(
        self,
        transforms: Sequence[Transform | BoundTransform],
        /,
        *,
        cotransform_cache: CotransformCache | None = None,
    ): ...

    @overload
    def __init__(
        self,
        *transforms: CompilePipeline | BoundTransform | Transform,
        cotransform_cache: CotransformCache | None = None,
    ): ...

    def __init__(
        self,
        *transforms: CompilePipeline
        | BoundTransform
        | Transform
        | Sequence[Transform | BoundTransform],
        cotransform_cache: CotransformCache | None = None,
    ):
        if len(transforms) == 1 and isinstance(transforms[0], Sequence):
            transforms = list(transforms[0])
            # If all elements are BoundTransform, store directly (already expanded)
            if all(isinstance(t, BoundTransform) for t in transforms):
                self._compile_pipeline = transforms
                self._markers: dict[str, int] = {}
                self.cotransform_cache = cotransform_cache
                return

        self._compile_pipeline = []
        self._markers: dict[str, int] = {}
        self.cotransform_cache = cotransform_cache
        for obj in transforms:
            if not isinstance(obj, (CompilePipeline, BoundTransform, Transform)):
                raise TypeError(
                    "CompilePipeline can only be constructed with a series of transforms "
                    "or compile pipelines, or with a single list of transforms."
                )
            self += obj

    def __copy__(self):
        old_markers = self._markers.copy()
        new_pipeline = CompilePipeline(
            self._compile_pipeline[:], cotransform_cache=self.cotransform_cache
        )
        new_pipeline._markers = old_markers
        return new_pipeline

    def __iter__(self):
        """list[BoundTransform]: Return an iterator to the underlying compile pipeline."""
        return self._compile_pipeline.__iter__()

    def __len__(self) -> int:
        """int: Return the number transforms in the program."""
        return len(self._compile_pipeline)

    @overload
    def __getitem__(self, idx: int) -> BoundTransform: ...

    @overload
    def __getitem__(self, idx: slice) -> CompilePipeline: ...

    def __getitem__(self, idx):
        """(BoundTransform, List[BoundTransform]): Return the indexed transform container from underlying
        compile pipeline"""
        if isinstance(idx, slice):
            markers = {k: v - idx.start for k, v in self._markers.items() if v >= (idx.start or 0)}
            compile_pipeline = CompilePipeline(self._compile_pipeline[idx])
            compile_pipeline._markers = markers
            return compile_pipeline
        return self._compile_pipeline[idx]

    def __bool__(self) -> bool:
        return bool(self._compile_pipeline)

    def __add__(self, other: CompilePipeline | BoundTransform | Transform) -> CompilePipeline:
        # Convert dispatcher to container if needed
        if isinstance(other, Transform):
            other = BoundTransform(other)

        # Handle BoundTransform
        if isinstance(other, BoundTransform):
            transforms = [other]
            if expand_transform := other.expand_transform:
                transforms.insert(0, expand_transform)
            other = CompilePipeline(transforms)

        # Handle CompilePipeline
        if isinstance(other, CompilePipeline):
            offset = len(self._compile_pipeline)
            for name, pos in other._markers.items():
                self._markers[name] = pos + offset

            if self.has_final_transform and other.has_final_transform:
                raise TransformError("The compile pipeline already has a terminal transform.")

            transforms = self._compile_pipeline[:]
            transforms.extend(other._compile_pipeline)

            cotransform_cache = None
            if self.cotransform_cache:
                if other.cotransform_cache:
                    raise ValueError("Cannot add two compile pipelines with cotransform caches.")
                cotransform_cache = self.cotransform_cache
            elif other.cotransform_cache:
                cotransform_cache = other.cotransform_cache
            new_pipeline = CompilePipeline(transforms, cotransform_cache=cotransform_cache)
            new_pipeline._markers = self._markers
            return new_pipeline

        return NotImplemented

    def __radd__(self, other: BoundTransform | Transform) -> CompilePipeline:
        """Right addition to prepend a transform to the program.

        Args:
            other: A BoundTransform or Transform to prepend.

        Returns:
            CompilePipeline: A new program with the transform prepended.
        """
        if isinstance(other, Transform):
            other = BoundTransform(other)

        if not isinstance(other, BoundTransform):
            return NotImplemented

        if self.has_final_transform and other.is_final_transform:
            raise TransformError("The compile pipeline already has a terminal transform.")

        transforms = [other]
        if expand_transform := other.expand_transform:
            transforms.insert(0, expand_transform)
        transforms += self._compile_pipeline
        return CompilePipeline(transforms, cotransform_cache=self.cotransform_cache)

    def __iadd__(self, other: CompilePipeline | BoundTransform | Transform) -> CompilePipeline:
        """In-place addition to append a transform to the program.

        Args:
            other: A BoundTransform, Transform, or CompilePipeline to append.

        Returns:
            CompilePipeline: This program with the transform(s) appended.
        """
        # Convert dispatcher to container if needed
        if isinstance(other, Transform):
            other = BoundTransform(other)

        if isinstance(other, BoundTransform):
            transforms = [other]
            if expand_transform := other.expand_transform:
                transforms.insert(0, expand_transform)
            other = CompilePipeline(transforms)

        if isinstance(other, CompilePipeline):
            offset = len(self._compile_pipeline)
            for name, pos in other._markers.items():
                self._markers[name] = pos + offset

            if self.has_final_transform and other.has_final_transform:
                raise TransformError("The compile pipeline already has a terminal transform.")

            self._compile_pipeline.extend(other._compile_pipeline)

            if other.cotransform_cache:
                if self.cotransform_cache:
                    raise ValueError("Cannot add two compile pipelines with cotransform caches.")
                self.cotransform_cache = other.cotransform_cache

            return self

        return NotImplemented

    def __mul__(self, n: int) -> CompilePipeline:
        """Right multiplication to repeat a program n times.

        Args:
            n (int): Number of times to repeat this program.

        Returns:
            CompilePipeline: A new program with this program repeated n times.
        """
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Cannot multiply compile pipeline by negative integer")

        if self.has_final_transform:
            raise TransformError(
                "Cannot multiply a compile pipeline that has a terminal transform."
            )

        transforms = self._compile_pipeline * n
        new_pipeline = CompilePipeline(transforms, cotransform_cache=self.cotransform_cache)
        new_pipeline._markers = self._markers.copy()
        return new_pipeline

    __rmul__ = __mul__

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        # See https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
        print(str(self))

    def __str__(self) -> str:
        """Returns a user friendly representation of the compile pipeline."""
        if not self:
            return "CompilePipeline()"

        def truncate(val: Any) -> str:
            _CHAR_THRESHOLD = 50
            val_str = str(val)
            return "..." if len(val_str) > _CHAR_THRESHOLD else val_str

        lines = []
        inv_marker_map = {v: k for k, v in self._markers.items()}
        for i, transform in enumerate(self):
            args_str = ", ".join(truncate(a) for a in transform.args)
            kwargs_str = ", ".join(f"{k}={truncate(v)}" for k, v in transform.kwargs.items())

            sep = ", " if args_str and kwargs_str else ""
            transform_str = f"{transform.tape_transform.__name__}({args_str}{sep}{kwargs_str})"
            lines.append(f"  [{i}] {transform_str}" + "," * bool(i != len(self) - 1))

            if inv_marker_map.get(i):
                lines.append(f"   └─▶ {inv_marker_map[i]}")

        contents = "\n".join(lines)
        return f"CompilePipeline(\n{contents}\n)"

    def __repr__(self) -> str:
        """The detailed string representation of the compile pipeline."""
        if not self:
            return "CompilePipeline()"

        lines = []
        inv_marker_map = {v: k for k, v in self._markers.items()}

        for i, transform in enumerate(self):
            lines.append(f"  [{i}] {repr(transform)}" + "," * bool(i != len(self) - 1))
            if inv_marker_map.get(i):
                lines.append(f"   └─▶ {inv_marker_map[i]}")

        contents = "\n".join(lines)
        return f"CompilePipeline(\n{contents}\n)"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CompilePipeline):
            return False

        return self._compile_pipeline == other._compile_pipeline

    def __contains__(self, obj) -> bool:
        if isinstance(obj, BoundTransform):
            return obj in self._compile_pipeline
        if isinstance(obj, Transform):
            return any(obj.tape_transform == t.tape_transform for t in self)
        return False

    def remove(self, obj: BoundTransform | Transform):
        """In place remove the input containers, specifically,
        1. if the input is a Transform, remove all containers matching the transform;
        2. if the input is a BoundTransform, remove all containers exactly matching the input.

        Args:
            obj (BoundTransform or Transform): The object to remove from the program.
        """
        if not isinstance(obj, (Transform, BoundTransform)):
            raise TypeError("Only BoundTransform or Transform can be removed.")

        i = len(self) - 1
        while i >= 0:
            # pylint: disable=protected-access
            if (isinstance(obj, Transform) and obj == self[i]._transform) or self[i] == obj:
                removed = self._compile_pipeline.pop(i)
                # Remove the associated expand_transform if present
                if i > 0 and removed.expand_transform == self[i - 1]:
                    self._compile_pipeline.pop(i - 1)
                    i -= 1
            i -= 1

    def append(self, transform: BoundTransform | Transform):
        """Add a transform to the end of the program.

        Args:
            transform (Transform or BoundTransform): A transform represented by its container.

        """
        if isinstance(transform, (list, tuple, CompilePipeline)):
            raise TypeError(
                "append() expects a single transform, not a sequence. "
                "Use extend() to add multiple transforms at once."
            )

        if not isinstance(transform, BoundTransform):
            transform = BoundTransform(transform)

        # Program can only contain one informative transform and at the end of the program
        if self.has_final_transform and transform.is_final_transform:
            raise TransformError("The compile pipeline already has a terminal transform.")

        if expand_transform := transform.expand_transform:
            self._compile_pipeline.append(expand_transform)
        self._compile_pipeline.append(transform)

    def extend(self, transforms: CompilePipeline | Sequence[BoundTransform | Transform]):
        """Extend the pipeline by appending transforms from an iterable.

        Args:
            transforms (CompilePipeline, or Sequence[BoundTransform | Transform]): A
                CompilePipeline or an iterable of transforms to append.

        """
        # Handle CompilePipeline by using __iadd__ which already handles this case
        if isinstance(transforms, CompilePipeline):
            self += transforms
            return

        # Handle iterables (list, tuple, etc.)
        for t in transforms:
            self += t

    def add_marker(self, level: str) -> None:
        """Adds a mark at the current level."""
        protected_options = {"top", "user", "gradient", "device", "all", "all-mlir"}
        if level in protected_options:
            raise ValueError(
                f"Found marker for protected level {level}. Protected options are {protected_options}."
            )
        if level in self._markers:
            raise ValueError(f"Found multiple markers for level {level}. Markers should be unique.")

        self._markers[level] = len(self._compile_pipeline) - 1 if self else 0

    @property
    def markers(self) -> list[str]:
        """Retrieve list of markers present in the compiliation pipeline."""
        return list(self._markers.keys())

    def get_marker_level(self, label: str) -> int | None:
        """Retrieve the level corresponding to a marker label."""
        return self._markers.get(label)

    def add_transform(self, transform: Transform, *targs, **tkwargs):
        """Add a transform to the end of the program.

        Note that this should be a function decorated with/called by
        ``qml.transform``, and not a ``BoundTransform``.

        Args:
            transform (Transform): The transform to add to the compile pipeline.
            *targs: Any additional arguments that are passed to the transform.

        Keyword Args:
            **tkwargs: Any additional keyword arguments that are passed to the transform.

        """
        if not isinstance(transform, Transform):
            raise TransformError("Only transforms can be added to the compile pipeline.")
        self.append(BoundTransform(transform, args=targs, kwargs=tkwargs))

    def insert(self, index: int, transform: Transform | BoundTransform):
        """Insert a transform at a given index.

        Args:
            index (int): The index to insert the transform.
            transform (transform or BoundTransform): the transform to insert

        """
        self._markers = {k: (v + 1 if v >= index else v) for k, v in self._markers.items()}

        if not isinstance(transform, BoundTransform):
            transform = BoundTransform(transform)

        # Program can only contain one informative transform and at the end of the program
        if self and transform.is_final_transform:
            raise TransformError("Terminal transform can only be added to the end of the pipeline.")

        self._compile_pipeline.insert(index, transform)
        if expand_transform := transform.expand_transform:
            self._compile_pipeline.insert(index, expand_transform)

    def pop(self, index: int = -1):
        """Pop the transform container at a given index of the program.

        Args:
            index (int): the index of the transform to remove.

        Returns:
            BoundTransform: The removed transform.

        """
        transform = self._compile_pipeline.pop(index)
        if index > 0 and transform.expand_transform == self._compile_pipeline[index - 1]:
            self._compile_pipeline.pop(index - 1)
        return transform

    @property
    def is_informative(self) -> bool:
        """``True`` if the compile pipeline is informative.

        Returns:
            bool: Boolean
        """
        return any(t.is_informative for t in self)

    @property
    def has_final_transform(self) -> bool:
        """``True`` if the compile pipeline has a terminal transform."""
        return any(t.is_final_transform for t in self)

    def has_classical_cotransform(self) -> bool:
        """Check if the compile pipeline has some classical cotransforms.

        Returns:
            bool: Boolean
        """
        return any(t.classical_cotransform is not None for t in self)

    def set_classical_component(self, qnode, args, kwargs):
        """Set the classical jacobians and argnums if the transform is hybrid with a classical cotransform."""
        # pylint: disable=no-member
        if self.has_classical_cotransform() and self[-1].kwargs.get("hybrid", True):
            self.cotransform_cache = CotransformCache(qnode, args, kwargs)

    def __call_tapes(
        self, tapes: QuantumScript | QuantumScriptBatch
    ) -> tuple[QuantumScriptBatch, BatchPostprocessingFn]:
        if not self:
            return tapes, null_postprocessing

        if isinstance(tapes, QuantumScript):
            tapes = (tapes,)

        processing_fns_stack = []

        for bound_transform in self:
            transform, targs, tkwargs, cotransform, _, _, _ = bound_transform
            tkwargs = {
                key: value for key, value in tkwargs.items() if key not in {"argnums", "hybrid"}
            }
            execution_tapes, fns, slices, classical_fns = [], [], [], []

            start = 0
            argnums = (
                self.cotransform_cache.get_argnums(bound_transform)
                if self.cotransform_cache
                else None
            )

            classical_jacobians = []
            for tape_idx, tape in enumerate(tapes):
                if argnums is not None:
                    tape.trainable_params = argnums[tape_idx]
                if transform is None:
                    raise NotImplementedError(
                        f"transform {bound_transform} has no defined tape transform."
                    )
                new_tapes, fn = transform(tape, *targs, **tkwargs)
                execution_tapes.extend(new_tapes)

                fns.append(fn)
                end = start + len(new_tapes)
                slices.append(slice(start, end))
                start = end

                jac = (
                    self.cotransform_cache.get_classical_jacobian(bound_transform, tape_idx)
                    if self.cotransform_cache
                    else None
                )
                classical_jacobians.append(jac)
                if cotransform and classical_jacobians[-1] is not None:
                    classical_fns.append(
                        partial(cotransform, cjac=classical_jacobians[-1], tape=tape)
                    )

            if cotransform and classical_fns:
                slices_classical = list(range(len(tapes)))
                batch_postprocessing_classical = partial(
                    _batch_postprocessing, individual_fns=classical_fns, slices=slices_classical
                )
                batch_postprocessing_classical.__doc__ = _batch_postprocessing.__doc__
                processing_fns_stack.append(batch_postprocessing_classical)

            batch_postprocessing = partial(_batch_postprocessing, individual_fns=fns, slices=slices)
            batch_postprocessing.__doc__ = _batch_postprocessing.__doc__
            processing_fns_stack.append(batch_postprocessing)

            # set input tapes for next iteration.
            tapes = execution_tapes

        postprocessing_fn = partial(
            _apply_postprocessing_stack,
            postprocessing_stack=processing_fns_stack,
        )

        postprocessing_fn.__doc__ = _apply_postprocessing_stack.__doc__

        # Reset classical jacobians
        return tuple(tapes), postprocessing_fn

    def __call_jaxpr(
        self, jaxpr: jax.extend.core.Jaxpr, consts: Sequence, *args
    ) -> jax.extend.core.ClosedJaxpr:
        # pylint: disable=import-outside-toplevel
        import jax

        cur_jaxpr = jax.extend.core.ClosedJaxpr(jaxpr, consts)
        for container in self:
            _, targs, tkwargs, _, plxpr_transform, _, _ = container
            cur_jaxpr = plxpr_transform(cur_jaxpr.jaxpr, cur_jaxpr.consts, targs, tkwargs, *args)

        return cur_jaxpr

    def __call_generic(self, obj):
        """Apply the transform program to a generic object (QNode, device, callable, etc.).

        This method chain-applies each transform using the generic dispatch system.

        Args:
            obj: The object to transform (QNode, device, callable, etc.).

        Returns:
            The transformed object.
        """
        result = obj
        for container in self:
            result = container(result)
        return result

    @overload
    def __call__(
        self, jaxpr: jax.extend.core.Jaxpr, consts: Sequence, *args
    ) -> jax.extend.core.ClosedJaxpr: ...

    @overload
    def __call__(self, qnode: qml.QNode, *args, **kwargs) -> qml.QNode: ...

    @overload
    def __call__(
        self, tape: QuantumScript | QuantumScriptBatch
    ) -> tuple[QuantumScriptBatch, BatchPostprocessingFn]: ...

    def __call__(self, *args, **kwargs):
        if type(args[0]).__name__ == "Jaxpr":
            return self.__call_jaxpr(*args, **kwargs)

        first_arg = args[0]

        # Sequence of QuantumScripts: QuantumScriptBatch
        if isinstance(first_arg, (QuantumScript, Sequence)):
            return self.__call_tapes(*args, **kwargs)

        # For any other object (QNode, device, callable, etc.),
        # chain-apply each transform using the generic dispatch system
        return self.__call_generic(first_arg)


@Transform.generic_register
def _apply_to_program(obj: CompilePipeline, transform, *targs, **tkwargs):
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
    program = copy(obj)
    program.append(BoundTransform(transform, args=targs, kwargs=tkwargs))
    return program


TransformProgram = CompilePipeline
