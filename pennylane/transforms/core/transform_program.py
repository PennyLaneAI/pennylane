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
This module contains the ``TransformProgram`` class.
"""
from collections.abc import Sequence
from functools import partial
from typing import overload

from pennylane.exceptions import TransformError
from pennylane.tape import QuantumScriptBatch
from pennylane.typing import BatchPostprocessingFn, PostprocessingFn, ResultBatch

from .cotransform_cache import CotransformCache
from .transform_dispatcher import TransformContainer, TransformDispatcher


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


class TransformProgram:
    """Class that contains a transform program and the methods to interact with it.

    The order of execution is the order in the list containing the containers.

    Args:
        initial_program (Optional[Sequence[TransformContainer]]): A sequence of transforms with
            which to initialize the program.
        cotransform_cache (Optional[CotransformCache]): A named tuple containing the ``qnode``,
            ``args``, and ``kwargs`` required to compute classical cotransforms.

    The main case where one would have to interact directly with a transform program is when developing a
    :class:`Device <pennylane.devices.Device>`. In this case, the pre-processing method of a device
    returns a transform program. You should directly refer to the device API documentation for more details.

    .. warning::

        This class is developer-facing and should not be used directly. Instead, use
        :func:`qml.transform <pennylane.transform>` if you would like to make a custom
        transform.

    .. seealso:: :func:`~.pennylane.transform`

    **Implemented Dunder methods**

    Programs have several implemented dunder methods for easy manipulation.

    >>> from pennylane.transforms.core.transform_program import TransformProgram
    >>> from copy import copy
    >>> program = TransformProgram()
    >>> program.add_transform(qml.compile)
    >>> program.add_transform(qml.transforms.cancel_inverses)
    >>> [t for t in program]  # Iteration
    [<compile([], {})>, <cancel_inverses([], {})>]
    >>> program[0]
    <compile([], {})>
    >>> program[::-1]
    TransformProgram(cancel_inverses, compile)
    >>> len(program)
    2
    >>> True if program else False
    True
    >>> True if TransformProgram() else False
    False
    >>> program2 = copy(program)
    >>> program2 == program
    True
    >>> qml.compile in program
    True
    >>> qml.transforms.split_non_commuting in program
    False
    >>> program + program
    TransformProgram(compile, cancel_inverses, compile, cancel_inverses)

    """

    def __init__(
        self,
        initial_program: Sequence[TransformContainer] | None = None,
        cotransform_cache: CotransformCache | None = None,
    ):
        self._transform_program = list(initial_program) if initial_program else []
        self.cotransform_cache = cotransform_cache

    def __iter__(self):
        """list[TransformContainer]: Return an iterator to the underlying transform program."""
        return self._transform_program.__iter__()

    def __len__(self) -> int:
        """int: Return the number transforms in the program."""
        return len(self._transform_program)

    @overload
    def __getitem__(self, idx: int) -> "TransformContainer": ...

    @overload
    def __getitem__(self, idx: slice) -> "TransformProgram": ...

    def __getitem__(self, idx):
        """(TransformContainer, List[TransformContainer]): Return the indexed transform container from underlying
        transform program"""
        if isinstance(idx, slice):
            return TransformProgram(self._transform_program[idx])
        return self._transform_program[idx]

    def __bool__(self) -> bool:
        return bool(self._transform_program)

    def __add__(self, other: "TransformProgram") -> "TransformProgram":
        if self.has_final_transform and other.has_final_transform:
            raise TransformError("The transform program already has a terminal transform.")

        transforms = self._transform_program + other._transform_program
        if self.has_final_transform:
            transforms.append(transforms.pop(len(self) - 1))

        cotransform_cache = None
        if self.cotransform_cache:
            if other.cotransform_cache:
                raise ValueError("Cannot add two transform programs with cotransform caches.")
            cotransform_cache = self.cotransform_cache
        elif other.cotransform_cache:
            cotransform_cache = other.cotransform_cache
        return TransformProgram(transforms, cotransform_cache=cotransform_cache)

    def __repr__(self):
        """The string representation of the transform program class."""
        contents = ", ".join(f"{transform_c.transform.__name__}" for transform_c in self)
        return f"TransformProgram({contents})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TransformProgram):
            return False

        return self._transform_program == other._transform_program

    def __contains__(self, obj) -> bool:
        if isinstance(obj, TransformContainer):
            return obj in self._transform_program
        if isinstance(obj, TransformDispatcher):
            return any(obj.transform == t.transform for t in self)
        return False

    def push_back(self, transform_container: TransformContainer):
        """Add a transform (container) to the end of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if not isinstance(transform_container, TransformContainer):
            raise TransformError("Only transform container can be added to the transform program.")

        # Program can only contain one informative transform and at the end of the program
        if self.has_final_transform:
            if transform_container.final_transform:
                raise TransformError("The transform program already has a terminal transform.")
            self._transform_program.insert(-1, transform_container)
            return
        self._transform_program.append(transform_container)

    def insert_front(self, transform_container: TransformContainer):
        """Insert the transform container at the beginning of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if (transform_container.final_transform) and not self.is_empty():
            raise TransformError(
                "Informative transforms can only be added at the end of the program."
            )
        self._transform_program.insert(0, transform_container)

    def add_transform(self, transform: TransformDispatcher, *targs, **tkwargs):
        """Add a transform (dispatcher) to the end of the program.

        Note that this should be a function decorated with/called by
        ``qml.transforms.transform``, and not a ``TransformContainer``.

        Args:
            transform (TransformDispatcher): The transform to add to the transform program.
            *targs: Any additional arguments that are passed to the transform.

        Keyword Args:
            **tkwargs: Any additional keyword arguments that are passed to the transform.

        """
        if not isinstance(transform, TransformDispatcher):
            raise TransformError("Only transform dispatcher can be added to the transform program.")

        if transform.expand_transform:
            self.push_back(
                TransformContainer(TransformDispatcher(transform.expand_transform), targs, tkwargs)
            )
        self.push_back(
            TransformContainer(
                transform,
                args=targs,
                kwargs=tkwargs,
            )
        )

    def insert_front_transform(self, transform: TransformDispatcher, *targs, **tkwargs):
        """Add a transform (dispatcher) to the beginning of the program.

        Args:
            transform(TransformDispatcher): The transform to add to the front of the transform program.
            *targs: Any additional arguments that are passed to the transform.

        Keyword Args:
            **tkwargs: Any additional keyword arguments that are passed to the transform.

        """
        if transform.final_transform and not self.is_empty():
            raise TransformError(
                "Informative transforms can only be added at the end of the program."
            )

        self.insert_front(
            TransformContainer(
                transform,
                args=targs,
                kwargs=tkwargs,
            )
        )

        if transform.expand_transform:
            self.insert_front(
                TransformContainer(TransformDispatcher(transform.expand_transform), targs, tkwargs)
            )

    def pop_front(self):
        """Pop the transform container at the beginning of the program.

        Returns:
            TransformContainer: The transform container at the beginning of the program.
        """
        return self._transform_program.pop(0)

    def get_last(self):
        """Get the last transform container.

        Returns:
            TransformContainer: The last transform in the program.

        Raises:
            TransformError: It raises an error if the program is empty.
        """
        if self:
            return self._transform_program[-1]
        raise TransformError(
            "The transform program is empty and you cannot get the last transform container."
        )

    def is_empty(self):
        """Check if the transform program is empty or not.

        Returns:
            bool: Boolean, True if empty, False otherwise.
        """
        return len(self) == 0

    @property
    def is_informative(self) -> bool:
        """``True`` if the transform program is informative.

        Returns:
            bool: Boolean
        """
        return self[-1].is_informative if self else False

    @property
    def has_final_transform(self) -> bool:
        """``True`` if the transform program has a terminal transform."""
        return self[-1].final_transform if self else False  # pylint: disable=no-member

    def has_classical_cotransform(self) -> bool:
        """Check if the transform program has some classical cotransforms.

        Returns:
            bool: Boolean
        """
        return any(t.classical_cotransform is not None for t in self)

    def set_classical_component(self, qnode, args, kwargs):
        """Set the classical jacobians and argnums if the transform is hybrid with a classical cotransform."""
        # pylint: disable=no-member
        if self.has_classical_cotransform() and self[-1].kwargs.get("hybrid", True):
            self.cotransform_cache = CotransformCache(qnode, args, kwargs)

    def prune_dynamic_transform(self, type_to_keep=1):
        """Ensures that only one or none ``dynamic_one_shot`` is applied.

        Args:
            type_to_keep (int): The type of the dynamic transform to keep. 0: keep none,
                1: dynamic_one_shot or mid_circuit_measurements, 2: only mid_circuit_measurements.

        Returns:
            bool: ``True`` if a dynamic transform was found, ``False`` otherwise.

        """

        i = len(self._transform_program) - 1
        found = False
        while i >= 0:
            t = self._transform_program[i]
            if "mid_circuit_measurements" in str(t) and type_to_keep > 0:
                type_to_keep = 0  # keep this and do not keep the rest
                found = True
            elif "dynamic_one_shot" in str(t) and type_to_keep == 1:
                type_to_keep = 0  # keep this and do not keep the rest
                found = True
            elif "dynamic_one_shot" in str(t) or "mid_circuit_measurements" in str(t):
                self._transform_program.pop(i)
            i -= 1
        return found

    def __call_tapes(
        self, tapes: QuantumScriptBatch
    ) -> tuple[QuantumScriptBatch, BatchPostprocessingFn]:
        if not self:
            return tapes, null_postprocessing

        processing_fns_stack = []

        for transform_container in self:
            transform, targs, tkwargs, cotransform, _, _, _ = transform_container
            tkwargs = {
                key: value for key, value in tkwargs.items() if key not in {"argnums", "hybrid"}
            }
            execution_tapes, fns, slices, classical_fns = [], [], [], []

            start = 0
            argnums = (
                self.cotransform_cache.get_argnums(transform_container)
                if self.cotransform_cache
                else None
            )

            classical_jacobians = []
            for tape_idx, tape in enumerate(tapes):
                if argnums is not None:
                    tape.trainable_params = argnums[tape_idx]
                new_tapes, fn = transform(tape, *targs, **tkwargs)
                execution_tapes.extend(new_tapes)

                fns.append(fn)
                end = start + len(new_tapes)
                slices.append(slice(start, end))
                start = end

                jac = (
                    self.cotransform_cache.get_classical_jacobian(transform_container, tape_idx)
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
        self, jaxpr: "jax.extend.core.Jaxpr", consts: Sequence, *args
    ) -> "jax.extend.core.ClosedJaxpr":
        # pylint: disable=import-outside-toplevel
        import jax

        cur_jaxpr = jax.extend.core.ClosedJaxpr(jaxpr, consts)
        for container in self:
            _, targs, tkwargs, _, plxpr_transform, _, _ = container
            cur_jaxpr = plxpr_transform(cur_jaxpr.jaxpr, cur_jaxpr.consts, targs, tkwargs, *args)

        return cur_jaxpr

    @overload
    def __call__(
        self, jaxpr: "jax.extend.core.Jaxpr", consts: Sequence, *args
    ) -> "jax.extend.core.ClosedJaxpr": ...

    @overload
    def __call__(
        self, tapes: QuantumScriptBatch
    ) -> tuple[QuantumScriptBatch, BatchPostprocessingFn]: ...

    def __call__(self, *args, **kwargs):
        if type(args[0]).__name__ == "Jaxpr":
            return self.__call_jaxpr(*args, **kwargs)
        return self.__call_tapes(*args, **kwargs)
