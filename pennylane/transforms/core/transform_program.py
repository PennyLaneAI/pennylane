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
This module contains the transform program class.
"""
from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence

from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape

from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher

PostProcessingFn = Callable[[ResultBatch], Result]
BatchPostProcessingFn = Callable[[ResultBatch], ResultBatch]


def _batch_postprocessing(
    results: ResultBatch, individual_fns: List[PostProcessingFn], slices: List[slice]
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
    return tuple(fn(results[sl]) for fn, sl in zip(individual_fns, slices))


def _apply_postprocessing_stack(
    results: ResultBatch,
    postprocessing_stack: List[BatchPostProcessingFn],
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
    .... return (results[0] + 1, results[1] + 2)
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
    """Class that contains a transform program and the methods to interact with it. The order of execution is the order
    in the list containing the containers.

    .. warning::

        This class is developer-facing and should not be used directly.

    .. seealso:: :func:`~.pennylane.transforms.core.transform`

    """

    def __init__(self, initial_program: Optional[Sequence] = None):
        self._transform_program = list(initial_program) if initial_program else []

    def __iter__(self):
        """list[TransformContainer]: Return an iterator to the underlying transform program."""
        return self._transform_program.__iter__()

    def __len__(self):
        """int: Return the number transforms in the program."""
        return len(self._transform_program)

    def __getitem__(self, idx):
        """(TransformContainer, List[TransformContainer]): Return the indexed transform container from underlying
        transform program"""
        return self._transform_program[idx]

    def __bool__(self):
        return bool(self._transform_program)

    def __add__(self, other):
        program = TransformProgram(self._transform_program)
        for container in other:
            program.push_back(container)
        return program

    def __repr__(self):
        """The string representation of the transform program class."""
        contents = ", ".join(f"{transform_c.transform.__name__}" for transform_c in self)
        return f"TransformProgram({contents})"

    def push_back(self, transform_container: TransformContainer):
        """Add a transform (container) to the end of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if not isinstance(transform_container, TransformContainer):
            raise TransformError("Only transform container can be added to the transform program.")

        # Program can only contain one informative transform and at the end of the program
        if self.is_informative:
            raise TransformError("The transform program already has an informative transform.")
        self._transform_program.append(transform_container)

    def insert_front(self, transform_container: TransformContainer):
        """Insert the transform container at the beginning of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if transform_container.is_informative and not self.is_empty():
            raise TransformError(
                "Informative transforms can only be added at the end of the program."
            )
        self._transform_program.insert(0, transform_container)

    def add_transform(self, transform: TransformDispatcher, *targs, **tkwargs):
        """Add a transform (dispatcher) to the end of the program.

        Note that this should be a function decorated with/called by
        `qml.transforms.transform`, and not a `TransformContainer`.

        Args:
            transform (TransformDispatcher): The transform to add to the transform program.
            *targs: Any additional arguments that are passed to the transform.

        Keyword Args:
            **tkwargs: Any additional keyword arguments that are passed to the transform.

        """
        if not isinstance(transform, TransformDispatcher):
            raise TransformError("Only transform dispatcher can be added to the transform program.")

        if transform.expand_transform:
            self.push_back(TransformContainer(transform.expand_transform, targs, tkwargs))
        self.push_back(
            TransformContainer(
                transform.transform,
                targs,
                tkwargs,
                transform.classical_cotransform,
                transform.is_informative,
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
        if transform.is_informative and not self.is_empty():
            raise TransformError(
                "Informative transforms can only be added at the end of the program."
            )

        self.insert_front(
            TransformContainer(
                transform.transform,
                targs,
                tkwargs,
                transform.classical_cotransform,
                transform.is_informative,
            )
        )

        if transform.expand_transform:
            self.insert_front(TransformContainer(transform.expand_transform, targs, tkwargs))

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
        """Check if the transform program is informative or not.

        Returns:
            bool: Boolean
        """
        return self[-1].is_informative if self else False

    def __call__(self, tapes: Tuple[QuantumTape]) -> Tuple[ResultBatch, BatchPostProcessingFn]:
        if self.is_informative:
            raise NotImplementedError("Informative transforms are not yet supported.")

        if not self:
            return tapes, null_postprocessing
        processing_fns_stack = []

        for transform_container in self:
            transform, args, kwargs, cotransform, _ = transform_container
            if cotransform:
                raise NotImplementedError(
                    "cotransforms are not yet integrated with TransformProgram"
                )

            execution_tapes = []
            fns = []
            slices = []

            start = 0
            for tape in tapes:
                new_tapes, fn = transform(tape, *args, **kwargs)
                execution_tapes.extend(new_tapes)
                fns.append(fn)
                end = start + len(new_tapes)
                slices.append(slice(start, end))
                start = end

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

        return tuple(tapes), postprocessing_fn
