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
from typing import Callable, List, Tuple, Optional

from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape

from .transform_dispatcher import TransformContainer, TransformError

PostProcessingFn = Callable[[ResultBatch], Result]
BatchPostProcessingFn = Callable[[ResultBatch], ResultBatch]


def _batch_postprocessing(results: ResultBatch, individual_fns: PostProcessingFn) -> ResultBatch:
    """Broadcast individual post processing functions onto the their respective tapes.

    Args:
        results (ResultBatch): The numeric outcome from executing a batch of :class:`~.QuantumTape`

    Keyword Args:
        individual_fns (Callable): postprocessing functions converting a batch of results into a single result
           corresponding to only a single :class:`~.QuantumTape`.

    Note that this function does not perform validation on the sizes.

    If there are ``N`` ``individual_fns`` each one accepts a batch of ``M`` results, then the input ``results`` must be
    ``M*N`` long.

    >>> results = (1.0, 2.0, 3.0, 4.0)
    >>> def postprocessing1(results):
    ...     return results[0] + results[1]
    >>> def postprocessing2(results):
    ...     return results[0]+0.5
    >>> _batch_postprocessing(results, (postprocessing1, postprocessing2))
    (3.0, 3.5)

    """
    num_results = len(results)
    num_input_tapes = len(individual_fns)
    results_per_input_tape = num_results // num_input_tapes

    new_results = []
    for i, post_processing_fn in enumerate(individual_fns):
        selected_results = results[i * results_per_input_tape : (i + 1) * results_per_input_tape]
        new_results.append(post_processing_fn(selected_results))

    return tuple(new_results)


def _apply_postprocessing_stack(
    results: ResultBatch,
    postprocessing_stack: List[BatchPostProcessingFn],
    cotransform_stack: List[Optional[BatchPostProcessingFn]],
) -> ResultBatch:
    """Applies the postprocessing and cotransform postprocessing functions in a Last-In-First-Out LIFO manner.

    Args:
        results (ResultBatch): The numeric outcome from executing a batch of :class:`~.QuantumTape`

    Keyword Args:
        postprocessing_stack (List(BatchPostProcessingFn)): a LIFO stack of post processing functions.
        cotransform_stack (List(BatchPostProcessingFn)): a LIFO stack of classical cotransform functions.

    Returns:
        ResultBatch: the post processed results.

    >>> results = (1.0, 2.0, 3.0, 4.0)
    >>> def postprocessing1(results):
    ...     return (results[0] + results[1], results[2] + results[3])
    >>> def postprocessing2(results):
    .... return (results[0] + 1, results[1] + 2)
    >>> _apply_postprocessing_stack(results, [postprocessing1], [None, None])
    (3.0, 7.0)
    >>> _apply_postprocessing_stack(results, [postprocessing2, postprocessing1], [None, None])
    (4.0, 9.0)

    """
    for postprocessing, cotransform in zip(postprocessing_stack[::-1], cotransform_stack[::-1]):
        if cotransform:
            results = cotransform(results)
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

    def __init__(self, initial_program: Optional["TransformProgram"] = None):
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
        if not self.is_informative():
            raise TransformError("The transform program already has an informative transform.")
        self._transform_program.append(transform_container)

    def insert_front(self, transform_container: TransformContainer):
        """Insert the transform container at the beginning of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if transform_container.is_informative() and not self.is_empty():
            raise TransformError(
                "Informative transforms can only be added at the end of the program."
            )
        self._transform_program.insert(0, transform_container)

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
        if not self:
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

    def is_informative(self) -> bool:
        """Check if the transform program is informative or not.

        Returns:
            bool: Boolean
        """
        return self[-1].is_informative() if self else False

    def __call__(self, tapes: Tuple[QuantumTape]) -> Tuple[ResultBatch, BatchPostProcessingFn]:
        if not self:
            return tapes, null_postprocessing
        processing_fns_stack = []
        classical_cotransforms_stack = []

        for transform_container in self:
            transform, args, kwargs, cotransform, _ = transform_container

            execution_tapes = []
            fns = []

            for tape in tapes:
                new_tapes, fn = transform(tape, *args, **kwargs)
                execution_tapes.extend(new_tapes)
                fns.append(fn)

            batch_postprocessing = partial(_batch_postprocessing, inidividual_fns=fns)
            batch_postprocessing.__doc__ = _batch_postprocessing.__doc__

            processing_fns_stack.append(batch_postprocessing)
            classical_cotransforms_stack.append(cotransform)

            # set input tapes for next iteration.
            tapes = execution_tapes

        postprocessing_fn = partial(
            _apply_postprocessing_stack,
            postprocessing_stack=processing_fns_stack,
            cotransfrom_stack=classical_cotransforms_stack,
        )
        postprocessing_fn.__doc__ = _apply_postprocessing_stack.__doc__

        return tuple(tapes), postprocessing_fn
