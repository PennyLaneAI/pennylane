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
from typing import Tuple, Callable

from pennylane.typing import Result, ResultBatch

from .transform_dispatcher import TransformContainer, TransformError

from pennylane.transforms import map_batch_transform


class TransformProgram(list):
    """Class that contains a transform program and the methods to interact with it. The order of execution is the order
    in the list containing the containers.

    .. warning::

        This class is developer-facing and should not be used directly.

    .. seealso:: :func:`~.pennylane.transforms.core.transform`

    """

    def __repr__(self):
        """The string representation of the transform program class."""
        transforms_repr = ", ".join(f"{transform_c.transform.__name__}" for transform_c in self)
        return f"TransformProgram({transforms_repr})"

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        return object.__eq__(self, other)

    def __call__(self, circuits):

        post_processing_queue = []
        for transform_container in self:

            def transform_partial(tape):
                return transform_container.transform(
                    tape, *transform_container.args, **transform_container.kwargs
                )

            circuits, post_processing_fn = map_batch_transform(transform_partial, circuits)
            post_processing_queue.append(post_processing_fn)

        def total_post_processing_fn(results: ResultBatch) -> ResultBatch:
            for fn in post_processing_queue[::-1]:
                results = fn(results)
            return results

        return circuits, total_post_processing_fn
