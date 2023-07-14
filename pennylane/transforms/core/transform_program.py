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
from .transform_dispatcher import TransformContainer, TransformError


class TransformProgram:
    """Class that contains a transform program and the methods to interact with it. The order of execution is the order
    in the list containing the containers.

    .. warning::

        This class is developer-facing and should not be used directly.

    .. seealso:: :func:`~.pennylane.transforms.core.transform`

    """

    def __init__(self):
        self._transform_program = []

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

    def __repr__(self):
        """The string representation of the transform program class."""
        repr = "TransformProgram("
        transforms_repr = ", ".join([f"{transform_c.transform.__name__}" for transform_c in self])
        end = ")"
        return repr + transforms_repr + end

    def push_back(self, transform_container: TransformContainer):
        """Add a transform (container) to the end of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
        if not isinstance(transform_container, TransformContainer):
            raise TransformError("Only transform container can be added to the transform program.")

        # Program can only contain one informative transform and at the end of the program
        if not self.is_empty() and self.get_last().is_informative:
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

    def pop_front(self):
        """Pop the transform container at the beginning of the program.

        Returns:
            TransformContainer: The transform container at the beginning of the program.
        """
        first_container = self._transform_program.pop(0)
        return first_container

    def get_last(self):
        """Get the last transform container.

        Returns:
            TransformContainer: The last transform in the program.

        Raises:
            TransformError: It raises an error if the program is empty.
        """
        if not self.is_empty():
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

    def is_informative(self):
        """Check if the transform program is informative or not.

        Returns:
            bool: Boolean, True if empty, False otherwise.
        """
        return self.get_last().is_informative

    def __call__(self, tapes):
        processing_fns_list = []
        classical_cotransforms_list = []

        for transform_container in self:
            num_tapes = len(tapes)
            transform, args, kwargs, cotransform, _ = transform_container

            execution_tapes = []
            fns = []

            for tape in tapes:
                new_tapes, fn = transform(tape, *args, **kwargs)
                execution_tapes.extend(new_tapes)
                fns.append(fn)

            new_num_tapes = len(new_tapes)

            # Merge the processing function into in a single one
            def processing_fn(
                res, num_tapes=num_tapes, new_num_tapes=new_num_tapes, fns=tuple(fns)
            ):
                final_results = [
                    fns[idx](res[idx * new_num_tapes : (idx + 1) * new_num_tapes])
                    for idx in range(num_tapes)
                ]
                return final_results

            processing_fns_list.append(processing_fn)

            # Merge the cotransform functions into in a single one
            if transform_container.classical_cotransform is None:
                classical_cotransforms_list.append(None)
            else:
                # TODO: temporary, to be replaced
                classical_cotransforms_list.append(cotransform)

            tapes = execution_tapes

        return tapes, processing_fns_list[::-1], classical_cotransforms_list[::-1]
