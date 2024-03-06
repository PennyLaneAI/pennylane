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
from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union

import numpy as np

import pennylane as qml
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
    """Class that contains a transform program and the methods to interact with it.

    The order of execution is the order in the list containing the containers.

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
    >>> program2 = copy.copy(program)
    >>> program2 == program
    True
    >>> qml.compile in program
    True
    >>> qml.transforms.hamiltonian_expand in program
    False
    >>> program + program
    TransformProgram(compile, cancel_inverses, compile, cancel_inverses)

    """

    def __init__(self, initial_program: Optional[Sequence] = None):
        self._transform_program = list(initial_program) if initial_program else []
        self._classical_jacobians = None
        self._argnums = None

    def __iter__(self):
        """list[TransformContainer]: Return an iterator to the underlying transform program."""
        return self._transform_program.__iter__()

    def __len__(self):
        """int: Return the number transforms in the program."""
        return len(self._transform_program)

    def __getitem__(self, idx) -> Union["TransformProgram", "TransformContainer"]:
        """(TransformContainer, List[TransformContainer]): Return the indexed transform container from underlying
        transform program"""
        if isinstance(idx, slice):
            return TransformProgram(self._transform_program[idx])
        return self._transform_program[idx]

    def __bool__(self):
        return bool(self._transform_program)

    def __add__(self, other):
        if self.has_final_transform and other.has_final_transform:
            raise TransformError("The transform program already has a terminal transform.")

        transforms = self._transform_program + other._transform_program
        if self.has_final_transform:
            transforms.append(transforms.pop(len(self) - 1))

        return TransformProgram(transforms)

    def __repr__(self):
        """The string representation of the transform program class."""
        contents = ", ".join(f"{transform_c.transform.__name__}" for transform_c in self)
        return f"TransformProgram({contents})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TransformProgram):
            return False

        return self._transform_program == other._transform_program

    def __contains__(self, obj):
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
            self.push_back(TransformContainer(transform.expand_transform, targs, tkwargs))
        self.push_back(
            TransformContainer(
                transform.transform,
                targs,
                tkwargs,
                transform.classical_cotransform,
                transform.is_informative,
                transform.final_transform,
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
                transform.transform,
                targs,
                tkwargs,
                transform.classical_cotransform,
                transform.is_informative,
                transform.final_transform,
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
        if not self.has_classical_cotransform():
            return
        hybrid = self[-1].kwargs.pop("hybrid", True)  # pylint: disable=no-member

        if hybrid:
            argnums = self[-1].kwargs.pop("argnums", None)  # pylint: disable=no-member
            self._set_all_classical_jacobians(qnode, args, kwargs, argnums)
            self._set_all_argnums(qnode, args, kwargs, argnums)

    def prune_dynamic_transform(self):
        """Ensure a single ``dynamic_one_shot`` transform is applied."""
        trans_type = np.zeros(len(self._transform_program), dtype=np.int32)
        for i, t in enumerate(self._transform_program):
            if "dynamic_one_shot" in str(t):
                trans_type[i] = 1
            if "mid_circuit_measurements" in str(t):
                trans_type[i] = 2
        if sum(trans_type) < 2:
            return
        keep = 2 if 2 in trans_type else 1
        found = False
        for i, ttype in enumerate(reversed(trans_type)):
            if not found and ttype == keep:
                found = True
                continue
            if found and ttype in [1, 2]:
                self._transform_program.pop(len(self._transform_program) - 1 - i)

    def _set_all_classical_jacobians(
        self, qnode, args, kwargs, argnums
    ):  # pylint: disable=too-many-statements
        """It can be called inside the QNode to get all the classical Jacobians for a gradient transform."""

        def classical_preprocessing(program, *args, **kwargs):
            """Returns the trainable gate parameters for a given QNode input."""
            kwargs.pop("shots", None)
            kwargs.pop("argnums", None)
            qnode.construct(args, kwargs)
            tape = qnode.qtape
            tapes, _ = program((tape,))
            res = tuple(qml.math.stack(tape.get_parameters(trainable_only=True)) for tape in tapes)
            if len(tapes) == 1:
                return res[0]
            return res

        def jacobian(classical_function, program, argnums, *args, **kwargs):
            indices = qml.math.get_trainable_indices(args)

            if qnode.interface in ["jax", "jax-jit"]:
                import jax  # pylint: disable=import-outside-toplevel

                if isinstance(args[0], jax.numpy.ndarray):
                    argnums = 0 if argnums is None else argnums

            if not indices and argnums is None:
                raise qml.QuantumFunctionError("No trainable parameters.")

            classical_function = partial(classical_function, program)

            if qnode.interface == "autograd":
                jac = qml.jacobian(classical_function, argnum=argnums)(*args, **kwargs)

            if qnode.interface == "tf":
                import tensorflow as tf  # pylint: disable=import-outside-toplevel

                def _jacobian(*args, **kwargs):
                    with tf.GradientTape() as tape:
                        gate_params = classical_function(*args, **kwargs)

                    jac = tape.jacobian(gate_params, args)
                    return jac

                jac = _jacobian(*args, **kwargs)

            if qnode.interface == "torch":
                import torch  # pylint: disable=import-outside-toplevel

                def _jacobian(*args, **kwargs):  # pylint: disable=unused-argument
                    jac = torch.autograd.functional.jacobian(classical_function, args)
                    return jac

                jac = _jacobian(*args, **kwargs)

            if qnode.interface in ["jax", "jax-jit"]:
                import jax  # pylint: disable=import-outside-toplevel

                argnums = 0 if argnums is None else argnums

                def _jacobian(*args, **kwargs):
                    return jax.jacobian(classical_function, argnums=argnums)(*args, **kwargs)

                jac = _jacobian(*args, **kwargs)

            return jac

        classical_jacobians = []
        for index, transform in enumerate(self):
            if transform.classical_cotransform:
                argnum = transform._kwargs.get("argnum", None)  # pylint: disable=protected-access
                if qnode.interface == "jax" and argnum:
                    raise qml.QuantumFunctionError(
                        "argnum does not work with the Jax interface. You should use argnums instead."
                    )
                sub_program = TransformProgram(self[0:index])
                classical_jacobian = jacobian(
                    classical_preprocessing, sub_program, argnums, *args, **kwargs
                )
                qnode.construct(args, kwargs)
                tapes, _ = sub_program((qnode.tape,))
                multi_tapes = len(tapes) > 1
                if not multi_tapes:
                    classical_jacobian = [classical_jacobian]
                classical_jacobians.append(classical_jacobian)
            else:
                classical_jacobians.append(None)
        self._classical_jacobians = classical_jacobians
        # Reset the initial tape
        qnode.construct(args, kwargs)

    def _set_all_argnums(self, qnode, args, kwargs, argnums):
        """It can be used inside the QNode to set all argnums (tape level) using argnums from the argnums at the QNode
        level.
        """

        argnums_list = []
        for index, transform in enumerate(self):
            argnums = [0] if qnode.interface in ["jax", "jax-jit"] and argnums is None else argnums
            # pylint: disable=protected-access
            if (transform._use_argnum or transform.classical_cotransform) and argnums:
                params = qml.math.jax_argnums_to_tape_trainable(
                    qnode, argnums, TransformProgram(self[0:index]), args, kwargs
                )
                argnums_list.append([qml.math.get_trainable_indices(param) for param in params])
            else:
                argnums_list.append(None)

        self._argnums = argnums_list

        qnode.construct(args, kwargs)

    def __call__(self, tapes: Tuple[QuantumTape]) -> Tuple[ResultBatch, BatchPostProcessingFn]:
        if not self:
            return tapes, null_postprocessing

        processing_fns_stack = []

        for i, transform_container in enumerate(self):
            transform, targs, tkwargs, cotransform, _, _ = transform_container

            execution_tapes = []
            fns = []
            slices = []

            classical_fns = []
            slices_classical = []

            start = 0
            start_classical = 0
            for j, tape in enumerate(tapes):
                if self._argnums is not None and self._argnums[i] is not None:
                    tape.trainable_params = self._argnums[i][j]
                new_tapes, fn = transform(tape, *targs, **tkwargs)
                execution_tapes.extend(new_tapes)

                fns.append(fn)
                end = start + len(new_tapes)
                slices.append(slice(start, end))
                start = end

                if cotransform and self._classical_jacobians:
                    classical_fns.append(
                        partial(cotransform, cjac=self._classical_jacobians[i][j], tape=tape)
                    )
                    slices_classical.append(slice(start_classical, start_classical + 1))
                    start_classical += 1

            if cotransform and self._classical_jacobians:
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
        self._classical_jacobians = []
        return tuple(tapes), postprocessing_fn
