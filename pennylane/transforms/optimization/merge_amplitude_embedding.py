# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform for merging AmplitudeEmbedding gates in a quantum circuit."""

from collections.abc import Sequence
from copy import copy
from functools import lru_cache, partial

import pennylane as qml
from pennylane import AmplitudeEmbedding
from pennylane.exceptions import DeviceError, TransformError
from pennylane.math import flatten, is_abstract, reshape
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


# pylint: disable=too-many-statements
@lru_cache
def _get_plxpr_merge_amplitude_embedding():
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr
        from jax.extend.core import Jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.base_interpreter import jaxpr_to_jaxpr
        from pennylane.capture.primitives import cond_prim, measure_prim
        from pennylane.operation import Operator
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name
    class MergeAmplitudeEmbeddingInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for merging AmplitudeEmbedding gates when program capture is enabled."""

        def __init__(self):
            self._env = {}
            self.dynamic_wires_encountered = False
            self.previous_ops = []
            # * visited_wires (set): tracks all wires we have encountered so far.
            # * dynamic_wires_found (bool): True if we have encountered any non-AmplitudeEmbedding
            #   ops that have dynamic wires so far.
            # * ops_found (bool): True if we have encountered any non-AmplitudeEmbedding ops so far.
            self.state = {"visited_wires": set(), "dynamic_wires_found": False, "ops_found": False}
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def setup(self) -> None:
            """Setup the interpreter for a new evaluation."""
            self.previous_ops = []
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def cleanup(self) -> None:
            """Clean up the interpreter after evaluation."""
            self.state = {"visited_wires": set(), "dynamic_wires_found": False, "ops_found": False}

        def interpret_operation(self, op: Operator) -> None:
            """Interpret a PennyLane operation instance.

            If the operator is not an ``AmplitudeEmbedding`` operator, it is added to the new operations list;
            otherwise, the wires and parameters are stored for future usage.

            Args:
                op (Operator): a pennylane operator instance

            Raises:
                DeviceError: if the AmplitudeEmbedding operator's wires have already been used by other operations

            Returns:
                None: returns None

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            """

            if not isinstance(op, AmplitudeEmbedding):
                if any(is_abstract(w) for w in op.wires):
                    if self.input_wires:
                        self._merge_and_insert_at_the_start()
                    self.interpret_all_previous_ops()
                    self.state["dynamic_wires_found"] = True

                self.state["ops_found"] = True
                self.previous_ops.append(op)
                self.state["visited_wires"] = self.state["visited_wires"].union(set(op.wires))
                return

            if self.state["dynamic_wires_found"]:
                raise TransformError(
                    "Cannot apply qml.AmplitudeEmbedding after operators with dynamic wires as it "
                    "is indeterminable if the wires overlap."
                )

            if self.state["ops_found"] and any(is_abstract(w) for w in op.wires):
                raise TransformError(
                    "Cannot apply qml.AmplitudeEmbedding with dynamic wires after other operators "
                    "as it is indeterminable if the wires overlap."
                )

            if len(self.state["visited_wires"].intersection(set(op.wires))) > 0:
                raise TransformError(
                    "qml.AmplitudeEmbedding cannot be applied on wires already used by other operations."
                )

            self.input_wires.append(op.wires)
            self.input_vectors.append(op.parameters[0])
            self.input_batch_size.append(op.batch_size)
            self.state["visited_wires"] = self.state["visited_wires"].union(set(op.wires))

        def _merge_and_insert_at_the_start(self) -> None:
            """Merge the AmplitudeEmbedding gates and insert it at the beginning of the previously seen operations."""
            final_wires = self.input_wires[0]
            final_vector = self.input_vectors[0]
            final_batch_size = self.input_batch_size[0]

            for w, v, b in zip(
                self.input_wires[1:],
                self.input_vectors[1:],
                self.input_batch_size[1:],
                strict=True,
            ):
                final_vector = final_vector[..., :, None] * v[..., None, :]
                final_batch_size = final_batch_size or b
                final_wires = final_wires + w

                if final_batch_size:
                    final_vector = reshape(final_vector, (final_batch_size, -1))
                else:
                    final_vector = flatten(final_vector)

            with qml.capture.pause():
                self.previous_ops.insert(0, qml.AmplitudeEmbedding(final_vector, wires=final_wires))
            # Clear history of amplitude embedding gates since we've merged
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def interpret_all_previous_ops(self) -> None:
            """Interpret all previous operations and clear the setup variables."""
            for op in self.previous_ops:
                super().interpret_operation(op)
            self.previous_ops.clear()

        # pylint: disable=too-many-branches
        def eval(self, jaxpr: Jaxpr, consts: Sequence, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            self._env = {}
            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:
                custom_handler = self._primitive_registrations.get(eqn.primitive, None)
                prim_type = getattr(eqn.primitive, "prim_type", "")

                # Currently cannot merge through higher order primitives.
                # Workaround is to merge and insert the merged gate before entering
                # a higher order primitive.
                if prim_type == "higher_order":
                    if len(self.input_wires) > 0:
                        self._merge_and_insert_at_the_start()
                    self.interpret_all_previous_ops()

                if custom_handler:
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif prim_type == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif prim_type == "measurement":
                    if len(self.input_wires) > 0:
                        self._merge_and_insert_at_the_start()
                    self.interpret_all_previous_ops()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    extra_args, params = eqn.primitive.get_bind_params(eqn.params)
                    outvals = eqn.primitive.bind(*extra_args, *invals, **params)

                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            # The following is needed because any operations inside self.previous_ops have not yet
            # been applied.
            if len(self.input_wires) > 0:
                self._merge_and_insert_at_the_start()
            self.interpret_all_previous_ops()

            # Read the final result of the Jaxpr from the environment
            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)

            self.cleanup()
            self._env = {}
            return outvals

    # Overwrite the cond primitive so that visited wires can be correctly
    # detected across the different branches.
    @MergeAmplitudeEmbeddingInterpreter.register_primitive(cond_prim)
    def _(self, *invals, jaxpr_branches, consts_slices, args_slice):
        args = invals[args_slice]

        new_jaxprs = []
        new_consts = []
        new_consts_slices = []
        end_const_ind = len(jaxpr_branches)

        # Store state before we begin to process the branches
        # (create copies as to not accidently mutate the original state).
        # We cannot just copy self.state because a shallow copy would not
        # create a copy of `visited_wires`, which is a set.
        # We cannot use deepcopy as `visited_wires` may have tracers inside,
        # which have hashes specific to the instance. Copying these will cause
        # the dynamic wires in the original and copy to be different.
        initial_wires = copy(self.state["visited_wires"])
        curr_wires = copy(self.state["visited_wires"])
        initial_dynamic_wires_found = self.state["dynamic_wires_found"]
        curr_dynamic_wires_found = self.state["dynamic_wires_found"]
        initial_ops_found = self.state["ops_found"]
        curr_ops_found = self.state["ops_found"]

        for const_slice, jaxpr in zip(consts_slices, jaxpr_branches, strict=True):
            consts = invals[const_slice]
            new_jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)

            # Update state so far so collisions with
            # newly seen states from the branches continue to be
            # detected after the cond
            curr_wires |= self.state["visited_wires"]
            curr_dynamic_wires_found = curr_dynamic_wires_found or self.state["dynamic_wires_found"]
            curr_ops_found = curr_ops_found or self.state["ops_found"]

            # Reset state for the next branch so we don't get false positive collisions
            # (copy so if state mutates we preserved true initial state)
            self.state = {
                "visited_wires": copy(initial_wires),
                "dynamic_wires_found": initial_dynamic_wires_found,
                "ops_found": initial_ops_found,
            }

            new_jaxprs.append(new_jaxpr.jaxpr)
            new_consts.extend(new_jaxpr.consts)
            new_consts_slices.append(slice(end_const_ind, end_const_ind + len(new_jaxpr.consts)))
            end_const_ind += len(new_jaxpr.consts)

        # Reset state to all updates from all branches in the cond
        self.state = {
            "visited_wires": curr_wires,
            "dynamic_wires_found": curr_dynamic_wires_found,
            "ops_found": curr_ops_found,
        }

        new_args_slice = slice(end_const_ind, None)
        return cond_prim.bind(
            *invals[: len(jaxpr_branches)],
            *new_consts,
            *args,
            jaxpr_branches=new_jaxprs,
            consts_slices=new_consts_slices,
            args_slice=new_args_slice,
        )

    @MergeAmplitudeEmbeddingInterpreter.register_primitive(measure_prim)
    def _(self, *invals, **params):
        # Make sure to record that we have visited the wires on this measurement
        # in order to be able to detect potential wire collisions with future AE gates
        self.state["visited_wires"] = self.state["visited_wires"].union(set(invals))
        self.state["dynamic_wires_found"] = any(is_abstract(w) for w in invals)
        self.state["ops_found"] = True

        # pylint: disable=protected-access
        if len(self.input_wires) > 0:
            self._merge_and_insert_at_the_start()
        self.interpret_all_previous_ops()

        _, params = measure_prim.get_bind_params(params)
        return measure_prim.bind(*invals, **params)

    def merge_amplitude_embedding_plxpr_to_plxpr(jaxpr, consts, _, __, *args):
        """Function for applying the ``merge_amplitude_embedding`` transform on plxpr."""
        interpreter = MergeAmplitudeEmbeddingInterpreter()

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return MergeAmplitudeEmbeddingInterpreter, merge_amplitude_embedding_plxpr_to_plxpr


MergeAmplitudeEmbeddingInterpreter, merge_amplitude_embedding_plxpr_to_plxpr = (
    _get_plxpr_merge_amplitude_embedding()
)


@partial(transform, plxpr_transform=merge_amplitude_embedding_plxpr_to_plxpr)
def merge_amplitude_embedding(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to combine amplitude embedding templates that act on different qubits.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    **Example**

    >>> dev = qml.device('default.qubit', wires=4)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @qml.transforms.merge_amplitude_embedding
        @qml.qnode(device=dev)
        def circuit():
            qml.CNOT(wires = [0,1])
            qml.AmplitudeEmbedding([0,1], wires = 2)
            qml.AmplitudeEmbedding([0,1], wires = 3)
            return qml.state()

    >>> circuit()
    [1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

    .. details::
        :title: Usage Details

        You can also apply it on quantum function.

        .. code-block:: python

            def qfunc():
                qml.CNOT(wires = [0,1])
                qml.AmplitudeEmbedding([0,1], wires = 2)
                qml.AmplitudeEmbedding([0,1], wires = 3)
                return qml.state()

        The circuit before compilation will not work because of using two amplitude embedding.

        Using the transformation we can join the different amplitude embedding into a single one:

        >>> optimized_qfunc = qml.transforms.merge_amplitude_embedding(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)())
        0: ─╭●──────────────────────┤  State
        1: ─╰X──────────────────────┤  State
        2: ─╭AmplitudeEmbedding(M0)─┤  State
        3: ─╰AmplitudeEmbedding(M0)─┤  State
        M0 =
        [0.+0.j 0.+0.j 0.+0.j 1.+0.j]

    """
    new_operations = []
    visited_wires = set()
    input_wires, input_vectors, input_batch_size = [], [], []
    for current_gate in tape.operations:
        wires_set = set(current_gate.wires)

        # Check if the current gate is an AmplitudeEmbedding.
        if not isinstance(current_gate, AmplitudeEmbedding):
            new_operations.append(current_gate)
            visited_wires = visited_wires.union(wires_set)
            continue

        # Check the qubits have not been used.
        if len(visited_wires.intersection(wires_set)) > 0:
            raise DeviceError(
                f"Operation {current_gate.name} cannot be used after other Operation applied in the same qubit "
            )
        input_wires.append(current_gate.wires)
        input_vectors.append(current_gate.parameters[0])
        input_batch_size.append(current_gate.batch_size)
        visited_wires = visited_wires.union(wires_set)

    if len(input_wires) > 0:
        final_wires = input_wires[0]
        final_vector = input_vectors[0]
        final_batch_size = input_batch_size[0]

        # Merge all parameters and qubits into a single one.
        for w, v, b in zip(input_wires[1:], input_vectors[1:], input_batch_size[1:], strict=True):
            final_vector = final_vector[..., :, None] * v[..., None, :]
            final_batch_size = final_batch_size or b
            final_wires = final_wires + w

            if final_batch_size:
                final_vector = reshape(final_vector, (final_batch_size, -1))
            else:
                final_vector = flatten(final_vector)

        with QueuingManager.stop_recording():
            new_operations.insert(0, AmplitudeEmbedding(final_vector, wires=final_wires))

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
