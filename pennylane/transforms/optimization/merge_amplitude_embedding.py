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

from functools import lru_cache, partial

import pennylane as qml
from pennylane import AmplitudeEmbedding
from pennylane.math import flatten, reshape
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


# pylint: disable=too-many-statements
@lru_cache
def _get_plxpr_merge_amplitude_embedding():  # pylint: disable=missing-docstring
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.operation import Operator
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name
    class MergeAmplitudeEmbeddingInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for merging AmplitudeEmbedding gates when program capture is enabled."""

        def __init__(self):
            self._env = {}
            self.new_operations = []
            self.visited_wires = set()
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def setup(self) -> None:
            """Setup the interpreter for a new evaluation."""
            self.new_operations = []
            self.visited_wires = set()
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def cleanup(self) -> None:
            """Cleanup the interpreter after an evaluation."""
            self.new_operations = []
            self.input_wires, self.input_vectors, self.input_batch_size = [], [], []

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance.

            If the operator is not an ``AmplitudeEmbedding`` operator, it is added to the new operations list;
            otherwise, the wires and parameters are stored for future usage.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                None: returns None

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`.

            """
            if not isinstance(op, AmplitudeEmbedding):
                self.new_operations.append(op)
                self.visited_wires = self.visited_wires.union(set(op.wires))
                return

            if len(self.visited_wires.intersection(set(op.wires))) > 0:
                raise qml.DeviceError(
                    f"qml.AmplitudeEmbedding cannot be applied on wires already used by other operations."
                )

            self.input_wires.append(op.wires)
            self.input_vectors.append(op.parameters[0])
            self.input_batch_size.append(op.batch_size)
            self.visited_wires = self.visited_wires.union(set(op.wires))

        def purge_seen_operations(self):
            """Merge the gates and insert it at the beginning of the "seen" gates; then interpret said gates."""
            if len(self.input_wires) > 0:
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

                self.new_operations.insert(0, AmplitudeEmbedding(final_vector, wires=final_wires))

            for op in self.new_operations:
                super().interpret_operation(op)
            self.cleanup()

        # pylint: disable=too-many-branches
        def eval(self, jaxpr, consts, *args):
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
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
                primitive = eqn.primitive
                custom_handler = self._primitive_registrations.get(primitive, None)

                if getattr(primitive, "prim_type", "") == "higher_order":
                    self.purge_seen_operations()

                if custom_handler:
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif getattr(primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(primitive, "prim_type", "") == "measurement":
                    self.purge_seen_operations()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    extra_args, params = primitive.get_bind_params(eqn.params)
                    outvals = primitive.bind(*extra_args, *invals, **params)

                if not primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            self.purge_seen_operations()

            # Read the final result of the Jaxpr from the environment
            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, qml.operation.Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)
            self.cleanup()
            self._env = {}
            return outvals

    def merge_amplitude_embedding_plxpr_to_plxpr(jaxpr, consts, _, __, *args):
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
            raise qml.DeviceError(
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
        for w, v, b in zip(input_wires[1:], input_vectors[1:], input_batch_size[1:]):
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
