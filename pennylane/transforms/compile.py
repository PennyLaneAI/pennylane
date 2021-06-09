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
"""Code for a compilation transform."""

from inspect import getmembers, isclass
import pennylane.templates as templates

# Get a list of the names of existing templates in PennyLane
template_list = [cls[0] for cls in getmembers(templates, isclass)]

from pennylane.tape import get_active_tape

from pennylane.transforms import invisible, qfunc_transform, cancel_inverses, cnot_to_cz

default_pipeline = [cancel_inverses]


@qfunc_transform
def compile(tape, pipeline=default_pipeline, basis_set=None, num_passes=1):
    """Compile a circuit by applying a series of transforms to a quantum function.

    Args:
        tape (.QuantumTape): A quantum tape.
        pipeline (List[qfunc_transform]): A list of quantum function transforms to apply.
        basis_set (List[str]): A list of basis gates. When expanding the tape, expansion
            will continue until gates in the specific set are reached. If no basis set is
            specified, no expansion will be done.
        num_passes (int): The number of times to repeat the set of operations in pipeline.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def bell_state():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    Suppose we would like to apply the following set of transforms:

    >>> my_transforms = [cnot_to_cz, cancel_inverses]

    We can use this as a pipeline for the ``compile`` transform:

    >>> compiled_bell_state = compile(pipeline=my_transforms)(bell_state)

    Now we can run this, and the compiled version will be executed:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> qnode = qml.QNode(compiled_bell_state, dev)
    >>> print(qml.draw(qnode)())
    0: ──H──╭Z──┤ ⟨Z⟩
    1: ──H──╰C──┤
    """

    # Expand the tape; this is done to unroll any templates that may be present,
    # as well as to decompose us over a specified basis set
    # First, though, we have to stop whatever tape may be recording so that we
    # don't queue anything as a result of the expansion process
    current_tape = get_active_tape()
    with current_tape.stop_recording():
        if basis_set is not None:
            expanded_tape = tape.expand(depth=5, stop_at=lambda obj: obj.name in basis_set)
        else:
            # Expand only the templates
            expanded_tape = tape.expand(stop_at=lambda obj: obj.name not in template_list)

    # Apply the compilation transforms
    for _ in range(num_passes):
        for transform in pipeline:
            expanded_tape = invisible(transform.tape_fn)(expanded_tape)

    # Queue the operations on the optimized tape
    for op in expanded_tape.operations + expanded_tape.measurements:
        op.queue()
