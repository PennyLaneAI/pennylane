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
"""Code for the high-level quantum function transform that executes compilation."""

from inspect import getmembers, isclass

from pennylane import apply
from pennylane.tape import get_active_tape

from pennylane.transforms import qfunc_transform
from pennylane.transforms.optimization import cancel_inverses, commute_controlled, merge_rotations

from pennylane import templates

# Get a list of the names of existing templates in PennyLane
template_list = [cls[0] for cls in getmembers(templates, isclass)]


default_pipeline = [commute_controlled, cancel_inverses, merge_rotations]


@qfunc_transform
def compile(tape, pipeline=None, basis_set=None, num_passes=1):
    """Compile a circuit by applying a series of transforms to a quantum function.

    The default set of transforms includes (in order):
     - pushing all commuting single-qubit gates as far right as possible
       (:func:`~.pennylane.transforms.optimization.commute_controlled`)
     - cancellation of adjacent inverse gates
       (:func:`~.pennylane.transforms.optimization.cancel_inverses`)
     - merging adjacent rotations of the same type
       (:func:`~.pennylane.transforms.optimization.merge_rotations`)

    Args:
        qfunc (function): A quantum function.
        pipeline (list[qfunc_transform]): A list of quantum function transforms
            to apply.
        basis_set (list[str]): A list of basis gates. When expanding the tape,
            expansion will continue until gates in the specific set are
            reached. If no basis set is specified, no expansion will be done.
        num_passes (int): The number of times to repeat the set of operations in
            ``pipeline``.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x, y, z):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Hadamard(wires=2)
            qml.RZ(z, wires=2)
            qml.CNOT(wires=[2, 1])
            qml.RX(z, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.RX(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.RZ(-z, wires=2)
            qml.RX(y, wires=2)
            qml.PauliY(wires=2)
            qml.CY(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=0))

    Visually, the original function looks like this:

    >>> dev = qml.device('default.qubit', wires=[0, 1, 2])
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(0.2, 0.3, 0.4))
     0: ──H──RX(0.4)──────╭X─────────RX(0.2)──╭X───────┤ ⟨Z⟩
     1: ──H───────────╭X──╰C──────────────────╰C──╭CY──┤
     2: ──H──RZ(0.4)──╰C───RZ(-0.4)──RX(0.3)───Y──╰CY──┤

    We can compile it down to a smaller set of gates using the ``qml.compile``
    transform.

    >>> compiled_qfunc = qml.compile()(qfunc)
    >>> compiled_qnode = qml.QNode(compiled_qfunc, dev)
    >>> print(qml.draw(compiled_qnode)(0.2, 0.3, 0.4))
     0: ──H───RX(0.6)───────────────────┤ ⟨Z⟩
     1: ──H──╭X─────────────────╭CY─────┤
     2: ──H──╰C────────RX(0.3)──╰CY──Y──┤

    You can change up the set of transforms by passing a custom ``pipeline`` to
    ``qml.compile``. The pipeline is a list of transform functions. Furthermore,
    you can specify a number of passes (repetitions of the pipeline), and a list
    of gates into which the compiler will first attempt to decompose the
    existing operations prior to applying any optimization transforms.

    .. code-block:: python3

        compiled_qfunc = qml.compile(
            pipeline=[
                qml.transforms.commute_controlled(direction="left"),
                qml.transforms.merge_rotations(atol=1e-6),
                qml.transforms.cancel_inverses
            ],
            basis_set=["CNOT", "RX", "RY", "RZ"],
            num_passes=2
        )(qfunc)

        compiled_qnode = qml.QNode(compiled_qfunc, dev)

    >>> print(qml.draw(compiled_qnode)(0.2, 0.3, 0.4))
     0: ──RZ(1.57)──RX(1.57)──RZ(1.57)───RX(0.6)───────────────────────────────────────────────────────────────────────┤ ⟨Z⟩
     1: ──RZ(1.57)──RX(1.57)──RZ(1.57)──╭X────────RZ(1.57)──────────────────────────────────────────╭C─────────────╭C──┤
     2: ──RZ(1.57)──RX(1.57)──RZ(1.57)──╰C────────RX(0.3)───RZ(1.57)──RY(3.14)──RZ(1.57)──RY(1.57)──╰X──RY(-1.57)──╰X──┤


    """
    if pipeline is None:
        pipeline = default_pipeline

    # Expand the tape; this is done to unroll any templates that may be present,
    # as well as to decompose over a specified basis set
    # First, though, we have to stop whatever tape may be recording so that we
    # don't queue anything as a result of the expansion or transform pipeline
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
                expanded_tape = transform.tape_fn(expanded_tape)

    # Queue the operations on the optimized tape
    for op in expanded_tape.operations + expanded_tape.measurements:
        apply(op)
