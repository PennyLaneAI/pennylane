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
# pylint: disable=too-many-branches
from functools import partial

from pennylane import apply
from pennylane.queuing import stop_recording
from pennylane.ops import __all__ as all_ops

from pennylane.transforms import single_tape_transform, qfunc_transform
from pennylane.transforms.optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    remove_barrier,
)


default_pipeline = [commute_controlled, cancel_inverses, merge_rotations, remove_barrier]


@qfunc_transform
def compile(tape, pipeline=None, basis_set=None, num_passes=1, expand_depth=5):
    """Compile a circuit by applying a series of transforms to a quantum function.

    The default set of transforms includes (in order):

    - pushing all commuting single-qubit gates as far right as possible
      (:func:`~pennylane.transforms.commute_controlled`)
    - cancellation of adjacent inverse gates
      (:func:`~pennylane.transforms.cancel_inverses`)
    - merging adjacent rotations of the same type
      (:func:`~pennylane.transforms.merge_rotations`)

    Args:
        qfunc (function): A quantum function.
        pipeline (list[single_tape_transform, qfunc_transform]): A list of
            tape and/or quantum function transforms to apply.
        basis_set (list[str]): A list of basis gates. When expanding the tape,
            expansion will continue until gates in the specific set are
            reached. If no basis set is specified, no expansion will be done.
        num_passes (int): The number of times to apply the set of transforms in
            ``pipeline``. The default is to perform each transform once;
            however, doing so may produce a new circuit where applying the set
            of transforms again may yield further improvement, so the number of
            such passes can be adjusted.
        expand_depth (int): When ``basis_set`` is specified, the depth to use
            for tape expansion into the basis gates.

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
    0: ──H──RX(0.40)────╭X──────────RX(0.20)─╭X────┤  <Z>
    1: ──H───────────╭X─╰●───────────────────╰●─╭●─┤
    2: ──H──RZ(0.40)─╰●──RZ(-0.40)──RX(0.30)──Y─╰Y─┤

    We can compile it down to a smaller set of gates using the ``qml.compile``
    transform.

    >>> compiled_qfunc = qml.compile()(qfunc)
    >>> compiled_qnode = qml.QNode(compiled_qfunc, dev)
    >>> print(qml.draw(compiled_qnode)(0.2, 0.3, 0.4))
    0: ──H──RX(0.60)─────────────────┤  <Z>
    1: ──H─╭X──────────────────╭●────┤
    2: ──H─╰●─────────RX(0.30)─╰Y──Y─┤

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

        print(qml.draw(compiled_qnode)(0.2, 0.3, 0.4))

    .. code-block::

        0: ──RZ(1.57)──RX(1.57)──RZ(1.57)──RX(0.60)─────────────────────────────────────────────────────
        1: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭X─────────RZ(1.57)─────────────────────────────────────────╭●
        2: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╰●─────────RX(0.30)──RZ(1.57)──RY(3.14)──RZ(1.57)──RY(1.57)─╰X

        ────────────────┤  <Z>
        ─────────────╭●─┤
        ───RY(-1.57)─╰X─┤

    """
    # Ensure that everything in the pipeline is a valid qfunc or tape transform
    if pipeline is None:
        pipeline = default_pipeline
    else:
        for p in pipeline:
            p_func = p.func if isinstance(p, partial) else p
            if not isinstance(p_func, single_tape_transform) and not hasattr(p_func, "tape_fn"):
                raise ValueError("Invalid transform function {p} passed to compile.")

    if num_passes < 1 or not isinstance(num_passes, int):
        raise ValueError("Number of passes must be an integer with value at least 1.")

    # Expand the tape; this is done to unroll any templates that may be present,
    # as well as to decompose over a specified basis set
    # First, though, we have to stop whatever tape may be recording so that we
    # don't queue anything as a result of the expansion or transform pipeline

    with stop_recording():
        if basis_set is not None:
            expanded_tape = tape.expand(
                depth=expand_depth, stop_at=lambda obj: obj.name in basis_set
            )
        else:
            # Expands out anything that is not a single operation (i.e., the templates)
            # expand barriers when `only_visual=True`
            def stop_at(obj):
                return (obj.name in all_ops) and (not getattr(obj, "only_visual", False))

            expanded_tape = tape.expand(stop_at=stop_at)

        # Apply the full set of compilation transforms num_passes times
        for _ in range(num_passes):
            for transform in pipeline:
                if isinstance(transform, (single_tape_transform, partial)):
                    expanded_tape = transform(expanded_tape)
                else:
                    expanded_tape = transform.tape_fn(expanded_tape)

    # Queue the operations on the optimized tape
    for op in expanded_tape.operations + expanded_tape.measurements:
        apply(op)
