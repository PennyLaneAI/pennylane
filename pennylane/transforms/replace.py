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
"""Transform for applying custom decompositions of operations."""

from pennylane import apply
from pennylane.tape import get_active_tape
from pennylane.transforms import qfunc_transform, invisible


def _validate_decomposition_function(op, custom_ops):
    """Given a set of user-defined decompositions, check to make sure there
    are no interdependencies that would potentially cause infinite loops
    to occur. For example, we cannot have a decomposition that looks like this:

    RX(theta) -> Y RX(-theta) Y

    because the RX depends on itself. Similarly, if we have both decompositions:

        CNOT -> H CZ H
        CZ -> H CNOT H

    we would end up in an infinite loop as well.
    """

    # Need to use invisible, otherwise these ops will get queued; we can't simply
    # stop recording the tape while doing this, it seems to need an active queueing context.
    if op.num_params > 0:
        decomp_ops = invisible(custom_ops[op.name])(*op.parameters, op.wires)
    else:
        decomp_ops = invisible(custom_ops[op.name])(op.wires)

    decomp_op_names = [op.name for op in decomp_ops]

    # Check that no decompositions of gates depend on themselves
    if op.name in decomp_op_names:
        raise ValueError(
            f"Custom decomposition for operation {op.name} invalid. "
            "Decomposition depends on the operator itself."
        )

    for decomp_op in decomp_ops:
        # Check that none of the decompositions of the operators
        # in the decomposition depend on this operator. The decomposition may be custom,
        if decomp_op.name in custom_ops.keys():
            sub_decomp_op_names = [
                sub_op.name for sub_op in invisible(custom_ops[decomp_op.name])(decomp_op.wires)
            ]

            if op.name in sub_decomp_op_names:
                raise ValueError(
                    f"Decomposition for operation {op.name} invalid. "
                    "Decomposition of an operator in its decomposition depends on it."
                )


def _unroll(op, custom_ops):
    """Decompose an operation as much as possible given a set of custom decompositions."""

    # Get the initial decomposition
    # How the function gets called depends on whether the op is parametrized or not.
    if op.num_params > 0:
        op_list = custom_ops[op.name](*op.parameters, op.wires)
    else:
        op_list = custom_ops[op.name](op.wires)

    ops_with_decomps = list(custom_ops.keys())

    more_to_decompose = True

    while more_to_decompose:

        updated_op_list = []

        # Get the decomposition of each operation in the list
        for decomp_op in op_list:
            if decomp_op.name in ops_with_decomps:

                if decomp_op.num_params > 0:
                    updated_op_list.extend(
                        custom_ops[decomp_op.name](*decomp_op.parameters, decomp_op.wires)
                    )
                else:
                    updated_op_list.extend(custom_ops[decomp_op.name](decomp_op.wires))

            else:
                updated_op_list.append(decomp_op)

        # If any of those operations themselves have decompositions, loop through again
        if not any(
            name in ops_with_decomps for name in [decomp_op.name for decomp_op in updated_op_list]
        ):
            more_to_decompose = False

        op_list = updated_op_list.copy()

    return op_list


@qfunc_transform
def replace(tape, custom_ops=None, validate=True):
    r"""Quantum function transform capable of applying user-specific decompositions
    in place of specified gates.

    Args:
        qfunc (function): a quantum function
        custom_ops (dict[str : function]): a dictionary containing
            pairs of operator names and alternative decomposition functions.
        validate (bool): whether or not to perform validation of the provided
            decompositions. Note: this does **not** check matrix equivalence*,
            but rather ensures that there are no interdependencies in the
            provided decompositions that could lead to infinite loops during the
            replacement.

    Returns:
        function: the transformed quantum function

    **Example**

    Suppose that instead of applying the typical decomposition of a Hadamard in
    a circuit,

    .. math::

        H = RZ(\pi/2) RX(\pi/2) RZ(\pi/2)

    we would instead like to use the decomposition

    .. math::

        H = X \cdot RY(\pi/2)

    We can define a custom decomposition function for the Hadamard. The
    signature must match the signature of the original decomposition.

    .. code-block:: python3

        def custom_hadamard(wires):
            return [qml.RY(np.pi/2, wires=wires), qml.PauliX(wires=wires)]

    We can do likewise for other gates. To use the ``replace`` transform, we
    pass a dictionary containing the mapping from operator name to decomposition:

    >>> custom_ops = {"Hadamard" : custom_hadamard}

    Let's create a quantum function:

    .. code-block:: python3

        def qfunc(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=1)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliZ(wires=1))

    Now we can apply the transform:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = replace(custom_ops=custom_ops)(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> print(qml.draw(qnode)(0.3))
     0: ──RY(1.57)──X──╭C────────────────────────┤
     1: ───────────────╰X──RX(0.3)──RY(1.57)──X──┤ ⟨Z⟩
    """

    # If any custom decompositions are passed...
    if custom_ops is not None:

        validated_ops = []

        ops_with_custom_decomps = list(custom_ops.keys())

        # Get the current tape
        current_tape = get_active_tape()

        # Go through the list of operations
        for op in tape.operations:

            # If this operation has a custom decomposition, fully unroll it,
            # then queue it on the tape
            if op.name in ops_with_custom_decomps:

                # If desired, validate each operation the first time its encountered
                if validate and op.name not in validated_ops:
                    _validate_decomposition_function(op, custom_ops)
                    validated_ops.append(op.name)

                with current_tape.stop_recording():
                    fully_unrolled_ops = _unroll(op, custom_ops)

                # Apply the decomposed operation
                for unrolled_op in fully_unrolled_ops:
                    apply(unrolled_op)

            # Otherwise we just apply the operation as normal
            else:
                apply(op)

        for m in tape.measurements:
            apply(m)

    else:
        for op in tape.operations + tape.measurements:
            apply(op)
