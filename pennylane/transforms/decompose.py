# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
A transform for decomposing quantum circuits into user defined gate sets. Offers an alternative to the more device-focused decompose transform. 
"""
# pylint: disable=protected-access, too-many-arguments

import pennylane as qml
from pennylane.operation import StatePrepBase
from pennylane.tape import QuantumScript
from pennylane.transforms.core import transform

def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]

@transform
def decompose(tape, gate_set = None, max_expansion = None):
    """Decomposes operations found in a quantum circuit into a desired gate set. 

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit. 
        gate_set (Callable or set, optional): A set of decomposition gates or a condition they satisfy. Defaults to None.
        max_expansion (int, optional): The maximum depth of the expansion. Defaults to None.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumScript], function]:

        The decomposed circuit. The output type is explained in :func:`qml.transform <pennylane.transform>`.
    
    Raises:
        Exception: Type defaults to ``qml.DeviceError``. 
            Raised if an operator is not accepted and does not define a decomposition, or if
            the decomposition enters an infinite loop and raises a ``RecursionError``.

    .. seealso:: :func:`~.pennylane.devices.preprocess.decompose` for a transform that performs the same job but is designed particularly for specific device architectures. 

    **Examples:**
    
    >>> tape = qml.tape.QuantumScript([qml.IsingXX(1.2, wires=(0,1))], [qml.expval(qml.Z(0))])
    >>> batch, fn = decompose(tape, gate_set = {"CNOT", "RX", "RZ"})
    >>> batch[0].circuit
    [CNOT(wires=[0, 1]),
    RX(1.2, wires=[0]),
    CNOT(wires=[0, 1]),
    expval(Z(0))] 

    >>> @partial(decompose, gate_set = lambda obj: len(obj.wires) <= 2)
    >>> @qml.qnode(device)
    >>> def circuit():
    >>>     qml.Toffoli(wires = range(2))
    >>>
    >>> print(qml.draw(circuit)())
    0: ───────────╭●───────────╭●────╭●──T──╭●─┤  
    1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤  
    2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤  
    
    """
    if gate_set is None:
        gate_set = set(qml.ops.__all__) 
        
    if isinstance(gate_set, (list, tuple)):
        gate_set = set(gate_set) 
        
    def decomposer(op):
        return op.decomposition()

    def stopping_condition(op):
        if not isinstance(op, qml.operation.Operator):
            return True
        if not op.has_decomposition:
            return True
        if isinstance(gate_set, set):
            return op.name in gate_set
        else:
            return gate_set(op)

    if tape.operations and isinstance(tape[0], StatePrepBase): 
        prep_op = [tape[0]]
    else:
        prep_op = []

    if all(stopping_condition(op) for op in tape.operations[len(prep_op) :]):
        return (tape,), null_postprocessing

    try:
        new_ops = [
            final_op
            for op in tape.operations[len(prep_op) :]
            for final_op in qml.devices.preprocess._operator_decomposition_gen(
                op,
                stopping_condition,
                decomposer=decomposer,
                max_expansion=max_expansion,
                error=qml.operation.DecompositionUndefinedError,
            )
        ]
    except RecursionError as e:
        raise qml.DeviceError(
            "Reached recursion limit trying to decompose operations. "
            "Operator decomposition may have entered an infinite loop."
        ) from e
    
    tape = QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)

    return (tape,), null_postprocessing
