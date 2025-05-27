# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains the set_shots transform, which sets the number of shots for a given tape.
"""
import warnings
from typing import Callable, Sequence, Union

from pennylane.exceptions import PennyLaneUserWarning
from pennylane.measurements import Shots
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


def null_postprocessing(results):
    """An empty post-processing function."""
    return results[0]


@transform
def set_shots(
    tape: QuantumScript, shots: Union[Shots, None, int, Sequence[Union[int, tuple[int, int]]]]
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transform used to set or update a circuit's shots.

    Args:
        tape (QuantumScript): The quantum circuit to be modified.
        shots (None or int or Sequence[int] or Sequence[tuple[int, int]] or pennylane.shots.Shots): The
            number of shots (or a shots vector) that the transformed circuit will execute.
            This specification will override any shots value previously associated
            with the circuit or QNode during execution.

    Returns:
        tuple[List[QuantumScript], function]: The transformed circuit as a batch of tapes and a
        post-processing function, as described in :func:`qml.transform <pennylane.transform>`. The output
        tape(s) will have their ``shots`` attribute set to the value provided in the ``shots`` argument.

    There are three ways to specify shot values (see :func:`qml.measurements.Shots <pennylane.measurements.Shots>` for more details):

    * The value ``None``: analytic mode, no shots
    * A positive integer: a fixed number of shots
    * A sequence consisting of either positive integers or a tuple-pair of positive integers of the form ``(shots, copies)``

    **Examples**

    Set the number of shots as a decorator:

    .. code-block:: python

        from functools import partial

        @partial(qml.set_shots, shots=2)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.sample(qml.Z(0))

    Run the circuit:

    >>> circuit()
    array([1., -1.])

    Update the shots in-line for an existing circuit:

    >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
    >>> new_circ()
    (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))

    """
    # Check if we're dealing with a QNode (using duck typing)
    is_qnode = hasattr(tape, "__call__") and hasattr(tape, "device")
    
    if is_qnode:
        # For QNodes, we need to create a wrapped function that warns about overrides
        # First, copy the QNode with the new shots
        qnode = tape.copy(shots=shots)
        
        # Store the original __call__ method
        orig_call = qnode.__call__
        
        # Create a wrapped function that checks for shots in kwargs
        def wrapped_call(*args, **kwargs):
            if "shots" in kwargs and kwargs["shots"] is not None:
                warnings.warn(
                    f"The shots value of the QNode is being overridden from {shots} to {kwargs['shots']}",
                    PennyLaneUserWarning,
                )
            return orig_call(*args, **kwargs)
        
        # Replace the __call__ method with our wrapped version
        qnode.__call__ = wrapped_call
        
        return (qnode,), null_postprocessing
    else:
        # Regular tape processing
        if hasattr(tape, "shots") and tape.shots != Shots(shots):
            tape = tape.copy(shots=shots)
        
        return (tape,), null_postprocessing
