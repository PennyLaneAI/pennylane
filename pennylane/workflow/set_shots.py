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
This module contains the set_shots decorator.
"""
from typing import Sequence, Tuple, Union

from pennylane.measurements import Shots

from .qnode import QNode


def set_shots(
    qnode: QNode,
    *,
    shots: Union[Shots, None, int, Sequence[Union[int, Tuple[int, int]]]] = None,
):
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
    # When called directly with a function/QNode
    if isinstance(qnode, QNode):
        return qnode.update_shots(shots)
    raise ValueError("set_shots can only be applied to QNodes")
