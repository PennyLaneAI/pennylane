# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transforms for optimizing circuit in the framework for ZX calculus."""

import pennylane as qml
from pennylane.transforms import qfunc_transform
from pennylane.queuing import QueuingManager
from .converter import tape_to_graph_zx, graph_zx_to_tape


@qfunc_transform
def zx_optimization(tape):
    r"""Quantum function transform to optimize a circuit in the ZX calculus framework

    Args:
        qfunc (function): A quantum function to be optimized.

    Returns:
        function: the transformed quantum function

    **Example**

    .. code-block:: python

        def circuit(x):
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.PauliZ(wires=0)
            return qml.expval(qml.PauliZ(0))

    from pennylane.transforms.optimization.zx_optimization import zx_optimization

    optimized_qfunc = zx_optimization(circuit)

    qnode_opt = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(qnode_opt)(0.4))
    0: ──RZ(3.14)──H──RZ(3.14)─┤  <Z>
    """
    # Try import pyzx
    try:
        # pylint: disable=import-outside-toplevel
        import pyzx

    except ImportError as Error:
        raise ImportError(
            "This feature requires pyzx. It can be installed with: pip install pyzx"
        ) from Error

    # Save the measurements
    measurements = tape.measurements

    # Tape to graph
    with QueuingManager.stop_recording():
        zx_g = tape_to_graph_zx(tape)

    # Optimization pass
    pyzx.full_reduce(zx_g)

    # ZX graph -> extract circuit
    zx_simplified_c = pyzx.extract_circuit(zx_g)

    # ZX circuit -> tape
    graph_zx_to_tape(zx_simplified_c.to_graph())

    for m in measurements:
        qml.apply(m)
