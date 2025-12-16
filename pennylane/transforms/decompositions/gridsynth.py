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
"""Alias transform function for the Ross-Selinger decomposition (GridSynth) for qjit."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="gridsynth")
def gridsynth(tape, *, epsilon=1e-4, ppr_basis=False):
    r"""Decomposes RZ and PhaseShift gates into the Clifford+T basis or the PPR basis.

    .. warning::

        This transform must be applied within a workflow compiled with :func:`pennylane.qjit`,
        as it is a frontend for Catalyst's ``gridsynth`` compilation pass.
        Consult the Catalyst documentation for more information.

    Args:
        tape (QNode): A quantum circuit.
        epsilon (float): The maximum permissible operator norm error per rotation gate. Defaults to ``1e-4``.
        ppr_basis (bool): If True, decompose into the PPR basis. If False, decompose into the Clifford+T basis. Defaults to ``False``.

    **Example**

    .. code-block:: python

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.Hadamard(0)
            qml.RZ(x, 0)
            qml.PhaseShift(x * 0.2, 0)
            return qml.state()

        gridsynth_circuit = qml.transforms.gridsynth(circuit, epsilon=1e-4)
        qjitted_circuit = qml.qjit(gridsynth_circuit)

    >>> circuit(1.1) # doctest: +SKIP
    [0.60282587-0.36959568j 0.5076395 +0.49224195j]
    >>> qjitted_circuit(1.1) # doctest: +SKIP
    [0.6028324 -0.3695921j  0.50763281+0.49224355j]


    .. warning::

        Using an ``epsilon`` value smaller than ``1e-7`` may lead to inaccurate results or errors,
        due to potential integer overflow in the solver.

    .. warning::

        Note: Simulating with ``ppr_basis=True`` is currently not supported.

    """

    raise NotImplementedError(
        "The gridsynth compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )
