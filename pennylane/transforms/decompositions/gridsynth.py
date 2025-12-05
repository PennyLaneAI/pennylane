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
"""Alias transform function for the Ross-Selinger decomposition (GridSynth) for qjit + capture."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="gridsynth")
def gridsynth(tape, *, epsilon, ppr_basis):
    r"""Decomposes RZ and PhaseShift gates into the Clifford+T basis or the PPR basis.

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``gridsynth`` compilation pass. Consult the Catalyst documentation for more information.

    Args:
        tape (QNode): A quantum circuit.
        epsilon (float): The maximum permissible operator norm error per rotation gate. Defaults to ``1e-4``.
        ppr_basis (bool): If True, decompose into the PPR basis. If False, decompose into the Clifford+T basis. Defaults to ``False``.

    **Example**

    .. code-block:: python

        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.Hadamard(0)
            qml.RZ(x, 0)
            qml.PhaseShift(x * 0.2, 0)
            return qml.state()

        result = circuit(1.1) # doctest: +SKIP
        gridsynth_circuit = qml.transforms.gridsynth(circuit, epsilon=1e-4) # doctest: +SKIP
        qjitted_circuit = qml.qjit(gridsynth_circuit) # doctest: +SKIP
        approx = qjitted_circuit(1.1) # doctest: +SKIP

    >>> result # doctest: +SKIP
    [0.60282587-0.36959568j 0.5076395 +0.49224195j]
    >>> approx # doctest: +SKIP
    [0.6028324 -0.3695921j  0.50763281+0.49224355j]


    .. warning::

        Using an ``epsilon`` value smaller than ``1e-7`` may lead to inaccurate results or errors,
        due to potential integer overflow in the solver.

    .. warning::

        Note: Simulating with ``ppr_basis=True`` is currently not supported.

    """

    raise NotImplementedError(  # pragma: no cover
        "This transform pass (gridsynth) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator. Otherwise, please use qml.transforms.clifford_t_decomposition."
    )
