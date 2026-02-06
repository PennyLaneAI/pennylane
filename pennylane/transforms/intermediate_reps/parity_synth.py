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
"""Alias for the ``parity_synth`` pass from Catalyst's passes module."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="parity-synth")
def parity_synth(tape):
    r"""Pass for applying ParitySynth to phase polynomials in a circuit.

    ParitySynth has been proposed by Vandaele et al. in `arXiv:2104.00934
    <https://arxiv.org/abs/2104.00934>`__ as a technique to synthesize
    `phase polynomials
    <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    into elementary quantum gates, namely ``CNOT`` and ``RZ``. For this, it synthesizes the
    `parity table <https://pennylane.ai/compilation/parity-table>`__ of the phase polynomial,
    and defers the remaining `parity matrix <https://pennylane.ai/compilation/parity-matrix>`__
    synthesis to `RowCol <https://pennylane.ai/compilation/rowcol-algorithm>`__.

    This pass walks over the input circuit and aggregates all ``CNOT`` and ``RZ`` operators
    into a subcircuit that describes a phase polyonomial. Other gates form the boundaries of
    these subcircuits, and whenever one is encountered the phase polynomial of the aggregated
    subcircuit is resynthesized with the ParitySynth algorithm. This implies that while this
    pass works on circuits containing any operations, it is recommended to maximize the
    subcircuits that represent phase polynomials (i.e. consist of ``CNOT`` and ``RZ`` gates) to
    enhance the effectiveness of the pass. This might be possible through decomposition or
    re-ordering of commuting gates.
    Note that higher-level program structures, such as nested functions and control flow, are
    synthesized independently, i.e., boundaries of such structures are always treated as boundaries
    of phase polynomial subcircuits as well. Similarly, dynamic wires create boundaries around the
    operations using them, potentially causing the separation of consecutive phase polynomial
    operations into multiple subcircuits.

    **Example**

    In the following, we apply the pass to a simple quantum circuit that has optimization
    potential in terms of commuting gates that can be interchanged to unlock a cancellation of
    a self-inverse gate (``CNOT``) with itself. Concretely, the circuit is:

    .. code-block:: python

        import pennylane as qp

        qml.capture.enable()
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: float, y: float, z: float):
            qml.CNOT((0, 1))
            qml.RZ(x, 1)
            qml.CNOT((0, 1))
            qml.RX(y, 1)
            qml.CNOT((1, 0))
            qml.RZ(z, 1)
            qml.CNOT((1, 0))
            return qml.state()

    We can draw the circuit and observe the last ``RZ`` gate being wrapped in a pair of ``CNOT``
    gates that commute with it:

    >>> print(qml.draw(circuit)(0.52, 0.12, 0.2))
    0: ─╭●───────────╭●───────────╭X───────────╭X─┤  State
    1: ─╰X──RZ(0.52)─╰X──RX(0.12)─╰●──RZ(0.20)─╰●─┤  State

    Now we apply the ``parity_synth_pass`` to the circuit and quantum just-in-time (qjit) compile
    the circuit into a reduced MLIR module:

    .. code-block:: python

        qjit_circuit = qml.qjit(qml.transforms.parity_synth(circuit))
        specs = qml.specs(qjit_circuit, level="device")(0.52, 0.12, 0.2)


    Looking at the resources of the compiled module, we find only five gates left in the program;
    the ``CNOT``\ s have been cancelled successfully.

    >>> print(specs.resources["gate_types"])
    {'RX': 1, 'RZ': 2, 'CNOT': 2}

    Note that for this circuit, ParitySynth is run twice; once
    for the first three gates and once for the last three gates. This is because ``RX`` is not
    a phase polynomial operation, so that it forms a boundary for the phase polynomial subcircuits
    that are re-synthesized by the pass.

    """
    raise NotImplementedError(
        "The parity_synth compilation pass has no tape implementation, and can only be applied "
        "when decorating the entire worfklow with @qml.qjit and when it is placed after all "
        "transforms that only have a tape implementation."
    )
