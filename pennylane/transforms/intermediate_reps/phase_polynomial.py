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
"""Phase polynomial intermediate representation"""

from collections.abc import Sequence
from functools import partial

import numpy as np

from pennylane.tape import QuantumScript
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn, TensorLike


@partial(transform, is_informative=True)
def phase_polynomial(
    circ: QuantumScript, wire_order: Sequence | None = None
) -> tuple[tuple[TensorLike, TensorLike, TensorLike], PostprocessingFn]:
    r"""
    Compute the `Phase polynomial intermediate representation <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    for circuits consisting of :class:`~.CNOT` and :class:`~.RZ` gates.

    For a given ciruit, it produces the `parity matrix <https://pennylane.ai/compilation/parity-matrix-intermediate-representation>`__,
    the `parity table <https://pennylane.ai/compilation/parity-table>`__ and the associated angles,
    which make up the phase polynomial intermediate representation.

    Args:
        circ (QNode or QuantumScript or Callable): Quantum circuit containing only ``CNOT`` and ``RZ`` gates.
        wire_order (Iterable): Indicates how rows and columns should be ordered. If ``None`` is provided, uses the wires of the input circuit (``tape.wires``).

    Returns:
        tuple(TensorLike, TensorLike, TensorLike):
            A tuple consisting of the :func:`~parity_matrix`, parity table and corresponding angles for each parity.
            In the case of inputting a callable function, a new callable with the same call signature is returned (see :func:`pennylane.transform`).

    **Example**

    We look at the circuit in Figure 1 in `arXiv:2104.00934 <https://arxiv.org/abs/2104.00934>`__.

    .. code-block: python

        import pennylane as qml

        def circuit():
            qml.CNOT((1, 0))
            qml.RZ(1, 0)
            qml.CNOT((2, 0))
            qml.RZ(2, 0)
            qml.CNOT((0, 1))
            qml.CNOT((3, 1))
            qml.RZ(3, 1)

    >>> print(qml.draw(circuit)())
    0: ─╭X──RZ(1.00)─╭X──RZ(2.00)─╭●──────────────┤
    1: ─╰●───────────│────────────╰X─╭X──RZ(3.00)─┤
    2: ──────────────╰●──────────────│────────────┤
    3: ──────────────────────────────╰●───────────┤

    The phase polynomial representation consisting of the parity matrix, parity table and
    associated angles are computed by ``phase_polynomial``.

    >>> from pennylane.transforms import phase_polynomial
    >>> pmat, ptab, angles = phase_polynomial(circuit, wire_order=range(4))()
    >>> pmat
    array([[1, 1, 1, 0],
           [1, 0, 1, 1],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])
    >>> ptab
    array([[1, 1, 1],
           [1, 1, 0],
           [0, 1, 1],
           [0, 0, 1]])
    >>> angles
    array([1, 2, 3])

    .. details::
        :title: Usage Details

        We can go through explicitly reconstructing the output wavefunction.
        First, let us compute the exact wavefunction from the circuit.

        .. code-block:: python

            input = np.array([1, 1, 1, 1]) # computational basis state

            def comp_basis_to_wf(basis_state):
                basis_state = qml.BasisState(np.array(basis_state), range(4))
                return basis_state.state_vector().reshape(-1)

            input_wf = comp_basis_to_wf(input)
            output_wf = qml.matrix(tape, wire_order=range(4)) @ input_wf

        The output wavefunction is given by :math:`e^{2i} |1 1 1 1\rangle`, which we can confirm:

        >>> np.allclose(output_wf, np.exp(2j) * input_wf)
        True

        Note that the action of an :class:`~RZ` gate is given by

        .. math:: R_Z(\theta) |x\rangle = e^{-i \frac{\theta}{2} Z} |x\rangle = e^{-i \frac{\theta}{2} (1 - 2x)} |x\rangle

        Hence, we need to convert the collected parities :math:`\boldsymbol{x}` as :math:`-(1 - 2\boldsymbol{x})/2`, accordingly. In particular, the collected phase :math:`p(x)` is given by

        >>> output_phase = -(1 - 2 * ((input @ ptab) % 2))/2
        >>> output_phase = output_phase @ angles

        The final output wavefunction from the phase polynomial description is then given by the following.

        >>> output_wf_re = np.exp(1j * output_phase) * comp_basis_to_wf(pmat @ input % 2)

        We can compare it to the exact output wavefunction and see that they match:

        >>> np.allclose(output_wf_re, output_wf)
        True

    .. details::
        :title: Theory

        Phase polynomial circuits can be described by a phase polynomial :math:`p(\boldsymbol{x})` and a :func:`~parity_matrix` :math:`P` acting on a computational basis state :math:`|\boldsymbol{x}\rangle = |x_1, x_2, .., x_n\rangle` in the following way:

        .. math:: U |\boldsymbol{x}\rangle = e^{i p(\boldsymbol{x})} |P \boldsymbol{x}\rangle.

        Since the parity matrix :math:`P` is part of this description, :math:`p` and :math:`P` in conjunction are sometimes referred to as the phase polynomial intermediate representation (IR).

        The phase polynomial :math:`p(\boldsymbol{x})` is described in terms of its parity table :math:`P_T` and associated angles. For this, note that
        the action of a :class:`~RZ` gate onto a computational basis state :math:`|x\rangle` is given by

        .. math:: R_Z(\theta) |x\rangle = e^{-i \frac{\theta}{2} (1 - 2x)} |x\rangle.

        The parity table :math:`P_T` is made up of the `parities` :math:`\boldsymbol{x}` at the point in the circuit where the associated :class:`~RZ` gate is acting.
        To track the impact of the gate, we thus simply collect the current parity and remember the angle.
        Take for example the circuit ``[CNOT((0, 1)), RZ(theta, 1), CNOT((0, 1))]`` (read from left to right like a circuit diagram). We start in some arbitrary computational basis state
        ``x = [x1, x2]``. The first CNOT is transforming the input state to ``[x1, x1 ⊕ x2]``.
        For the action of ``RZ`` we remember the angle ``theta`` as well as the current parity ``x1 ⊕ x2`` on that wire.
        The second CNOT gate undoes the parity change and restores the original computational basis state ``[x1, x2]``.

        Hence, the parity matrix is simply the identity, but the parity table for the phase polynomial is ``P_T = [[x1 ⊕ x2]]`` (or ``[[1, 1]]``) together with the angle ``theta`` in the list of angles ``[theta]``.
        The computation of the circuit is thus simply

        .. math:: U |x_1, x_2\rangle = e^{-i \frac{\theta}{2} \left(1 - 2(x_1 \oplus x_2) \right)} |x_1, x_2\rangle

        The semantics of this function is roughly given by the following implementation:

        .. code-block:: python

            def compute_phase_polynomial(tape):
                wires = tape.wires
                parity_matrix = np.eye(len(wires), dtype=int)
                parity_table = []
                angles = []

                for op in tape.operations:

                    if op.name == "CNOT":
                        control, target = op.wires
                        parity_matrix[target] = (parity_matrix[target] + parity_matrix[control]) % 2

                    elif op.name == "RZ":
                        angles.append(op.data[0]) # append theta_i
                        # append _current_ parity (hence the copy)
                        parity_table.append(parity_matrix[op.wires[0]].copy())

                return parity_matrix, np.array(parity_table).T, angles

    """

    def postprocessing_fn(tapes):
        circ = tapes[0]

        wires = circ.wires
        w_order = wire_order

        if w_order is None:
            w_order = wires

        wire_map = {wire: idx for idx, wire in enumerate(w_order)}

        parity_matrix = np.eye(len(wire_map), dtype=int)
        parity_table = []
        angles = []
        i = 0
        for op in circ.operations:
            if op.name == "CNOT":
                control, target = op.wires
                parity_matrix[wire_map[target]] = (
                    parity_matrix[wire_map[target]] + parity_matrix[wire_map[control]]
                ) % 2
            elif op.name == "RZ":
                angles.append(op.data[0])  # append theta_i
                RZ_wire = wire_map[op.wires[0]]

                # append _current_ parity (hence the copy)
                parity_table.append(parity_matrix[RZ_wire].copy())
            else:
                raise TypeError(
                    f"phase_polynomial can only handle CNOT and RZ operators, received {op}"
                )

            i += 1

        return parity_matrix, np.array(parity_table).T, np.array(angles)

    return [circ], postprocessing_fn
