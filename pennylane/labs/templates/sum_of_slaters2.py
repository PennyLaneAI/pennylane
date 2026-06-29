# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Contains the SumOfSlatersPrep2 template, a variant of qp.SumOfSlatersPrep that handles work
wires explicitly."""

import numpy as np

import pennylane as qp
from pennylane.core.wires import Wires
from pennylane.templates.state_preparations.sum_of_slaters import (
    _preprocess,
    _sos_state_prep_resources,
    _sos_state_prep_with_wires,
)


class SumOfSlatersPrep2(qp.SumOfSlatersPrep):
    r"""Prepare an arbitrary quantum state with the sum-of-Slaters technique.
    In contrast to :class:`~.SumOfSlatersPrep`, this operation handles work wires explicitly.

    This operation prepares an arbitrary state

    .. math:: |\psi\rangle = \sum_{\ell \in L } c_\ell |\ell\rangle,

    where :math:`L` denotes the set of ``indices`` and :math:`c_\ell` is the ``coefficient``
    corresponding to the index :math:`\ell\in L`.
    The state :math:`|\ell\rangle` is a computational basis state, interpreted via the
    binary representation of :math:`\ell`.

    This state preparation technique was introduced in Sec. III A of
    `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__
    and is tailored to sparse states.

    .. seealso:: :class:`~.SumOfSlatersPrep`

    Args:
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (~.WiresLike): Wires on which to prepare the state.
        enumeration_wires (~.WiresLike): Work wires used for the enumeration register. For
            :math:`d` entries in the state, :math:`\lceil \log_2 (d)\rceil` qubits are required.
        identification_wires (~.WiresLike): Work wires used for the identification register.
            The required number of qubits depends on the particular ``indices`` of the sparse state,
            but it is at most :math:`2d-1` for :math:`d` entries in the state.
        qrom_work_wires (~.WiresLike): Work wires used by the :class:`~.QROM` subroutine. For
            :math:`d` entries in the state, :math:`\lceil \log_2 (d)\rceil - 1` qubits are required.
        mcx_cache_wires (~.WiresLike): Work wires used for caching AND values when uncomputing
            the enumeration register with multicontrolled bit flips.
            The required number of qubits depends on the particular ``indices`` of the sparse state,
            but it is at most :math:`2d-2` for :math:`d` entries in the state.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.

    The required sizes for the numerous work wire registers can be computed with
    ``SumOfSlatersPrep2.required_register_sizes``.

    .. warning::

        Note that we require ``coefficients`` to be treated as numerical data in the form of an
        array, whereas the ``indices`` need to be hashable, and thus will be treated as static
        information. This is because ``indices`` significantly impacts the structure and size of
        the circuit that realizes the state preparation.

    **Example**

    Consider a sparse state specified by normalized coefficients and statevector
    indices pointing to the populated computational basis states:

    .. code-block:: python

        import pennylane as qp
        import numpy as np

        coefficients = np.array([1, -1j, 1j, 1, 1, -1j, 1, 1j]) / np.sqrt(8)
        indices = (0, 1, 2, 4, 8, 16, 32, 64)
        n = 7
        wires = list(range(n))

    This is all the information we require to create the state: ``coefficients``, ``indices``,
    and ``wires``. The ``indices`` correspond to the computational basis states interpreted
    via their binary representation (e.g., :math:`|3\rangle = |11\rangle` for two qubits
    or :math:`|3\rangle = |011\rangle` for three qubits).
    However, we also need to provide multiple sets of work wires. We can conveniently
    compute the required register sizes with ``SumOfSlatersPrep2.required_register_sizes``:

    .. code-block:: python

        from pennylane.labs.templates import SumOfSlatersPrep2

        sizes = SumOfSlatersPrep2.required_register_sizes(indices, n)

    >>> print(sizes)
    {'wires': 7, 'enumeration_wires': 3, 'identification_wires': 5, 'qrom_work_wires': 2, 'mcx_cache_wires': 4}

    Then, we can create the wire registers, here we use :func:`~.registers`, which allocates
    wires with consecutive integer labels.

    .. code-block:: python

        all_wires = qp.registers(sizes) # Includes the target wires

    >>> print(*all_wires.items(), sep="\n")
    ('wires', Wires([0, 1, 2, 3, 4, 5, 6]))
    ('enumeration_wires', Wires([7, 8, 9]))
    ('identification_wires', Wires([10, 11, 12, 13, 14]))
    ('qrom_work_wires', Wires([15, 16]))
    ('mcx_cache_wires', Wires([17, 18, 19, 20]))

    With our work wires set up, we can construct the state preparation circuit and check the
    prepared state for correctness:

    .. code-block:: python

        qp.decomposition.enable_graph()

        gate_set = {"QROM", "TemporaryAND", "Adjoint(TemporaryAND)", "MultiplexerStatePreparation", "CNOT", "X"}

        num_wires = sum(sizes.values())

        @qp.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            SumOfSlatersPrep2(coefficients, **all_wires, indices=indices)
            return qp.state()

    We can check that we prepared the right state:

    >>> prepared_state = circuit()[::2**14] # Slice the state, as there are 14 work wires
    >>> where = np.where(prepared_state)
    >>> print(where)
    (array([ 0,  1,  2,  4,  8, 16, 32, 64]),)
    >>> # Adding 0.0 to the rounded result will prevent stochastic signed zeros
    >>> print(np.round(prepared_state[where], 4) + 0.0)
    [0.3536+0.j     0.    -0.3536j 0.    +0.3536j 0.3536+0.j
     0.3536+0.j     0.    -0.3536j 0.3536+0.j     0.    +0.3536j]

    That looks exactly right! Internally, the state preparation looks like this:

    >>> print(qp.draw(circuit, show_matrices=False, max_length=195)())
     0: ──────────────────────────────────╭QROM(M1)─╭●───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     1: ──────────────────────────────────├QROM(M1)─│────────╭●──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     2: ──────────────────────────────────├QROM(M1)─│──╭●────│──╭●───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     3: ──────────────────────────────────├QROM(M1)─│──│─────│──│─────╭●─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     4: ──────────────────────────────────├QROM(M1)─│──│──╭●─│──│──╭●─│──╭●──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     5: ──────────────────────────────────├QROM(M1)─│──│──│──│──│──│──│──│──╭●───────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     6: ──────────────────────────────────├QROM(M1)─│──│──│──│──│──│──│──│──│──╭●────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     7: ─╭MultiplexerStatePreparation(M0)─├◑────────│──│──│──│──│──│──│──│──│──│─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ···
     8: ─├MultiplexerStatePreparation(M0)─├◑────────│──│──│──│──│──│──│──│──│──│───────────────────────────────────────────────────╭X────────────────────────────────╭X──────────────────────────── ···
     9: ─╰MultiplexerStatePreparation(M0)─├◑────────│──│──│──│──│──│──│──│──│──│─────────────────╭X────────────────────────────────│─────────────────────────────────│──╭X───────────────────────── ···
    10: ──────────────────────────────────│─────────╰X─╰X─╰X─│──│──│──│──│──│──│───X─╭●──────────│───────────────●╮────╭●──────────│───────────────●╮──X─╭●──────────│──│───────────────●╮──X─╭●─── ···
    11: ──────────────────────────────────│──────────────────╰X─╰X─╰X─│──│──│──│───X─├●──────────│───────────────●┤────├●──────────│───────────────●┤──X─├●──────────│──│───────────────●┤──X─├●─── ···
    12: ──────────────────────────────────│───────────────────────────╰X─╰X─│──│───X─│──╭●───────│───────────●╮───│────│──╭●───────│───────────●╮───│──X─│──╭●───────│──│───────────●╮───│────│──╭● ···
    13: ──────────────────────────────────│─────────────────────────────────╰X─│───X─│──│──╭●────│───────●╮───│───│──X─│──│──╭●────│───────●╮───│───│──X─│──│──╭●────│──│───────●╮───│───│────│──│─ ···
    14: ──────────────────────────────────│────────────────────────────────────╰X────│──│──│──╭●─│───●╮───│───│───│──X─│──│──│──╭●─│───●╮───│───│───│────│──│──│──╭●─│──│───●╮───│───│───│────│──│─ ···
    15: ──────────────────────────────────├work──────────────────────────────────────│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│─ ···
    16: ──────────────────────────────────╰work──────────────────────────────────────│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│─ ···
    17: ─────────────────────────────────────────────────────────────────────────────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────╰⊕─├● ···
    18: ────────────────────────────────────────────────────────────────────────────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯───────────╰⊕ ···
    19: ───────────────────────────────────────────────────────────────────────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯───────────────── ···
    20: ──────────────────────────────────────────────────────────────────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯───────────────────── ···
    <BLANKLINE>
     0: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╭●────────────────────────────┤ ╭State
     1: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│────────╭●───────────────────┤ ├State
     2: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│──╭●────│──╭●────────────────┤ ├State
     3: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│──│─────│──│─────╭●──────────┤ ├State
     4: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│──│──╭●─│──│──╭●─│──╭●───────┤ ├State
     5: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│──│──│──│──│──│──│──│──╭●────┤ ├State
     6: ··· ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│──│──│──│──│──│──│──│──│──╭●─┤ ├State
     7: ··· ───────╭X────────────────────────────────╭X───────────────────────────────────╭X───────────────────────────────────╭X───────────────────────│──│──│──│──│──│──│──│──│──│──┤ ├State
     8: ··· ───────│─────────────────────────────────│────────────────────────────────────│──╭X────────────────────────────────│──╭X────────────────────│──│──│──│──│──│──│──│──│──│──┤ ├State
     9: ··· ───────│─────────────────────────────────│──╭X────────────────────────────────│──│─────────────────────────────────│──│──╭X─────────────────│──│──│──│──│──│──│──│──│──│──┤ ├State
    10: ··· ───────│───────────────●╮──X─╭●──────────│──│───────────────●╮──X─╭●──────────│──│───────────────●╮──X─╭●──────────│──│──│───────────────●╮─╰X─╰X─╰X─│──│──│──│──│──│──│──┤ ├State
    11: ··· ───────│───────────────●┤──X─├●──────────│──│───────────────●┤────├●──────────│──│───────────────●┤──X─├●──────────│──│──│───────────────●┤──X───────╰X─╰X─╰X─│──│──│──│──┤ ├State
    12: ··· ───────│───────────●╮───│──X─│──╭●───────│──│───────────●╮───│────│──╭●───────│──│───────────●╮───│────│──╭●───────│──│──│───────────●╮───│──X────────────────╰X─╰X─│──│──┤ ├State
    13: ··· ─╭●────│───────●╮───│───│────│──│──╭●────│──│───────●╮───│───│────│──│──╭●────│──│───────●╮───│───│────│──│──╭●────│──│──│───────●╮───│───│──X──────────────────────╰X─│──┤ ├State
    14: ··· ─│──╭●─│───●╮───│───│───│────│──│──│──╭●─│──│───●╮───│───│───│────│──│──│──╭●─│──│───●╮───│───│───│────│──│──│──╭●─│──│──│───●╮───│───│───│──X─────────────────────────╰X─┤ ├State
    15: ··· ─│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──│────│───│───│───│───────────────────────────────┤ ├State
    16: ··· ─│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──│────│───│───│───│───────────────────────────────┤ ├State
    17: ··· ─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│──│────│───│──●┤──⊕╯───────────────────────────────┤ ├State
    18: ··· ─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│──│────│──●┤──⊕╯───────────────────────────────────┤ ├State
    19: ··· ─╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯──────────────────╰⊕─├●─│──│──│───●┤──⊕╯───────────────────────────────────────┤ ├State
    20: ··· ────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●─╰●──⊕╯───────────────────────────────────────────┤ ╰State

    Here, the first seven wires (``0`` to ``6``) are the target wires of the state preparation,
    wires ``7, 8, 9`` form the enumeration register, the next five wires (``10`` to ``14``)
    are the encoding register, and the pair of wires ``15, 16`` as well as the wires ``17``
    to ``20`` are work wires for the ``QROM`` and the enumeration uncomputation, respectively.

    .. details::
        :title: Usage details

        **Reduced circuit complexity for identity encodings**

        Depending on the ``indices`` passed to the state preparation, they may or may not
        be reducible to short enough sub-bitstrings such that no further encoding is required.
        In this case, the blocks of ``CNOT`` gates after the ``QROM`` and at the end,
        as seen in the example above, are not needed.
        For example, consider the following modification of the example:

        .. code-block:: python

            coefficients = np.array([0.25, 0.25j, -0.25, 0.5, 0.5, 0.25, -0.25j, 0.25, -0.25, 0.25])
            indices = (0, 1, 4, 13, 14, 17, 19, 22, 23, 25)
            n = 5
            wires = list(range(n))

            sizes = SumOfSlatersPrep2.required_register_sizes(indices, n)
            all_wires = qp.registers(sizes) # Includes the target wires

            num_wires = sum(sizes.values())

            @qp.decompose(gate_set=gate_set)
            @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
            def circuit():
                SumOfSlatersPrep2(coefficients, **all_wires, indices=indices)
                return qp.state()

        In this case, we only require eight work wires, because the encoding blocks can be skipped.

        >>> prepared_state = circuit()[::2**11] # Slice the state, as there are eleven work wires
        >>> where = np.where(np.abs(prepared_state)>1e-12)
        >>> print(where)
        (array([ 0,  1,  4, 13, 14, 17, 19, 22, 23, 25]),)
        >>> # Adding 0.0 to the rounded result will prevent stochastic signed zeros
        >>> print(np.round(prepared_state[where], 4) + 0.0)
        [ 0.25+0.j    0.  +0.25j -0.25+0.j    0.5 +0.j    0.5 +0.j    0.25+0.j
          0.  -0.25j  0.25+0.j   -0.25+0.j    0.25+0.j  ]

        The reduced circuit looks like this:

        >>> print(qp.draw(circuit, show_matrices=False, max_length=205)())
         0: ──────────────────────────────────╭QROM(M1)──X─╭●──────────────────────────●╮────╭●──────────────────────────●╮────╭●─────────────────────────────●╮────╭●──────────────────────────●╮──X─╭●───────── ···
         1: ──────────────────────────────────├QROM(M1)──X─├●──────────────────────────●┤────├●──────────────────────────●┤──X─├●─────────────────────────────●┤────├●──────────────────────────●┤──X─├●───────── ···
         2: ──────────────────────────────────├QROM(M1)──X─│──╭●───────────────────●╮───│──X─│──╭●───────────────────●╮───│────│──╭●──────────────────────●╮───│────│──╭●───────────────────●╮───│──X─│──╭●────── ···
         3: ──────────────────────────────────├QROM(M1)──X─│──│──╭●────────────●╮───│───│────│──│──╭●────────────●╮───│───│────│──│──╭●───────────────●╮───│───│──X─│──│──╭●────────────●╮───│───│──X─│──│──╭●─── ···
         4: ──────────────────────────────────├QROM(M1)────│──│──│──╭●─────●╮───│───│───│──X─│──│──│──╭●─────●╮───│───│───│──X─│──│──│──╭●────────●╮───│───│───│──X─│──│──│──╭●─────●╮───│───│───│──X─│──│──│──╭● ···
         5: ─╭MultiplexerStatePreparation(M0)─├◑───────────│──│──│──│───────│───│───│───│────│──│──│──│───────│───│───│───│────│──│──│──│──────────│───│───│───│────│──│──│──│───────│───│───│───│────│──│──│──│─ ···
         6: ─├MultiplexerStatePreparation(M0)─├◑───────────│──│──│──│───────│───│───│───│────│──│──│──│───────│───│───│───│────│──│──│──│──────────│───│───│───│────│──│──│──│──╭X───│───│───│───│────│──│──│──│─ ···
         7: ─├MultiplexerStatePreparation(M0)─├◑───────────│──│──│──│───────│───│───│───│────│──│──│──│──╭X───│───│───│───│────│──│──│──│──╭X──────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│─ ···
         8: ─╰MultiplexerStatePreparation(M0)─├◑───────────│──│──│──│──╭X───│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──╭X───│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│─ ···
         9: ──────────────────────────────────├work────────│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│─ ···
        10: ──────────────────────────────────├work────────│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│─ ···
        11: ──────────────────────────────────╰work────────│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│─ ···
        12: ───────────────────────────────────────────────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│─ ···
        13: ──────────────────────────────────────────────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│─ ···
        14: ─────────────────────────────────────────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯──────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├● ···
        15: ────────────────────────────────────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕ ···
        <BLANKLINE>
         0: ··· ────────────────────●╮────╭●─────────────────────────────●╮────╭●────────────────────────────────●╮────╭●──────────────────────────●╮────╭●─────────────────────────────●╮────┤ ╭State
         1: ··· ────────────────────●┤────├●─────────────────────────────●┤────├●────────────────────────────────●┤────├●──────────────────────────●┤──X─├●─────────────────────────────●┤────┤ ├State
         2: ··· ────────────────●╮───│────│──╭●──────────────────────●╮───│──X─│──╭●─────────────────────────●╮───│────│──╭●───────────────────●╮───│──X─│──╭●──────────────────────●╮───│──X─┤ ├State
         3: ··· ────────────●╮───│───│──X─│──│──╭●───────────────●╮───│───│────│──│──╭●──────────────────●╮───│───│────│──│──╭●────────────●╮───│───│──X─│──│──╭●───────────────●╮───│───│──X─┤ ├State
         4: ··· ────────●╮───│───│───│────│──│──│──╭●────────●╮───│───│───│──X─│──│──│──╭●───────────●╮───│───│───│──X─│──│──│──╭●─────●╮───│───│───│────│──│──│──╭●────────●╮───│───│───│────┤ ├State
         5: ··· ─────────│───│───│───│────│──│──│──│──────────│───│───│───│────│──│──│──│─────────────│───│───│───│────│──│──│──│──╭X───│───│───│───│────│──│──│──│──╭X──────│───│───│───│────┤ ├State
         6: ··· ─╭X──────│───│───│───│────│──│──│──│──╭X──────│───│───│───│────│──│──│──│──╭X─────────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│───────│───│───│───│────┤ ├State
         7: ··· ─│───────│───│───│───│────│──│──│──│──│──╭X───│───│───│───│────│──│──│──│──│──╭X──────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│───────│───│───│───│────┤ ├State
         8: ··· ─│──╭X───│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──╭X───│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──╭X───│───│───│───│────┤ ├State
         9: ··· ─│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────┤ ├State
        10: ··· ─│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────┤ ├State
        11: ··· ─│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────│──│──│──│──│──│──│────│───│───│───│────│──│──│──│──│────│───│───│───│────│──│──│──│──│──│────│───│───│───│────┤ ├State
        12: ··· ─│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│────│───│──●┤──⊕╯────╰⊕─├●─│──│──│──│────│───│──●┤──⊕╯────┤ ├State
        13: ··· ─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│────│──●┤──⊕╯───────────╰⊕─├●─│──│──│────│──●┤──⊕╯────────┤ ├State
        14: ··· ─│──│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯──────────────────╰⊕─├●─│──│──│───●┤──⊕╯──────────────────╰⊕─├●─│───●┤──⊕╯──────────────────╰⊕─├●─│──│───●┤──⊕╯────────────┤ ├State
        15: ··· ─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●─╰●──⊕╯─────────────────────────╰⊕─╰●──⊕╯─────────────────────────╰⊕─╰●─╰●──⊕╯────────────────┤ ╰State

        As we can see, the ladders of elbow, or :class:`~.TemporaryAND` gates are now
        controlled on the target register directly, rather than the encoding register, which
        we thus can skip for the identity encoding.

    """

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self):
        indices = self.hyperparameters["indices"]
        num_wires = len(self.wires)
        v_bits = qp.math.int_to_binary(np.array(indices), num_wires).T
        selector_ids, _ = qp.select_sos_rows(v_bits)
        return {"num_entries": len(indices), "num_bits": len(selector_ids), "num_wires": num_wires}

    def __init__(
        self,
        coefficients,
        wires,
        *_,
        enumeration_wires,
        identification_wires,
        qrom_work_wires,
        mcx_cache_wires,
        indices,
    ):  # pylint: disable=too-many-arguments
        super().__init__(coefficients, wires, indices)
        self.hyperparameters["enumeration_wires"] = Wires(enumeration_wires)
        self.hyperparameters["identification_wires"] = Wires(identification_wires)
        self.hyperparameters["qrom_work_wires"] = Wires(qrom_work_wires)
        self.hyperparameters["mcx_cache_wires"] = Wires(mcx_cache_wires)


@qp.register_resources(_sos_state_prep_resources, exact=False)
def _sos_state_prep(
    coefficients,
    *_,
    indices=None,
    **all_wires,
):  # pylint: disable=too-many-arguments, no-value-for-parameter, unused-argument
    """Compute the decomposition of the sum-of-Slaters state preparation technique."""
    n = len(all_wires["wires"])
    num_entries = len(indices)
    v_bits = qp.math.int_to_binary(np.array(indices), n).T  # Shape (n, num_entries)
    if num_entries == 1:
        qp.BasisState(v_bits[:, 0], wires=all_wires["wires"])
        return
    assert v_bits.shape == (n, num_entries)

    selected_wires, *data = _preprocess(v_bits, all_wires["wires"])
    all_wires["selected_wires"] = selected_wires
    data = (coefficients, v_bits, *data)
    _sos_state_prep_with_wires(data, **all_wires)


qp.add_decomps(SumOfSlatersPrep2, _sos_state_prep)
