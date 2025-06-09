# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for Cartan decomposition"""
# pylint: disable= missing-function-docstring
from functools import partial

import numpy as np

from pennylane import QubitUnitary
from pennylane.liealg import cartan_decomp, check_cartan_decomp
from pennylane.liealg.involutions import int_log2

IDENTITY = object()


def pauli_y_eigenbasis(wire, num_wires):
    V = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    return QubitUnitary(V, wire).matrix(wire_order=range(num_wires))


_basis_change_constructors = {
    ("AI", "BDI"): IDENTITY,
    ("AI", "DIII"): IDENTITY,
    ("AII", "CI"): IDENTITY,
    ("AII", "CII"): IDENTITY,
    ("AIII", "A"): IDENTITY,
    ("BDI", "BD"): IDENTITY,
    ("CI", "AI"): pauli_y_eigenbasis,
    ("CI", "AII"): pauli_y_eigenbasis,
    ("CI", "AIII"): IDENTITY,
    ("CII", "C"): IDENTITY,
    ("DIII", "AI"): pauli_y_eigenbasis,
    ("DIII", "AII"): pauli_y_eigenbasis,
    ("DIII", "AIII"): pauli_y_eigenbasis,
    ("A", "AI"): IDENTITY,
    ("A", "AII"): IDENTITY,
    ("A", "AIII"): IDENTITY,
    ("BD", "BDI"): IDENTITY,
    ("BD", "DIII"): IDENTITY,
    ("C", "CI"): IDENTITY,
    ("C", "CII"): IDENTITY,
}


def _check_abcd_sequence(before, current, after):
    if before == "AIII" and current == "A" and after in ("AI", "AII", "AIII"):
        return
    if before == "BDI" and current == "BD" and after in ("BDI", "DIII"):
        return
    if before == "CII" and current == "C" and after in ("CI", "CII"):
        return
    raise ValueError(
        f"The 3-sequence ({before}, {current}, {after}) of involutions is not a valid sequence."
    )


def _check_chain(chain, num_wires):
    """Validate a chain of involutions for a recursive Cartan decomposition."""
    # Take the function name or its `func` attribute if it exists (e.g., for `partial` of an involution)
    names = [getattr(phi, "func", phi).__name__ for phi in chain]

    # Assume some standard behaviour regarding the wires on which we need to perform basis changes
    basis_changes = []
    wire = 0
    for i, name in enumerate(names[:-1]):
        invol_pair = (name, names[i + 1])
        if invol_pair not in _basis_change_constructors:
            raise ValueError(
                f"The specified chain contains the pair {'-->'.join(invol_pair)}, "
                "which is not a valid pair."
            )
        # Run specific check for sequence of three involutions where A, BD or C is the middle one
        if name in ("A", "BD", "C") and i > 0:
            _check_abcd_sequence(names[i - 1], name, names[i + 1])
        bc_constructor = _basis_change_constructors[invol_pair]
        if bc_constructor is IDENTITY:
            bc = bc_constructor
        else:
            bc = bc_constructor(wire, num_wires)
            # Next assumption: The wire is only incremented if a basis change is applied.
            wire += 1
        basis_changes.append(bc)

    basis_changes.append(IDENTITY)  # Do not perform any basis change after last involution.
    return names, basis_changes


def _apply_basis_change(change_op, targets):
    """Helper function for recursive Cartan decompositions that applies a basis change matrix
    ``change_op`` to a batch of matrices ``targets`` (with leading batch dimension)."""
    # Compute x V^\dagger for all x in ``targets``. ``moveaxis`` brings the batch axis to the front
    out = np.moveaxis(np.tensordot(change_op, targets, axes=[[1], [1]]), 1, 0)
    out = np.tensordot(out, change_op.conj().T, axes=[[2], [0]])
    return out


def recursive_cartan_decomp(g, chain, validate=True, verbose=True):
    r"""Apply a recursive Cartan decomposition specified by a chain of decomposition types.
    The decompositions will use canonical involutions and hardcoded basis transformations
    between them in order to obtain a valid recursion.

    This function tries to make sure that only sensible involution sequences are applied,
    and to raise an error otherwise. However, the involutions still need to be configured
    properly, regarding the wires their conjugation operators act on.

    Args:
        g (tensor_like): Basis of the algebra to be decomposed.
        chain (Iterable[Callable]): Sequence of involutions. Each callable should be
            one of
            :func:`~.pennylane.liealg.A`,
            :func:`~.pennylane.liealg.AI`,
            :func:`~.pennylane.liealg.AII`,
            :func:`~.pennylane.liealg.AIII`,
            :func:`~.pennylane.liealg.BD`,
            :func:`~.pennylane.liealg.BDI`,
            :func:`~.pennylane.liealg.DIII`,
            :func:`~.pennylane.liealg.C`,
            :func:`~.pennylane.liealg.CI`,
            :func:`~.pennylane.liealg.CII`,
            or a partial evolution thereof.
        validate (bool): Whether or not to verify that the involutions return a subalgebra.
        verbose (bool): Whether or not to print status updates during the computation.

    Returns:
        dict: The decompositions at each level. The keys are (zero-based) integers for the
        different levels of the recursion, the values are tuples ``(k, m)`` with subalgebra
        ``k`` and horizontal space ``m``. For each level, ``k`` and ``m`` combine into
        ``k`` from the previous recursion level.

    **Examples**

    Let's set up the special unitary algebra on 2 qubits. Note that we are using the Hermitian
    matrices that correspond to the skew-Hermitian algebra elements via multiplication
    by :math:`i`. Also note that :func:`~.pauli.pauli_group` returns the identity as the first
    element, which is not part of the special unitary algebra of traceless matrices.

    >>> g = [qml.matrix(op, wire_order=range(2)) for op in qml.pauli.pauli_group(2)] # u(4)
    >>> g = g[1:] # Remove identity: u(4) -> su(4)

    Now we can apply Cartan decompositions of type AI and DIII in sequence:

    >>> from pennylane.labs.dla import recursive_cartan_decomp
    >>> from pennylane.liealg import AI, DIII
    >>> chain = [AI, DIII]
    >>> decompositions = recursive_cartan_decomp(g, chain)
    Iteration 0:   15 -----AI---->    6,   9
    Iteration 1:    6 ----DIII--->    4,   2

    The function prints the progress of the decompositions by default, which can be deactivated by
    setting ``verbose=False``. Here we see how the initial :math:`\mathfrak{g}=\mathfrak{su(4)}`
    was decomposed by AI into the six-dimensional :math:`\mathfrak{k}_1=\mathfrak{so(4)}` and a
    horizontal space of dimension nine. Then, :math:`\mathfrak{k}_1` was further decomposed
    by the DIII decomposition into the four-dimensional :math:`\mathfrak{k}_2=\mathfrak{u}(2)`
    and a two-dimensional horizontal space.

    In a more elaborate example, let's apply a chain of decompositions AII, CI, AI, BDI, and DIII
    to the four-qubit unitary algebra. While we took care of the global phase term of :math:`u(4)`
    explicitly above, we leave it in the algebra here, and see that it does not cause problems.
    We discuss the ``wire`` keyword argument below.

    >>> from pennylane.liealg import AII, CI, BD, BDI
    >>> from functools import partial
    >>> chain = [
    ...     AII,
    ...     CI,
    ...     AI,
    ...     partial(BDI, wire=1),
    ...     partial(BD, wire=1),
    ...     partial(DIII, wire=2),
    ... ]
    >>> g = [qml.matrix(op, wire_order=range(4)) for op in qml.pauli.pauli_group(4)] # u(16)
    >>> decompositions = recursive_cartan_decomp(g, chain)
    Iteration 0:  256 ---AII--->  136, 120
    Iteration 1:  136 ----CI--->   64,  72
    Iteration 2:   64 ----AI--->   28,  36
    Iteration 3:   28 ---BDI--->   12,  16
    Iteration 4:   12 ----BD--->    6,   6
    Iteration 5:    6 ---DIII-->    4,   2

    The obtained chain of algebras is

    .. math::

        \mathfrak{u}(16)
        \rightarrow \mathfrak{sp}(8)
        \rightarrow \mathfrak{u}(8)
        \rightarrow \mathfrak{so}(8)
        \rightarrow \mathfrak{so}(4) \oplus \mathfrak{so}(4)
        \rightarrow \mathfrak{so}(4)
        \rightarrow \mathfrak{u}(2).

    What about the wire keyword argument to the used involutions?
    A good rule of thumb is that it should start at ``0`` and increment by one every second
    involution. For the involution :func:`~.pennylane.liealg.CI` it should additionally be
    increased by one. As ``0`` is the default for ``wire``, it usually does not have to be
    provided explicitly for the first two involutions, unless ``CI`` is among them.

    .. note::

        A typical effect of setting the wire wrongly is that the decomposition does not
        split the subalgebra from the previous step but keeps it intact and returns a
        zero-dimensional horizontal space. For example:

        >>> g = [qml.matrix(op, wire_order=range(2)) for op in qml.pauli.pauli_group(2)] # u(4)
        >>> chain = [AI, DIII, AII]
        >>> decompositions = recursive_cartan_decomp(g, chain)
        Iteration 0:   16 -----AI---->    6,  10
        Iteration 1:    6 ----DIII--->    4,   2
        Iteration 2:    4 ----AII---->    4,   0

        We see that the ``AII`` decomposition did not further decompose :math:`\mathfrak{u}(2)`.
        It works if we provide the correct ``wire`` argument:

        >>> chain = [AI, DIII, partial(AII, wire=1)]
        >>> decompositions = recursive_cartan_decomp(g, chain)
        Iteration 0:   16 -----AI---->    6,  10
        Iteration 1:    6 ----DIII--->    4,   2
        Iteration 2:    4 ----AII---->    3,   1

        We obtain :math:`\mathfrak{sp}(1)` as expected from the decomposition of type AII.

    """

    # Prerun the validation by obtaining the required basis changes and raising an error if
    # an invalid pair is found.
    num_wires = int_log2(np.shape(g)[-1])
    names, basis_changes = _check_chain(chain, num_wires)

    decompositions = {}
    for i, (phi, name, bc) in enumerate(zip(chain, names, basis_changes)):
        try:
            k, m = cartan_decomp(g, phi)
        except ValueError as e:
            if "please specify p and q for the involution" in str(e):
                phi = partial(phi, p=2 ** (num_wires - 1), q=2 ** (num_wires - 1))
                k, m = cartan_decomp(g, phi)
            else:
                raise ValueError from e

        if validate:
            check_cartan_decomp(k, m, verbose=verbose)
        if verbose:
            print(f"Iteration {i}: {len(g):>4} -{name:-^8}> {len(k):>4},{len(m):>4}")
        decompositions[i] = (k, m)
        if not bc is IDENTITY:
            k = _apply_basis_change(bc, k)
            m = _apply_basis_change(bc, m)
        g = k

    return decompositions
