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
This module contains the functions for converting an openfermion fermionic operator to Pennylane
FermiWord or FermiSentence operators.
"""
# pylint: disable= import-outside-toplevel,no-member,unused-import
from pennylane.fermi import FermiSentence, FermiWord


def from_openfermion(openfermion_op, tol=1e-16):
    r"""Convert OpenFermion
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__
    object to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` objects.

    Args:
        openfermion_op (FermionOperator): OpenFermion fermionic operator
        tol (float): tolerance for discarding negligible coefficients

    Returns:
        Union[FermiWord, FermiSentence]: the fermionic operator object

    **Example**

    >>> from openfermion import FermionOperator
    >>> openfermion_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = from_openfermion(openfermion_op)
    >>> print(pl_op)
        0.5 * a⁺(0) a(2)
        + 1.0 * a(0) a⁺(2)
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion"
        ) from Error

    typemap = {0: "-", 1: "+"}

    fermi_words = []
    fermi_coeffs = []

    for ops, val in openfermion_op.terms.items():
        fw_dict = {(i, op[0]): typemap[op[1]] for i, op in enumerate(ops)}
        fermi_words.append(FermiWord(fw_dict))
        fermi_coeffs.append(val)

    if len(fermi_words) == 1 and fermi_coeffs[0] == 1.0:
        return fermi_words[0]

    pl_op = FermiSentence(dict(zip(fermi_words, fermi_coeffs)))
    pl_op.simplify(tol=tol)

    return pl_op
