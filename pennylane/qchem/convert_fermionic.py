# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the functions for converting an openfermion fermionic operator to Pennylane FermiWord or FermiSentence operators
"""

import openfermion as of
import pennylane as qml
from pennylane import fermi


def from_openfermion(of_fermi_op, tol=1e-16):
    r"""Convert openfermion ``FermionOperator`` objects to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` objects.

    Args:
        of_ferm_op (FermionOperator): OpenFermion fermionic operator
        tol (float): Tolerance for discarding coefficients

    Returns:
        Union[FermiWord, FermiSentence]: the fermionic operator object

    **Example**

    >>> of_fermi_op = 0.5 * of.FermionOperator('0^ 2') + of.FermionOperator('0 2^')
    >>> pl_fermi_op = qml.qchem.from_openfermion(of_fermi_op)
    >>> print(pl_ferm_op)
        0.5 * a(0) a⁺(2)
        + 1.0 * a⁺(0) a(2)
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion"
        ) from Error

    action = {"0": "-", "1": "+"}

    if len(of_fermi_op.terms) == 1 and list(of_fermi_op.terms.values())[0] == 1.0:
        fermi_str = " ".join(
            [str(operator[0]) + action[str(operator[1])] for operator in list(of_fermi_op.terms)[0]]
        )
        return fermi.from_string(fermi_str)

    fermi_pl = 0 * fermi.FermiWord({})
    for term in of_fermi_op.terms:
        fermi_str = " ".join([str(operator[0]) + action[str(operator[1])] for operator in term])
        fermi_pl += of_fermi_op.terms[term] * fermi.from_string(fermi_str)

    fermi_pl.simplify(tol=tol)
    return fermi_pl
