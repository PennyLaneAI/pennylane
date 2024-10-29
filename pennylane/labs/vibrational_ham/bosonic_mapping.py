import numpy as np
from functools import singledispatch
from typing import Union

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord

from .bosonic import BoseSentence, BoseWord

def get_pauli_op(i,j,qub_id):

    if i==0 and j==0:
        return(PauliSentence({PauliWord({}):0.5,
                             PauliWord({qub_id:'Z'}):0.5
                              }))
    elif i==0 and j==1:
        return(PauliSentence({PauliWord({qub_id:'X'}):0.5,
                             PauliWord({qub_id:'Y'}):0.5j
                             }))
    elif i==1 and j==0:
        return(PauliSentence({PauliWord({qub_id:'X'}):0.5,
                             PauliWord({qub_id:'Y'}):-0.5j
                              }))
    else:
        return(PauliSentence({PauliWord({}):0.5,
                             PauliWord({qub_id:'Z'}):-0.5
                              }))

def binary_mapping(bose_operator: Union[BoseWord, BoseSentence],
                   d: int = 2,
                   tol: float = None):
    r"""Convert a bosonic operator to a qubit operator using the standard-binary mapping.
    Args:
      bose_operator(BoseWord, BoseSentence): the bosonic operator
      d(int): Number of states in the boson.

    Returns:
      a linear combination of qubit operators
    
    """

    return _binary_mapping_dispatch(bose_operator, d, tol)

@singledispatch
def _binary_mapping_dispatch(bose_operator, d, tol):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")

@_binary_mapping_dispatch.register
def _(bose_operator: BoseWord, d, tol=None):
    nqub_per_boson = int(np.ceil(np.log2(d)))

    cr = np.zeros((d, d))
    for m in range(d - 1):
        cr[m + 1, m] = np.sqrt(m + 1.0)

    d_mat = {'+': cr, '-':cr.T}

    if len(bose_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})

    else:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})  # Identity PS to multiply PSs with
        for item in bose_operator.items():
            (_, boson), sign = item
            oper = PauliSentence({PauliWord({}): 0.0})  # Identity PS to multiply PSs with
            for i in range(d):
                for j in range(d):
                    if d_mat[sign][i][j] != 0:

                        coeff = d_mat[sign][i][j]
                        pauliOp =  PauliSentence({PauliWord({}): 1.0})
                        binary_row = list(map(int, bin(i)[2:]))[::-1]

                        if nqub_per_boson > len(binary_row):
                            binary_row += [0] * (nqub_per_boson - len(binary_row))
                        
                        binary_col = list(map(int, bin(j)[2:]))[::-1]
                        if nqub_per_boson > len(binary_col):
                            binary_col += [0] * (nqub_per_boson - len(binary_col))

                        for n in range(nqub_per_boson):

                            pauliOp @= get_pauli_op(binary_row[n], binary_col[n], n+boson*nqub_per_boson)

                        oper += coeff * pauliOp
            qubit_operator @= oper
    qubit_operator.simplify()
    return qubit_operator
               
@_binary_mapping_dispatch.register
def _(bose_operator: BoseSentence, d, tol=None):

    qubit_operator = PauliSentence()  # Empty PS as 0 operator to add Pws to

    for bw, coeff in bose_operator.items():
        bose_word_as_ps = binary_mapping(bw, d=d)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    return qubit_operator

