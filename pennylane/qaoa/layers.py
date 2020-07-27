# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Methods that define cost and mixer layers for use in QAOA workflows.
"""
import pennylane as qml
from pennylane.templates import ApproxTimeEvolution
from pennylane.operation import Tensor


def _diagonal_terms(hamiltonian):
    r"""Checks if all terms in a Hamiltonian are products of diagonal Pauli gates (PauliZ and Identity).

    Args:
        hamiltonian (qml.Hamiltonian): The Hamiltonian being checked

    Returns:
        bool: ``True`` if all terms are products of diagonal Pauli gates, ``False`` otherwise
    """
    val = True

    for i in hamiltonian.ops:
        i = Tensor(i) if isinstance(i.name, str) else i
        for j in i.obs:
            if j.name != "PauliZ" and j.name != "Identity":
                val = False

    return val


def cost_layer(hamiltonian):
    r"""Builds a QAOA cost layer, for a given cost Hamiltonian.

    The cost layer for cost Hamiltonian :math:`H_C` is defined as the following unitary:

    .. math:: U_C \ = \ e^{-i \gamma H_C}

    where :math:`\gamma` is a variational parameter.

    Args:
        hamiltonian (qml.Hamiltonian): The cost Hamiltonian

    Raises:
        ValueError: if the terms of the supplied cost Hamiltonian are not
        exclusively products of diagonal Pauli gates
    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    if not _diagonal_terms(hamiltonian):
        raise ValueError("hamiltonian must be written only in terms of PauliZ and Identity gates")

    return lambda gamma : ApproxTimeEvolution(hamiltonian, gamma, 1)


def mixer_layer(hamiltonian):
    r"""Builds a QAOA mixer layer, for a given mixer Hamiltonian.

        The mixer layer for cost Hamiltonian :math:`H_M` is defined as the following unitary:

        .. math:: U_M \ = \ e^{-i \alpha H_M}

        where :math:`\alpha` is a variational parameter.

        Args:
            hamiltonian (qml.Hamiltonian): The mixer Hamiltonian
        """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    return lambda alpha : ApproxTimeEvolution(hamiltonian, alpha, 1)
