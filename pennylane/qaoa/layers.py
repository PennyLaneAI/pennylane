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
        ValueError: if the terms of the supplied cost Hamiltonian are not exclusively products of diagonal Pauli gates

    .. UsageDetails::

        To define a cost layer, one must define a cost Hamiltonian
        and pass it into ``cost_layer``:

        .. code-block:: python

            from pennylane import qaoa
            import pennylane as qml

            cost_h = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)])
            cost_layer = qaoa.cost_layer(cost_h)

        We can then use the cost layer within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(gamma):

                for i in range(2):
                    qml.Hadamard(wires=i)

                cost_layer(gamma)

                return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> circuit(0.5)

        .. code-block:: none

             0: ──H──RZ(-1.0)──╭RZ(-1.0)──┤ ⟨Z⟩
             1: ──H────────────╰RZ(-1.0)──┤ ⟨Z⟩

    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    if not _diagonal_terms(hamiltonian):
        raise ValueError("hamiltonian must be written only in terms of PauliZ and Identity gates")

    return lambda gamma: qml.templates.ApproxTimeEvolution(hamiltonian, gamma, 1)


def mixer_layer(hamiltonian):
    r"""Builds a QAOA mixer layer, for a given mixer Hamiltonian.

    The mixer layer for cost Hamiltonian :math:`H_M` is defined as the following unitary:

    .. math:: U_M \ = \ e^{-i \alpha H_M}

    where :math:`\alpha` is a variational parameter.

    Args:
        hamiltonian (qml.Hamiltonian): The mixer Hamiltonian

    .. UsageDetails::

        To define a mixer layer, one must define a mixer Hamiltonian
        and pass it into ``mixer_layer``:

        .. code-block:: python

            from pennylane import qaoa
            import pennylane as qml

            mixer_h = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1)])
            mixer_layer = qaoa.mixer_layer(mixer_h)

        We can then use the cost layer within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(alpha):

                for i in range(2):
                    qml.Hadamard(wires=i)

                mixer_layer(alpha)

                return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> circuit(0.5)

        .. code-block:: none

             0: ──H──RZ(-1.0)──H──H──╭RZ(-1.0)──H──┤ ⟨Z⟩
             1: ──H──────────────────╰RZ(-1.0)──H──┤ ⟨Z⟩

    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    return lambda alpha: qml.templates.ApproxTimeEvolution(hamiltonian, alpha, 1)
