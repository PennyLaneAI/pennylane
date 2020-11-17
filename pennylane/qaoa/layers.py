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
    r"""Checks if all terms in a Hamiltonian are products of diagonal Pauli gates
    (:class:`~.PauliZ` and :class:`~.Identity`).

    Args:
        hamiltonian (.Hamiltonian): The Hamiltonian being checked

    Returns:
        bool: ``True`` if all terms are products of diagonal Pauli gates, ``False`` otherwise
    """
    val = True

    for i in hamiltonian.ops:
        obs = i.obs if isinstance(i, Tensor) else [i]
        for j in obs:
            if j.name not in ("PauliZ", "Identity"):
                val = False
                break

    return val


def cost_layer(gamma, hamiltonian):
    r"""Applies the QAOA cost layer corresponding to a cost Hamiltonian.

    For the cost Hamiltonian :math:`H_C`, this is defined as the following unitary:

    .. math:: U_C \ = \ e^{-i \gamma H_C}

    where :math:`\gamma` is a variational parameter.

    Args:
        gamma (int or float): The variational parameter passed into the cost layer
        hamiltonian (.Hamiltonian): The cost Hamiltonian

    Raises:
        ValueError: if the terms of the supplied cost Hamiltonian are not exclusively products of diagonal Pauli gates

    .. UsageDetails::

        We first define a cost Hamiltonian:

        .. code-block:: python3

            from pennylane import qaoa
            import pennylane as qml

            cost_h = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)])

        We can then pass it into ``qaoa.cost_layer``, within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(gamma):

                for i in range(2):
                    qml.Hadamard(wires=i)

                cost_layer(gamma, cost_h)

                return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> circuit(0.5)
        >>> print(circuit.draw())
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

    qml.templates.ApproxTimeEvolution(hamiltonian, gamma, 1)


def mixer_layer(alpha, hamiltonian):
    r"""Applies the QAOA mixer layer corresponding to a mixer Hamiltonian.

    For a mixer Hamiltonian :math:`H_M`, this is defined as the following unitary:

    .. math:: U_M \ = \ e^{-i \alpha H_M}

    where :math:`\alpha` is a variational parameter.

    Args:
        alpha (int or float): The variational parameter passed into the mixer layer
        hamiltonian (.Hamiltonian): The mixer Hamiltonian

    .. UsageDetails::

        We first define a mixer Hamiltonian:

        .. code-block:: python3

            from pennylane import qaoa
            import pennylane as qml

            mixer_h = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1)])

        We can then pass it into ``qaoa.mixer_layer``, within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(alpha):

                for i in range(2):
                    qml.Hadamard(wires=i)

                qaoa.mixer_layer(alpha, mixer_h)

                return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> circuit(0.5)
        >>> print(circuit.draw())
        0: ──H──RZ(-1.0)──H──H──╭RZ(-1.0)──H──┤ ⟨Z⟩
        1: ──H──────────────────╰RZ(-1.0)──H──┤ ⟨Z⟩

    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    qml.templates.ApproxTimeEvolution(hamiltonian, alpha, 1)
