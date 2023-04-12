# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes/functions specific for simulation of superconducting transmon hardware systems"""
import warnings

from dataclasses import dataclass
from typing import Callable, List, Union

import pennylane as qml
import pennylane.numpy as np
from pennylane.pulse import HardwareHamiltonian
from pennylane.typing import TensorLike
from pennylane.wires import Wires


# TODO ladder operators once there is qudit support
# pylint: disable=unused-argument
def a(wire, d=2):
    """creation operator"""
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(0.5j, qml.PauliY(wire))


def ad(wire, d=2):
    """annihilation operator"""
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(-0.5j, qml.PauliY(wire))


# pylint: disable=too-many-arguments
def transmon_interaction(
    omega: Union[float, list],
    connections: list,
    g: Union[float, list],
    wires: list,
    anharmonicity=None,
    d=2,
):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the circuit QED Hamiltonian of a superconducting transmon system.

    The Hamiltonian is given by

    .. math::

        H = \sum_{q\in \text{wires}} \omega_q a^\dagger_q a_q
        + \sum_{(i, j) \in \mathcal{C}} g_{ij} \left(a^\dagger_i a_j + a_j^\dagger a_i \right)
        + \sum_{q\in \text{wires}} \alpha_q a^\dagger_q a^\dagger_q a_q a_q

    where :math:`[a^\dagger_p, a_q] = i \delta_{pq}` are bosonic creation and annihilation operators.
    The first term describes the dressed qubit frequencies :math:`\omega_q`, the second term their
    coupling :math:`g_{ij}` and the last the anharmonicity :math:`\alpha_q`, which all can vary for
    different qubits. In practice, the bosonic operators are restricted to a finite dimension of the
    local Hilbert space (default ``d=2`` corresponds to qubits).
    In that case, the anharmonicity is set to :math:`\alpha=0` and ignored.

    The values of :math:`\omega` and :math:`\alpha` are typically around :math:`5 \times 2\pi \text{GHz}` and :math:`0.3 \times 2\pi \text{GHz}`, respectively.
    It is common for different qubits to be out of tune with different energy gaps. The coupling strength
    :math:`g` typically varies betwewen :math:`[0.001, 0.1] \times 2\pi \text{GHz}`. For some example parameters,
    see e.g. `arXiv:1804.04073 <https://arxiv.org/abs/1804.04073>`_,
    `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`_, or `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`_.

    .. note:: Currently only supporting ``d=2`` with qudit support planned in the future.

    .. seealso::

        :func:`~.drive`

    Args:
        omega (Union[float, list[float]]): List of dressed qubit frequencies in GHz. Needs to match the length of ``wires``.
            When passing a single float all qubits are assumed to have that same frequency.
        connections (list[tuple(int)]): List of connections ``(i, j)`` between qubits i and j.
            When the wires in ``connections`` are not contained in ``wires``, a warning is raised.
        g (Union[float, list[float]]): List of coupling strengths in GHz. Needs to match the length of ``connections``.
            When passing a single float need explicit ``wires``.
        anharmonicity (Union[float, list[float]]): List of anharmonicities in GHz. Ignored when ``d=2``.
            When passing a single float all qubits are assumed to have that same anharmonicity.
        wires (list): Needs to be of the same length as omega. Note that there can be additional
            wires in the resulting operator from the ``connections``, which are treated independently.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        HardwareHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can set up the transmon interaction Hamiltonian with uniform coefficients by passing ``float`` values.

    .. code-block::

        connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
        H = qml.pulse.transmon_interaction(omega=0.5, connections=connections, g=1.)

    The resulting :class:`~.ParametrizedHamiltonian` consists of ``4`` coupling terms and ``6`` qubits
    because there are six different wire indices in ``connections``.

    >>> print(H)
    ParametrizedHamiltonian: terms=10

    We can also provide individual values for each of the qubit energies and connections.

    .. code-block::

        omega = [0.5, 0.4, 0.3, 0.2, 0.1, 0.]
        g = [1., 2., 3., 4.]
        H = qml.pulse.transmon_interaction(omega=omega, connections=connections, g=g)

    """
    if d != 2:
        raise NotImplementedError(
            "Currently only supporting qubits. Qutrits and qudits are planned in the future."
        )

    # if wires is None and qml.math.ndim(omega) == 0:
    #     raise ValueError(
    #         f"Cannot instantiate wires automatically. Either need specific wires or a list of omega."
    #         f"Received wires {wires} and omega of type {type(omega)}"
    #     )

    # wires = wires or list(range(len(omega)))

    n_wires = len(wires)

    if not Wires(wires).contains_wires(Wires(np.unique(connections).tolist())):
        warnings.warn(
            f"Caution, wires and connections do not match. "
            f"I.e., wires in connections {connections} are not contained in the wires {wires}"
        )

    # Prepare coefficients
    if anharmonicity is None:
        anharmonicity = [0.0] * n_wires

    # TODO: make coefficients callable / trainable. Currently not supported
    if qml.math.ndim(omega) == 0:
        omega = [omega] * n_wires
    if len(omega) != n_wires:
        raise ValueError(
            f"Number of qubit frequencies omega = {omega} does not match the provided wires = {wires}"
        )

    if qml.math.ndim(g) == 0:
        g = [g] * len(connections)
    if len(g) != len(connections):
        raise ValueError(
            f"Number of coupling terms {g} does not match the provided connections = {connections}"
        )

    # qubit term
    coeffs = list(omega)
    observables = [ad(i, d) @ a(i, d) for i in wires]

    # coupling term term
    coeffs += list(g)
    observables += [ad(i, d) @ a(j, d) + ad(j, d) @ a(i, d) for (i, j) in connections]

    # TODO Qudit support. Currently not supported but will be in the future.
    # if d>2:
    #     if anharmonicity is None:
    #         anharmonicity = [0.] * n_wires
    #     if qml.math.ndim(anharmonicity)==0:
    #         anharmonicity = [anharmonicity] * n_wires
    #     if len(anharmonicity) != n_wires:
    #         raise ValueError(f"Number of qubit anharmonicities anharmonicity = {anharmonicity} does not match the provided wires = {wires}")
    #     # anharmonicity term
    #     coeffs += list(anharmonicity)
    #     observables += [ad(i, d) @ ad(i, d) @ a(i, d) @ a(i, d) for i in wires]

    settings = TransmonSettings(connections, omega, g, anharmonicity=anharmonicity)

    return HardwareHamiltonian(coeffs, observables, settings=settings)


@dataclass
class TransmonSettings:
    """Dataclass that contains the information of a Transmon setup.

    .. see-also:: :func:`transmon_interaction`

    Args:
            connections (List): List `[[idx_q0, idx_q1], ..]` of connected qubits (wires)
            omega (List[float, Callable]):
            anharmonicity (List[float, Callable]):
            g (List[list, TensorLike, Callable]):

    """

    connections: List
    omega: Union[float, Callable]
    g: Union[list, TensorLike, Callable]
    anharmonicity: Union[float, Callable]

    def __eq__(self, other):
        return (
            qml.math.all(self.connections == other.connections)
            and qml.math.all(self.omega == other.omega)
            and qml.math.all(self.g == other.g)
            and qml.math.all(self.anharmonicity == other.anharmonicity)
        )

    def __add__(self, other):
        if other is not None:
            new_connections = list(self.connections) + list(other.connections)
            new_omega = list(self.omega) + list(other.omega)
            new_g = list(self.g) + list(other.g)
            new_anh = list(self.anharmonicity) + list(other.anharmonicity)
            return TransmonSettings(new_connections, new_omega, new_g, anharmonicity=new_anh)

        return self
