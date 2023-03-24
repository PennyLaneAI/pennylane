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
from dataclasses import dataclass
from typing import Callable, List, Union

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.typing import TensorLike


def a(wire, d=2):
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(0.5j, qml.PauliY(wire))


def ad(wire, d=2):
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(-0.5j, qml.PauliY(wire))


def transmon_interaction(
    connections: list, omega: Union[float, list], g: Union[float, list], delta=None, wires=None, d=2
):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the cQED Hamiltonian of a superconducting transmon system.

    The Hamiltonian is given by

    .. math::

        H = \sum_q \omega_q a^\dagger_q a_q + \sum_{(i, j) \in \mathcal{C}} g_{ij} \left(a^\dagger_i a_j + a_j^\dagger a_i \right)
        + \sum_q \delta_q a^\dagger_q a^\dagger_q a_q a_q

    where :math:`[a^\dagger_p, a_q] = i \delta_{pq}` are bosonic creation and annihilation operators.
    The first term describes the dressed qubit frequencies :math:`\omega_q`, the second term their coupling :math:`g_{ij}` and the last the anharmonicity :math:`\delta_q`,
    which all can vary for different qubits. In practice, the bosonic operators are restricted to a finite dimension of the local Hilbert space (default ``d=2`` corresponds to qubits).
    In that case, the anharmonicity is set to :math:`delta=0` and ignored.

    .. note:: Currently only supporting ``d=2`` with qudit support planned in the future.

    TODO: resource / source for good values.

    .. seealso::

        :func:`~.transmon_drive`

    Args:
        connections (list[tuple(int)]): List of connections ``(i, j)`` between qubits i and j.
        omega (Union[float, list[float]]): List of dressed qubit frequencies in GHz.
            When passing a single float all qubits are assumed to have that same frequency.
        g (Union[float, list[float]]): List of coupling strengths in GHz. Needs to match the length of ``connections``.
            When passing a single float all connections are assumed to have the same strength.
        delta (Union[float, list[float]]): List of anharmonicities in GHz. Ignored when ``d=2``.
            When passing a single float all qubits are assumed to have that same anharmonicity.
        wires (list): Optional, defaults to the unique wires in ``connections`` when set to ``None``. When passing explicit ``wires``,
            needs to at least contain all unique wires in ``connections`` and can be used to initiate additional, unconnected qubits.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        HardwareHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can set up the transmon interaction Hamiltonian with uniform coefficients by passing ``float`` values.

    .. code-block::

        connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
        H = qml.pulse.transmon_interaction(connections, omega=0.5, g=1.)

    The resulting :class:`~.ParametrizedHamiltonian` consists of ``4`` coupling terms and ``6`` qubit terms.

    >>> print(H)
    ParametrizedHamiltonian: terms=10

    """
    if d != 2:
        raise NotImplementedError(
            "Currently only supporting qubits. Qutrits and qudits are planned in the future."
        )

    if wires is not None and not all(i in wires for i in qml.math.unique(connections)):
        raise ValueError(
            f"There are wires in connections {connections} that are not in the provided wires {wires}"
        )

    wires = wires or qml.math.unique(connections)
    n_wires = len(wires)

    # Prepare coefficients
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
    #     if delta is None:
    #         delta = [0.] * n_wires
    #     if qml.math.ndim(delta)==0:
    #         delta = [delta] * n_wires
    #     if len(delta) != n_wires:
    #         raise ValueError(f"Number of qubit anharmonicities delta = {delta} does not match the provided wires = {wires}")
    #     # anharmonicity term
    #     coeffs += list(delta)
    #     observables += [ad(i, d) @ ad(i, d) @ a(i, d) @ a(i, d) for i in wires]

    settings = TransmonSettings(connections, omega, delta, g)

    return HardwareHamiltonian(coeffs, observables, settings=settings)

@dataclass
class TransmonSettings:
    """Dataclass that contains the information of a Rydberg setup.

    Args:
            connections (List): List `[[idx_q0, idx_q1], ..]` of connected qubits (wires)
            omega (Union[float, Callable]):
            delta (Union[float, Callable]):
            g (Union[list, TensorLike, Callable]):
    
    """

    connections: List
    omega: Union[float, Callable]
    g: Union[list, TensorLike, Callable]
    delta: Union[float, Callable] = None

    def __eq__(self, other):
        return (
            self.connections == other.connections
            and self.omega == other.omega
            and self.g == other.g
            and self.delta == other.delta
        )
    
    def __add__(self, other):
        if other is not None:

            new_connections = list(self.connections) + list(other.connections)
            new_omega = list(self.omega) + list(other.omega)
            new_g = list(self.g) + list(other.g)
            if self.delta is None and other.delta is None:
                new_delta = None
            elif self.delta is None and not other.delta is None:
                new_delta = other.delta
            elif self.delta is not None and other.delta is None:
                new_delta = self.delta
            elif self.delta is not None and other.delta is not None:
                new_delta = list(self.delta) + list(other.delta)

            return TransmonSettings(new_connections, new_omega, new_g, delta=new_delta)

        return self
