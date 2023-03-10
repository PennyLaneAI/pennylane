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
"""This module contains the classes/functions needed to simulate the evolution of ensembles of
individual (trapped) rydberg atoms under the excitation of several laser fields."""
import warnings
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

import pennylane as qml
from pennylane.wires import Wires

from .parametrized_hamiltonian import ParametrizedHamiltonian


def rydberg_interaction(register: list, wires=None, interaction_coeff: float = 862690):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the interaction of an ensemble of
    Rydberg atoms due to the Rydberg blockade

    .. math::

        \sum_{i<j} V_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Rydberg state of the atom :math:`i`, and
    :math:`V_{ij}` is the van der Waals potential:

    .. math::

        V_{ij} = \frac{C_6}{R_{ij}^6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
    is the Rydberg interaction constant, which defaults to :math:`862690 \text{MHz} \times \mu \text{m}^6`.
    The unit of time for the evolution of this Rydberg interaction term is in :math:`\mu \text{s}`.
    This interaction term can be combined with laser drive terms (:func:`~.rydberg_drive`) to create
    a Hamiltonian describing a driven Rydberg atom system.

    .. seealso::

        :func:`~.rydberg_drive`

    Args:
        register (list): list of coordinates of the Rydberg atoms (in micrometers)
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``register``. If ``None``, each atom's wire value will
            correspond to its index in the ``register`` list.
        interaction_coeff (float): Rydberg interaction constant in units: :math:`\text{MHz} \times \mu \text{m}^6`.
            Defaults to :math:`862690 \text{ MHz} \times \mu \text{m}^6`. This value is based on an assumption that
            frequencies and energies in the Hamiltonian are provided in units of MHz.

    Returns:
        RydbergHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the atom interaction

    **Example**

    We create a Hamiltonian describing the van der Waals interaction among 9 Rydberg atoms in a square lattice:

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 5], [0, 10], [5, 0], [5, 5], [5, 10], [10, 0], [10, 5], [10, 10]]
        wires = [1, 5, 0, 2, 4, 3, 8, 6, 7]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires=wires)

    >>> H_i
    ParametrizedHamiltonian: terms=36

    As expected, we have :math:`\frac{N(N-1)}{2} = 36` terms for N=6 atoms.

    The interaction term is dependent only on the number and positions of the Rydberg atoms. We can execute this
    pulse program, which corresponds to all driving laser fields being turned off and therefore has no trainable
    parameters. To add a driving laser field, see :func:`~.rydberg_drive`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=9)

        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.evolve(H_i)([], t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    Array(1., dtype=float32)
    """
    if wires is None:
        wires = list(range(len(register)))
    elif len(wires) != len(register):
        raise ValueError("The length of the wires and the register must match.")

    coeffs = []
    observables = []
    for idx, (pos1, wire1) in enumerate(zip(register[:-1], wires[:-1])):
        for pos2, wire2 in zip(register[(idx + 1) :], wires[(idx + 1) :]):
            atom_distance = np.linalg.norm(qml.math.array(pos2) - pos1)
            Vij = interaction_coeff / (abs(atom_distance) ** 6)  # van der Waals potential
            coeffs.append(Vij)
            observables.append(qml.prod(qml.Projector([1], wire1), qml.Projector([1], wire2)))
            # Rydberg projectors

    return RydbergHamiltonian(
        coeffs, observables, register=register, interaction_coeff=interaction_coeff
    )


def rydberg_drive(amplitude, detuning, phase, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser
    field with the given rabi frequency, detuning and phase acting on the given wires

    .. math::

        \frac{1}{2} \Omega(t) \sum_{i \in \text{wires}} (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
        \frac{1}{2} \delta(t) \sum_{i \in \text{wires}} \sigma_i^z

    where :math:`\Omega`, :math:`\delta` and :math:`\phi` correspond to the rabi frequency, detuning
    and phase of the laser, :math:`i` correspond to the wire index, and :math:`\sigma^\alpha` for
    :math:`\alpha = x,y,z` are the Pauli matrices. The unit of time for the  evolution of this Rydberg
    drive term is :math:`\mu \text{s}`. This driving term can be combined with an interaction term to
    create a Hamiltonian describing a driven Rydberg atom system. Multiple driving terms can be combined
    by summing them (see example).

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a
            laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        phase (float): float containing the phase (in radians) of the laser field
        wires (Union[int, List[int]]): integer or list containing wire values for the Rydberg atoms that
            the laser field acts on

    Returns:
        RydbergHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the laser field
        on the Rydberg atoms.

    .. seealso::

        :func:`~.rydberg_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing a laser acting on 4 wires (Rydberg atoms) with a fixed detuning and
    phase, and a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-atom interactions due to van der Waals forces, as well as the driving term for the laser driving
    the atoms:

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        detuning = 3 * jnp.pi / 4
        phase = jnp.pi / 2
        wires = [0, 1, 2, 3]
        H_d = qml.pulse.rydberg_drive(amplitude, detuning, phase, wires)

    >>> H_i
    ParametrizedHamiltonian: terms=6
    >>> H_d
    ParametrizedHamiltonian: terms=2

    The two terms of the drive field correspond to the first and second sum, corresponding to the drive and the shift term.
    This drive term corresponds to a global drive that acts on all wires of the device.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_i + H_d)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.97137696, dtype=float32)
    >>> jax.grad(circuit)(params)
    [Array(0.10493923, dtype=float32)]

    We can also create a Hamiltonian with multiple local drives. The following circuit corresponds to the
    evolution where an additional local drive acting on wires ``[0, 1]`` is added to the Hamiltonian:

    .. code-block:: python

        amplitude_local = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        detuning_local = lambda p, t: p * jnp.exp(-0.25 * t)
        phase_local = jnp.pi / 4
        H_local = qml.pulse.rydberg_drive(amplitude_local, detuning_local, phase_local, [0, 1])

        H = H_i + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4, [1.3, -2.0]]
    >>> circuit_local(params)
    Array(0.45782223, dtype=float64)
    >>> jax.grad(circuit_local_drives)(params)
    [Array(-0.33522988, dtype=float64),
     [Array(0.40320718, dtype=float64, weak_type=True),
      Array(-0.12003976, dtype=float64, weak_type=True)]]
    """
    if isinstance(wires, int):
        wires = [wires]

    # We compute the `coeffs` and `observables` of the laser field
    coeffs = [amplitude, detuning]
    amplitude_observable = sum(
        qml.math.cos(phase) * qml.PauliX(wire) - qml.math.sin(phase) * qml.PauliY(wire)
        for wire in wires
    )
    detuning_observable = sum(qml.PauliZ(wire) for wire in wires)
    observables = [amplitude_observable, detuning_observable]

    # We convert the pulse data into a list of ``RydbergPulse`` objects
    pulses = [RydbergPulse(amplitude, detuning, phase, wires)]
    return RydbergHamiltonian(coeffs, observables, pulses=pulses)


class RydbergHamiltonian(ParametrizedHamiltonian):
    r"""Internal class used to keep track of the required information to translate a ``ParametrizedHamiltonian``
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` that represent one or more
    terms of the Hamiltonian of an ensemble of Rydberg atoms under the action of local and global
    laser fields:

    .. math::

        H = \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi_i)\sigma_i^x - \sin(\phi_i)\sigma_i^y) -
        \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} V_{ij} n_i n_j

    Additionally, it also contains two more attributes (``register`` and ``pulses``) that contain
    the information that the hardware needs to execute this Hamiltonian.

    .. warning::

        This class should NEVER be initialized directly! Please use the functions
        :func:`rydberg_interaction` and :func:`rydberg_drive` instead.

    .. seealso:: :func:`rydberg_interaction`, :func:`rydberg_drive`, :class:`ParametrizedHamiltonian`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        register (list): list of coordinates (in micrometers) of each atom in the ensemble
        pulses (list): list of ``RydbergPulse`` classes containing the information about the
            amplitude, phase, detuning and wires of each pulse
        interaction_coeff (float): Rydberg interaction constant in units: :math:`\text{MHz} \times \mu m^6`.
            Defaults to :math:`862690 \text{MHz} \times \mu m^6`.

    Returns:
        RydbergHamiltonian: class representing the Hamiltonian of an ensemble of Rydberg atoms
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeffs,
        observables,
        register: list = None,
        pulses: List["RydbergPulse"] = None,
        interaction_coeff: float = 862690,
    ):
        self.register = register
        self.pulses = [] if pulses is None else pulses
        self.interaction_coeff = interaction_coeff
        super().__init__(coeffs, observables)

    def __add__(self, other):
        if not isinstance(other, RydbergHamiltonian):
            return super().__add__(other)

        # Update coeffs, obs and hardware attributes
        if self.register is not None:
            if other.register is not None:
                raise ValueError("We cannot add two Hamiltonians with an interaction term!")
            if not self.wires.contains_wires(other.wires):
                warnings.warn(
                    "The wires of the laser fields are not present in the Rydberg ensemble."
                )
        elif other.register is not None and not other.wires.contains_wires(self.wires):
            warnings.warn("The wires of the laser fields are not present in the Rydberg ensemble.")

        new_register = self.register or other.register
        new_pulses = self.pulses + other.pulses
        new_ops = self.ops + other.ops
        new_coeffs = self.coeffs + other.coeffs
        return RydbergHamiltonian(new_coeffs, new_ops, register=new_register, pulses=new_pulses)


@dataclass
class RydbergPulse:
    """Dataclass that contains the information of a single Rydberg pulse. This class is used
    internally in PL to group into a single object all the data related to a single laser field.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a laser
            field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        phase (float): float containing the phase (in radians) of the laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on
    """

    amplitude: Union[float, Callable]
    detuning: Union[float, Callable]
    phase: float
    wires: List[Wires]

    def __post_init__(self):
        self.wires = Wires(self.wires)

    def __eq__(self, other):
        return (
            self.amplitude == other.amplitude
            and self.phase == other.phase
            and self.detuning == other.detuning
            and self.wires == other.wires
        )
