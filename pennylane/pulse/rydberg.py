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
from dataclasses import dataclass
import numpy as np
import pennylane as qml

from pennylane.pulse import HardwareHamiltonian, HardwarePulse, drive
from pennylane.wires import Wires
from pennylane.pulse.hardware_hamiltonian import _reorder_parameters


def rydberg_interaction(
    register: list, wires=None, interaction_coeff: float = 862690, max_distance: float = np.inf
):
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
        max_distance (float): Threshold for distance in :math:`\mu \text{m}` between two Rydberg atoms beyond which their
            contribution to the interaction term is removed from the Hamiltonian.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the atom interaction

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
            if atom_distance > max_distance:
                continue
            Vij = interaction_coeff / (abs(atom_distance) ** 6)  # van der Waals potential
            coeffs.append(Vij)
            observables.append(qml.prod(qml.Projector([1], wire1), qml.Projector([1], wire2)))
            # Rydberg projectors

    settings = RydbergSettings(register, interaction_coeff)

    return HardwareHamiltonian(
        coeffs, observables, settings=settings, reorder_fn=_reorder_parameters
    )


def rydberg_drive(amplitude, phase, detuning, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser field

    .. math::

        \frac{1}{2} \Omega(t) \sum_{q \in \text{wires}} (\cos(\phi(t))\sigma_q^x - \sin(\phi(t))\sigma_q^y) -
        \delta(t) \sum_{q \in \text{wires}} \sigma_q^z

    where :math:`\Omega`, :math:`\phi` and :math:`\delta` correspond to the amplitude, phase,
    and detuning of the laser, :math:`i` correspond to the wire index, and :math:`\sigma^\alpha` for
    :math:`\alpha = x,y,z` are the Pauli matrices. For hardware execution, time is expected to be in units
    of :math:`\text{µs}`, and the frequency in units of :math:`\text{MHz}`. It is recommended to also follow
    this convention for simulation, as it avoids numerical problems due to using very large and very small
    numbers. This driving term can be combined with an interaction term to create a Hamiltonian describing a
    driven Rydberg atom system. Multiple driving terms can be combined by summing them (see example).

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a
            laser field
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values for the Rydberg atoms that
            the laser field acts on

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the action of the laser field on the Rydberg atoms.

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
        wires = [0, 1, 2, 3]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        detuning = 3 * jnp.pi / 4
        H_d = qml.pulse.rydberg_drive(amplitude, phase, detuning, wires)

    >>> H_i
    ParametrizedHamiltonian: terms=6
    >>> H_d
    ParametrizedHamiltonian: terms=3

    The first two terms of the drive Hamiltonian ``H_d`` correspond to the first sum (the sine and cosine terms),
    describing drive between the ground and excited states. The third term corresponds to the shift term
    due to detuning from resonance. This drive term corresponds to a global drive that acts on all 4 wires of
    the device.

    The full Hamiltonian evolution and expectation value measurement can be executed in a ``QNode``:

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=wires)
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_i + H_d)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.94301294, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(0.59484969, dtype=float64)]

    We can also create a Hamiltonian with local drives. The following circuit corresponds to the
    evolution where additional local drives acting on wires ``0`` and ``1`` respectively are added to the
    Hamiltonian:

    .. code-block:: python

        amplitude_local_0 = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        phase_local_0 = jnp.pi / 4
        detuning_local_0 = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local_0 = qml.pulse.rydberg_drive(amplitude_local_0, phase_local_0, detuning_local_0, [0])

        amplitude_local_1 = lambda p, t: jnp.cos(jnp.pi * t) + p
        phase_local_1 = jnp.pi
        detuning_local_1 = lambda p, t: jnp.sin(jnp.pi * t) + p
        H_local_1 = qml.pulse.rydberg_drive(amplitude_local_1, phase_local_1, detuning_local_1, [1])

        H = H_i + H_d + H_local_0 + H_local_1

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> p_global = 2.4
    >>> p_local_amp_0 = [1.3, -2.0]
    >>> p_local_det_0 = -1.5
    >>> p_local_amp_1 = -0.9
    >>> p_local_det_1 = 3.1
    >>> params = [p_global, p_local_amp_0, p_local_det_0, p_local_amp_1, p_local_det_1]
    >>> circuit_local(params)
    Array(0.6464017, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    [Array(-1.47216703, dtype=float64),
     [Array(0.25855072, dtype=float64, weak_type=True),
      Array(0.1453378, dtype=float64, weak_type=True)],
     Array(0.18329746, dtype=float64),
     Array(0.05901987, dtype=float64),
     Array(-0.23426886, dtype=float64)]
    """
    wires = Wires(wires)
    trivial_detuning = not callable(detuning) and qml.math.isclose(detuning, 0.0)

    if not callable(amplitude) and qml.math.isclose(amplitude, 0.0):
        if trivial_detuning:
            raise ValueError(
                "Expected non-zero value for at least one of either amplitude or detuning, but "
                f"received amplitude={amplitude} and detuning={detuning}. All terms are zero."
            )

        amplitude_term = HardwareHamiltonian([], [])

    else:
        amplitude_term = drive(amplitude, phase, wires)

    detuning_obs, detuning_coeffs = [], []
    if not trivial_detuning:
        detuning_obs.append(-1.0 * sum(qml.PauliZ(wire) for wire in wires))
        detuning_coeffs.append(detuning)

    detuning_term = HardwareHamiltonian(detuning_coeffs, detuning_obs)
    pulses = [HardwarePulse(amplitude, phase, detuning, wires)]

    drive_term = amplitude_term + detuning_term
    drive_term.pulses = pulses

    return drive_term


@dataclass
class RydbergSettings:
    """Dataclass that contains the information of a Rydberg setup.

    Args:
        register (list): coordinates of atoms
        interaction_coeff (float): interaction coefficient C6 from C6/(Rij)**6 term in :func:`rydberg_interaction`
    """

    register: list
    interaction_coeff: float = 0.0

    def __eq__(self, other):
        return (
            qml.math.all(self.register == other.register)
            and self.interaction_coeff == other.interaction_coeff
        )

    def __add__(self, other):
        if other is not None:
            raise ValueError(
                "Cannot add two `HardwareHamiltonian` instances with an interaction term. "
                f"Obtained two instances with settings {self} and {other}. "
                "You most likely tried to add two terms generated by `qml.pulse.rydberg_interaction`"
            )

        return self

    __radd__ = __add__
