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
    is the Rydberg interaction constant, which defaults to :math:`862690 \times 2 \pi \text{MHz } \mu \text{m}^6`.
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
        interaction_coeff (float): Defaults to :math:`862690 \times 2 \pi \text{MHz } \mu\text{m}^6`.
            The value will be multiplied by :math:`2 \pi` internally to convert to angular frequency,
            such that only the value in standard frequency (i.e., 862690 in the default example) should be
            passed.
        max_distance (float): Threshold for distance in :math:`\mu\text{m}` between two Rydberg atoms beyond which their
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
    HardwareHamiltonian: terms=36

    As expected, we have :math:`\frac{N(N-1)}{2} = 36` terms for N=9 atoms.

    The interaction term is dependent only on the number and positions of the Rydberg atoms. We can execute this
    pulse program, which corresponds to all driving laser fields being turned off and therefore has no trainable
    parameters. To add a driving laser field, see :func:`~.rydberg_drive`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=9)

        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.evolve(H_i)([], t=[0, 10])
            return qml.expval(qml.Z(0))

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
            # factor 2pi converts interaction coefficient from standard to angular frequency
            Vij = (
                2 * np.pi * interaction_coeff / (abs(atom_distance) ** 6)
            )  # van der Waals potential
            coeffs.append(Vij)
            with qml.QueuingManager.stop_recording():
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
        \delta(t) \sum_{q \in \text{wires}} n_q

    where :math:`\Omega/(2\pi)`, :math:`\phi` and :math:`\delta/(2\pi)` correspond to the amplitude, phase,
    and detuning of the laser, :math:`q` corresponds to the wire index, and
    :math:`\sigma_q^\alpha` for :math:`\alpha = x,y` are the Pauli matrices on the corresponding
    qubit. Finally, :math:`n_q=\frac{1}{2}(\mathbb{I}_q-\sigma_q^z)` is the number operator on qubit :math:`q`.

    .. note::
        For hardware execution, input time is expected to be in units of :math:`\mu\text{s}`, and the frequency
        in units of MHz. It is recommended to also follow this convention for simulation,
        as it avoids numerical problems due to using very large and very small numbers. Frequency inputs will be
        converted internally to angular frequency, such that ``amplitude`` :math:`= \Omega(t)/ (2 \pi)` and
        ``detuning`` :math:`= \delta(t) / (2 \pi)`.

    This driving term can be combined with an interaction term to create a Hamiltonian describing a
    driven Rydberg atom system. Multiple driving terms can be combined by summing them (see example).

    Args:
        amplitude (Union[float, Callable]): Float or callable representing the amplitude of a laser field.
            This should be in units of frequency (MHz), and will be converted to amplitude in angular frequency,
            :math:`\Omega(t)`, internally where needed, i.e. multiplied by :math:`2 \pi`.
        phase (Union[float, Callable]): float or callable representing the phase (in radians) of the laser field
        detuning (Union[float, Callable]): Float or callable representing the detuning of a laser field.
            This should be in units of frequency (MHz), and will be converted to detuning in angular frequency,
            :math:`\delta(t)`, internally where needed, i.e. multiplied by :math:`2 \pi`.
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
    the atoms.

    We provide all frequencies in the driving term in MHz (conversion to angular frequency, i.e. multiplication
    by :math:`2 \pi`, is taken care of internally where needed). Phase (in radians) will not undergo unit conversion.

    For the driving field, we specify a detuning of
    :math:`\delta = 1 \times 2 \pi \text{MHz}`, and an
    amplitude :math:`\Omega(t)` defined by a sinusoidal oscillation, squared to ensure a positve amplitude
    (a requirement for some hardware implementations). The maximum amplitude will dependent on the parameter ``p``
    passed to the amplitude function later, and should also be passed in units of MHz. We introduce a small phase
    shift as well, on the order of 1 rad.

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
        wires = [0, 1, 2, 3]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t) ** 2
        phase = 0.25
        detuning = 1.
        H_d = qml.pulse.rydberg_drive(amplitude, phase, detuning, wires)

    >>> H_i
    HardwareHamiltonian: terms=6
    >>> H_d
    HardwareHamiltonian: terms=3

    The first two terms of the drive Hamiltonian ``H_d`` correspond to the first sum (the sine and cosine terms),
    describing drive between the ground and excited states. The third term corresponds to the shift term
    due to detuning from resonance. This drive term corresponds to a global drive that acts on all 4 wires of
    the device.

    The full Hamiltonian evolution and expectation value measurement can be executed in a ``QNode``:

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=wires)
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_i + H_d)(params, t=[0, 0.5])
            return qml.expval(qml.Z(0))

    Here we set a maximum amplitude of :math:`2.4 \times 2 \pi \text{MHz}`, and calculate the result of running the pulse program:

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.78301974, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(-0.6250622, dtype=float64)]

    We can also create a Hamiltonian with local drives. The following circuit corresponds to the
    evolution where additional local drives acting on wires ``0`` and ``1`` respectively are added to the
    Hamiltonian:

    .. code-block:: python

        amplitude_local_0 = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) ** 2 + p[1]
        phase_local_0 = jnp.pi / 4
        detuning_local_0 = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local_0 = qml.pulse.rydberg_drive(amplitude_local_0, phase_local_0, detuning_local_0, [0])

        amplitude_local_1 = lambda p, t: jnp.cos(jnp.pi * t) ** 2 + p
        phase_local_1 = jnp.pi
        detuning_local_1 = lambda p, t: jnp.sin(jnp.pi * t) + p
        H_local_1 = qml.pulse.rydberg_drive(amplitude_local_1, phase_local_1, detuning_local_1, [1])

        H = H_i + H_d + H_local_0 + H_local_1

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 0.5])
            return qml.expval(qml.Z(0))

        p_global = 2.4
        p_local_amp_0 = [1.3, -2.0]
        p_local_det_0 = -1.5
        p_local_amp_1 = -0.9
        p_local_det_1 = 3.1
        params = [p_global, p_local_amp_0, p_local_det_0, p_local_amp_1, p_local_det_1]


    >>> circuit_local(params)
    Array(0.62640288, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    [Array(1.07614151, dtype=float64),
     [Array(0.36370049, dtype=float64, weak_type=True),
      Array(0.91057498, dtype=float64, weak_type=True)],
     Array(1.3166343, dtype=float64),
     Array(-0.11102892, dtype=float64),
     Array(-0.02205843, dtype=float64)]
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
        # 2pi factors are to convert detuning frequency to angular frequency
        detuning_obs.append(
            # Global phase from the number operator
            -0.5 * sum(qml.Identity(wire) for wire in wires) * np.pi * 2
            # Equivalent of the number operator up to the global phase above
            + 0.5 * sum(qml.Z(wire) for wire in wires) * np.pi * 2
        )
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
