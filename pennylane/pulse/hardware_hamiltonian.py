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
"""This module contains the classes/functions needed to simulate and execute pulse programs for real hardware"""
import warnings
from dataclasses import dataclass
from typing import Callable, List, Union

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian


from .parametrized_hamiltonian import ParametrizedHamiltonian


def drive(amplitude, phase, detuning, wires):
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
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values for the Rydberg atoms that
            the laser field acts on

    Returns:
        HardwareHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the laser field
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
        wires = [1, 2, 3, 4]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        detuning = 3 * jnp.pi / 4
        H_d = qml.pulse.drive(amplitude, phase, detuning, wires)

    >>> H_i
    ParametrizedHamiltonian: terms=6
    >>> H_d
    ParametrizedHamiltonian: terms=3

    The first two terms of the drive Hamiltonian ``H_d`` correspond to the first sum (the sine and cosine terms),
    describing drive between the ground and excited states. The third term corresponding to the shift term
    due to detuning from resonance. This drive term corresponds to a global drive that acts on all 4 wires of
    the device.

    The full Hamiltonian can be evaluated:

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
        phase_local = lambda p, t: p * jnp.exp(-0.25 * t)
        detuning_local = jnp.pi / 4
        H_local = qml.pulse.drive(amplitude_local, phase_local, detuning_local, [0, 1])

        H = H_i + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4, [1.3, -2.0]]
    >>> circuit_local(params)
    Array(0.45782223, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    [Array(-0.33522988, dtype=float64),
     [Array(0.40320718, dtype=float64, weak_type=True),
      Array(-0.12003976, dtype=float64, weak_type=True)]]
    """
    if isinstance(wires, int):
        wires = [wires]

    # We compute the `coeffs` and `observables` of the laser field
    coeffs = [
        amplitude_and_phase(qml.math.cos, amplitude, phase),
        amplitude_and_phase(qml.math.sin, amplitude, phase),
        detuning,
    ]

    drive_x_term = sum(qml.PauliX(wire) for wire in wires)
    drive_y_term = sum(-qml.PauliY(wire) for wire in wires)
    detuning_term = sum(qml.PauliZ(wire) for wire in wires)

    observables = [drive_x_term, drive_y_term, detuning_term]

    # We convert the pulse data into a list of ``HardwarePulse`` objects
    pulses = [HardwarePulse(amplitude, phase, detuning, wires)]

    return HardwareHamiltonian(coeffs, observables, pulses=pulses)


class HardwareHamiltonian(ParametrizedHamiltonian):
    r"""Internal class used to keep track of the required information to translate a ``ParametrizedHamiltonian``
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` that represent one or more
    terms of the Hamiltonian of an ensemble of Hardware atoms under the action of local and global
    laser fields:

    .. math::

        H = \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi_i)\sigma_i^x - \sin(\phi_i)\sigma_i^y) -
        \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} V_{ij} n_i n_j

    Additionally, it also contains two more attributes (``register`` and ``pulses``) that contain
    the information that the hardware needs to execute this Hamiltonian.

    .. warning::

        This class should NEVER be initialized directly! Please use the functions
        :func:`rydberg_interaction` and :func:`drive` instead.

    .. seealso:: :func:`rydberg_interaction`, :func:`drive`, :class:`ParametrizedHamiltonian`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        register (list): list of coordinates (in micrometers) of each atom in the ensemble
        pulses (list): list of ``HardwarePulse`` classes containing the information about the
            amplitude, phase, detuning and wires of each pulse
        interaction_coeff (float): Hardware interaction constant in units: :math:`\text{MHz} \times \mu m^6`.
            Defaults to :math:`862690 \text{MHz} \times \mu m^6`.

    Returns:
        HardwareHamiltonian: class representing the Hamiltonian of an ensemble of Rydberg atoms
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeffs,
        observables,
        register: list = None,
        pulses: List["HardwarePulse"] = None,
        interaction_coeff: float = 862690,
    ):
        self.register = register
        self.pulses = [] if pulses is None else pulses
        self.interaction_coeff = interaction_coeff
        super().__init__(coeffs, observables)

    def __call__(self, params, t):
        params = _reorder_parameters(params, self.coeffs_parametrized)
        return super().__call__(params, t)

    def __add__(self, other):
        if isinstance(other, HardwareHamiltonian):
            # Update coeffs, obs and hardware attributes
            if self.register is not None:
                if other.register is not None:
                    raise ValueError("We cannot add two Hamiltonians with an interaction term!")
                if not self.wires.contains_wires(other.wires):
                    warnings.warn(
                        "The wires of the laser fields are not present in the Rydberg ensemble."
                    )
            elif other.register is not None and not other.wires.contains_wires(self.wires):
                warnings.warn(
                    "The wires of the laser fields are not present in the Rydberg ensemble."
                )

            new_register = self.register or other.register
            new_pulses = self.pulses + other.pulses
            new_ops = self.ops + other.ops
            new_coeffs = self.coeffs + other.coeffs
            return HardwareHamiltonian(
                new_coeffs, new_ops, register=new_register, pulses=new_pulses
            )

        ops = self.ops.copy()
        coeffs = self.coeffs.copy()
        register = self.register
        pulses = self.pulses

        if isinstance(other, (Hamiltonian, ParametrizedHamiltonian)):
            new_coeffs = coeffs + other.coeffs.copy()
            new_ops = ops + other.ops.copy()
            return HardwareHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        if isinstance(other, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = coeffs + [other.scalar]
            new_ops = ops + [other.base]
            return HardwareHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        if isinstance(other, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [other]
            return HardwareHamiltonian(new_coeffs, new_ops, register=register, pulses=pulses)

        return NotImplemented

    def __radd__(self, other):
        """Deals with the special case where a HardwareHamiltonian is added to a
        ParametrizedHamiltonian. Ensures that this returs a HardwareHamiltonian where
        the order of the parametrized coefficients and operators matches the order of
        the hamiltonians, i.e. that

        ParametrizedHamiltonian + HardwareHamiltonian

        returns a HardwareHamiltonian where the call expects params = [params_PH] + [params_RH]
        """
        if isinstance(other, ParametrizedHamiltonian):
            ops = self.ops.copy()
            coeffs = self.coeffs.copy()

            new_coeffs = other.coeffs.copy() + coeffs
            new_ops = other.ops.copy() + ops

            return HardwareHamiltonian(
                new_coeffs, new_ops, register=self.register, pulses=self.pulses
            )

        return self.__add__(other)


@dataclass
class HardwarePulse:
    """Dataclass that contains the information of a single Hardware pulse. This class is used
    internally in PL to group into a single object all the data related to a single laser field.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a laser
            field
        phase (Union[float, Callable]): float containing the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on
    """

    amplitude: Union[float, Callable]
    phase: Union[float, Callable]
    detuning: Union[float, Callable]
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


def amplitude_and_phase(trig_fn, amp, phase):
    """Wrapper function for combining amplitude and phase into a single callable
    (or constant if neither amplitude nor phase are callable)."""
    if not callable(amp) and not callable(phase):
        return amp * trig_fn(phase)
    return AmplitudeAndPhase(trig_fn, amp, phase)


# pylint:disable = too-few-public-methods
class AmplitudeAndPhase:
    """Class storing combined amplitude and phase callable if either or both
    of amplitude or phase are callable."""

    def __init__(self, trig_fn, amp, phase):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)

        def callable_amp_and_phase(params, t):
            return amp(params[0], t) * trig_fn(phase(params[1], t))

        def callable_amp(params, t):
            return amp(params, t) * trig_fn(phase)

        def callable_phase(params, t):
            return amp * trig_fn(phase(params, t))

        if self.amp_is_callable and self.phase_is_callable:
            self.func = callable_amp_and_phase

        elif self.amp_is_callable:
            self.func = callable_amp

        elif self.phase_is_callable:
            self.func = callable_phase

    def __call__(self, params, t):
        return self.func(params, t)


def _reorder_parameters(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude.

    Consolidates phase and amplitude parameters in the case that both are callable,
    and duplicates phase and/or amplitude parameters if either are callables, since
    they will be passed to two operators in the Hamiltonian"""

    reordered_params = []

    coeff_idx = 0
    params_idx = 0

    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhase):
                if coeff.phase_is_callable and coeff.amp_is_callable:
                    # add the joined parameters twice, and skip an index
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    coeff_idx += 2
                    params_idx += 2
                elif coeff.phase_is_callable or coeff.amp_is_callable:
                    reordered_params.append(params[params_idx])
                    reordered_params.append(params[params_idx])
                    coeff_idx += 2
                    params_idx += 1
            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1

    return reordered_params
