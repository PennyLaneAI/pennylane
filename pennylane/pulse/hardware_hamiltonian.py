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
"""This module contains the classes/functions needed to simulate and execute the evolution of real
Hardware Hamiltonians"""
import warnings
from dataclasses import dataclass
from typing import Callable, List, Union

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian


from .parametrized_hamiltonian import ParametrizedHamiltonian


def drive(amplitude, phase, detuning, wires):
    r"""Constructs a :class:`ParametrizedHamiltonian` representing the action of a driving electromagnetic
    field with a qubit.

    .. math::
        \frac{1}{2} \sum_{j \in \text{wires}} \Omega(t) \left(e^{i \phi(t)} \sigma^+_j + e^{-i \phi(t)} \sigma^-_j \right) -
        \Delta(t) \sigma^z_j

    where :math:`\Omega`, :math:`\phi` and :math:`\Delta` correspond to the amplitude, phase and detuning of the electromagnetic
    driving field and :math:`j` corresponds to the wire index. We are describing the Hamiltonian in terms of ladder operators
    :math:`\sigma^\pm = \frac{1}{2}(\sigma_x \pm i \sigma_y)`.
    Note that the detuning :math:`\Delta := \omega_q - \nu` is defined as the difference between the qubit frequency :math:`\omega_q`
    and the electromagntic field driving frequency :math:`\nu`. For more details, see the theoretical background section below.

    Common hardware systems are superconducting qubits and neutral atoms. The electromagnetic field of the drive is realized by microwave
    and laser fields, respectively, operating at very differnt wavelengths.

    Note that to avoid nummerical problems due to using both very large and very small numbers, it is advisable to match
    the order of magnitudes of frequency and time arguments. For example, when frequencies are of order MHz (microwave pulses for superconducting systems),
    then one can ommit the explicit factor :math:`10^6` by treating the times passed to the constructed :class:`ParametrizedHamiltonian` in :math:`\mu s = 10^{-6}s`
    to be able to use numerical units of order :math:`\mathcal{O}(1)`. We further elaborate on that in the examples below.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude of an
            electromagnetic field
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the electromagnetic field
        detuning (Union[float, Callable]): float or callable returning the detuning of a
            electromagnetic field
        wires (Union[int, List[int]]): integer or list containing wire values for the qubits that
            the electromagnetic field acts on

    Returns:
        HardwareHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the electromagnetic field
        on the qubits.

    .. seealso::

        :func:`~.rydberg_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing a electromagnetic field acting on 4 qubits with a fixed detuning and
    phase, as well as a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-qubit interactions.

    .. code-block:: python

        wires = [0, 1, 2, 3]
        H_int = sum([qml.PauliX(i) @ qml.PauliX((i+1)%len(wires)) for i in wires])

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        detuning = 3 * jnp.pi / 4
        H_d = qml.pulse.drive(amplitude, phase, detuning, wires)

    >>> H_int
    (1) [X0 X1]
    + (1) [X1 X2]
    + (1) [X2 X3]
    + (1) [X3 X0]
    >>> H_d
    ParametrizedHamiltonian: terms=3

    The first two terms of the drive Hamiltonian ``H_d`` correspond to the two terms :math:`\Omega e^{i \phi(t)} \sigma^+_j + \Omega e^{-i \phi(t)} \sigma^-_j`,
    describing a drive between the ground and excited states. The third term corresponding to the shift term
    due to detuning from resonance. In this case, the drive term corresponds to a global drive, as it acts on all 4 wires of
    the device.

    The full Hamiltonian can be evaluated:

    .. code-block:: python3

        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_int + H_d)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.77627534, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(-0.0159532, dtype=float64)]

    We can also create a Hamiltonian with multiple local drives. The following circuit corresponds to the
    evolution where an additional local drive that changes in time is acting on wires ``[0, 1]`` is added to the Hamiltonian:

    .. code-block:: python3

        amplitude_local = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        phase_local = lambda p, t: p * jnp.exp(-0.25 * t)
        detuning_local = jnp.pi / 4
        H_local = qml.pulse.drive(amplitude_local, phase_local, detuning_local, [0, 1])

        H = H_int + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

        p_global = 2.4
        p_amp = [1.3, -2.0]
        p_phase = 0.5
        params = (p_global, p_amp, p_phase)

    >>> circuit_local(params)
    Array(0.4494223, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    (Array(0.17258209, dtype=float64),
     [Array(-0.39050511, dtype=float64, weak_type=True),
      Array(-0.15865324, dtype=float64, weak_type=True)],
     Array(-0.16458317, dtype=float64))

    .. details::
        :title: Theoretical background
        :href: theory

        Depending on the community and field it is often common to write the driving field Hamiltonian as

        .. math::
            H = \frac{1}{2} \Omega(t) \sum_{j \in \text{wires}} \left(e^{i (\phi(t) + \nu t)} \sigma^+_j + e^{-i (\phi(t) + \nu t)} \sigma^-_j \right)
            + \frac{1}{2} \omega_q \sum_{j \in \text{wires}} \sigma^z_j,

        with amplitude :math:`\Omega`, phase :math:`\phi` and drive frequency :math:`\nu` of the electromagnetic field, as well as the qubit frequency :math:`\omega_q`.
        We can move to the rotating frame of the driving field by applying :math:`U = e^{-i\nu t \sigma^z / 2}` which yields the new Hamiltonian

        .. math::
            H = \frac{1}{2} \Omega(t) \sum_{j \in \text{wires}} \left(e^{i \phi(t)} \sigma^+_j + e^{-i \phi(t)} \sigma^-_j \right)
            - \frac{1}{2} (\nu - \omega_q) \sum_{j \in \text{wires}} \sigma^z_j

        We can define :math:`\Delta = \nu - \omega_q` to arrive at the definition above. Note that a potential anharmonicity term,
        as is common for transmon systems when taking into account higher energy levels,
        is unaffected by this transformation.

    .. details::
        **Neutral Atom Rydberg systems**

        In neutral atom systems for quantum computation and quantum simulation, a Rydberg transition is driven by an optical laser that is close to the transition's resonant frequency (with a potential detuning with regards to the resonant frequency on the order of MHz).
        The interaction between different atoms is given by the :func:`rydberg_interaction`, for which we pass the atomic coordinates (in µm),
        here arranged in a square of length :math:`4 \mu m`.

        .. code-block:: python3

            atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
            wires = [1, 2, 3, 4]
            assert len(wires) == len(atom_coordinates)
            H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        We can now simulate driving those atoms with an oscillating amplitude :math:`\Omega` that is trainable, for a duration of :math:`10 \mu s`.
        The total Hamiltonian of that evolution is given by

        .. math::
            \frac{1}{2} p \sin(\pi t) \sum_{j \in \text{wires}} \left(e^{i \pi/2} \sigma^+_j + e^{-i \pi/2} \sigma^-_j \right) -
            \frac{1}{2} \frac{3 \pi}{4} \sum_{j \in \text{wires}} \sigma^z_j + \sum_{k<\ell} V_{k \ell} n_k n_\ell

        and can be executed and differentiated via the following code.

        .. code-block:: python3

            amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
            phase = jnp.pi / 2
            detuning = 3 * jnp.pi / 4
            H_d = qml.pulse.drive(amplitude, phase, detuning, wires)

            dev = qml.device("default.qubit.jax", wires=wires)

            @qml.qnode(dev, interface="jax")
            def circuit(params):
                qml.evolve(H_i + H_d)(params, t=[0, 10])
                return qml.expval(qml.PauliZ(1))

        >>> params = [2.4]
        >>> circuit(params)
        Array(0.94301294, dtype=float64)
        >>> jax.grad(circuit)(params)
        [Array(0.59484969, dtype=float64)]

    """
    if isinstance(wires, int):
        wires = [wires]
    trivial_detuning = not callable(detuning) and qml.math.isclose(detuning, 0.0)

    if not callable(amplitude) and qml.math.isclose(amplitude, 0.0):
        if trivial_detuning:
            raise ValueError(
                f"Expected non-zero value for at least one of either amplitude or detuning, but received amplitude={amplitude} and detuning={detuning}. All terms are zero."
            )

        coeffs = []
        observables = []

    # TODO: use sigma+ and sigma- (not necessary as terms are the same, but for consistency)
    # We compute the `coeffs` and `observables` of the electromagnetic field
    else:
        # We compute the `coeffs` and `observables` of the EM field
        coeffs = [
            amplitude_and_phase(qml.math.cos, amplitude, phase),
            amplitude_and_phase(qml.math.sin, amplitude, phase),
        ]

        drive_x_term = 0.5 * sum(qml.PauliX(wire) for wire in wires)
        drive_y_term = -0.5 * sum(qml.PauliY(wire) for wire in wires)

        observables = [drive_x_term, drive_y_term]

    if not trivial_detuning:
        detuning_term = -1.0 * sum(qml.PauliZ(wire) for wire in wires)
        coeffs.append(detuning)
        observables.append(detuning_term)

    # We convert the pulse data into a list of ``HardwarePulse`` objects
    pulses = [HardwarePulse(amplitude, phase, detuning, wires)]

    return HardwareHamiltonian(coeffs, observables, pulses=pulses)


class HardwareHamiltonian(ParametrizedHamiltonian):
    r"""Internal class used to keep track of the required information to translate a ``ParametrizedHamiltonian``
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` to construct the :class:`ParametrizedHamiltonian`,
    but on top of that also contains attributes that store parameteres relevant for real hardware execution.

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
        interaction_coeff (float): Rydberg interaction constant in units: :math:`\text{MHz} \times \mu m^6`.
            Defaults to :math:`862690 \text{MHz} \times \mu m^6`.

    Returns:
        HardwareHamiltonian: class representing the Hamiltonian of Rydberg or Transmon device.
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
                    warnings.warn("The wires of the drive fields are not present in the ensemble.")
            elif other.register is not None and not other.wires.contains_wires(self.wires):
                warnings.warn("The wires of the drive fields are not present in the ensemble.")

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
            new_coeffs = coeffs + list(other.coeffs.copy())
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
        ParametrizedHamiltonian. Ensures that this returns a HardwareHamiltonian where
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
    """Dataclass that contains the information of a single drive pulse. This class is used
    internally in PL to group into a single object all the data related to a single EM field.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude of an EM
            field
        phase (Union[float, Callable]): float containing the phase (in radians) of the EM field
        detuning (Union[float, Callable]): float or callable returning the detuning of a
            EM field
        wires (Union[int, List[int]]): integer or list containing wire values that the EM field
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
