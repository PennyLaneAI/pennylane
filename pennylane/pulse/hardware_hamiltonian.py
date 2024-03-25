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

from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator


from .parametrized_hamiltonian import ParametrizedHamiltonian


def drive(amplitude, phase, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving electromagnetic
    field with a set of qubits.

    .. math::
        \frac{1}{2} \sum_{j \in \text{wires}} \Omega(t) \left(e^{i \phi(t)} \sigma^+_j + e^{-i \phi(t)} \sigma^-_j \right)

    where :math:`\Omega` and :math:`\phi` correspond to the amplitude and phase of the
    electromagnetic driving field and :math:`j` corresponds to the wire index. We are describing the Hamiltonian
    in terms of ladder operators :math:`\sigma^\pm = \frac{1}{2}(\sigma_x \pm i \sigma_y)`. Note that depending on the
    hardware realization (neutral atoms, superconducting qubits), there are different conventions and notations.
    E.g., for superconducting qubits it is common to describe the exponent of the phase factor as :math:`\exp(i(\phi(t) + \nu t))`, where :math:`\nu` is the
    drive frequency. We describe their relations in the theoretical background section below.

    Common hardware systems are superconducting qubits and neutral atoms. The electromagnetic field of the drive is
    realized by microwave and laser fields, respectively, operating at very different wavelengths.
    To avoid nummerical problems due to using both very large and very small numbers, it is advisable to match
    the order of magnitudes of frequency and time arguments.
    Read the usage details for more information on how to choose :math:`\Omega` and :math:`\phi`.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude of an
            electromagnetic field
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the electromagnetic field
        wires (Union[int, List[int]]): integer or list containing wire values for the qubits that
            the electromagnetic field acts on

    Returns:
        ParametrizedHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the electromagnetic field
        on the qubits.

    .. seealso::

        :func:`~.rydberg_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing an electromagnetic field acting on 4 qubits with a fixed
    phase, as well as a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-qubit interactions.

    .. code-block:: python3

        wires = [0, 1, 2, 3]
        H_int = sum([qml.X(i) @ qml.X((i+1)%len(wires)) for i in wires])

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        H_d = qml.pulse.drive(amplitude, phase, wires)

    >>> H_int
    (1) [X0 X1]
    + (1) [X1 X2]
    + (1) [X2 X3]
    + (1) [X3 X0]
    >>> H_d
    HardwareHamiltonian:: terms=2

    The terms of the drive Hamiltonian ``H_d`` correspond to the two terms
    :math:`\Omega e^{i \phi(t)} \sigma^+_j + \Omega e^{-i \phi(t)} \sigma^-_j`,
    describing a drive between the ground and excited states.
    In this case, the drive term corresponds to a global drive, as it acts on all 4 wires of
    the device.

    The full Hamiltonian can be evaluated:

    .. code-block:: python3

        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_int + H_d)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.32495208, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(1.31956098, dtype=float64)]

    We can also create a Hamiltonian with multiple local drives. The following circuit corresponds to the
    evolution where an additional local drive that changes in time is acting on wires ``[0, 1]`` is added to the Hamiltonian:

    .. code-block:: python3

        amplitude_local = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        phase_local = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local = qml.pulse.drive(amplitude_local, phase_local, [0, 1])

        H = H_int + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

        p_global = 2.4
        p_amp = [1.3, -2.0]
        p_phase = 0.5
        params = (p_global, p_amp, p_phase)

    >>> circuit_local(params)
    Array(-0.5334795, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    (Array(0.01654573, dtype=float64),
     [Array(-0.04422795, dtype=float64, weak_type=True),
      Array(-0.51375441, dtype=float64, weak_type=True)],
     Array(0.21901967, dtype=float64))

    .. details::
        :title: Theoretical background
        :href: theory

        Depending on the community and field it is often common to write the driving field Hamiltonian as

        .. math::
            H = \frac{1}{2} \Omega(t) \sum_{j \in \text{wires}} \left(e^{i (\phi(t) + \nu t)} \sigma^+_j + e^{-i (\phi(t) + \nu t)} \sigma^-_j \right)
            + \omega_q \sum_{j \in \text{wires}} \sigma^z_j,

        with amplitude :math:`\Omega`, phase :math:`\phi` and drive frequency :math:`\nu` of the electromagnetic field, as well as the qubit frequency :math:`\omega_q`.
        We can move to the rotating frame of the driving field by applying :math:`U = e^{-i\nu t \sigma^z}` which yields the new Hamiltonian

        .. math::
            H = \frac{1}{2} \Omega(t) \sum_{j \in \text{wires}} \left(e^{i \phi(t)} \sigma^+_j + e^{-i \phi(t)} \sigma^-_j \right)
            - (\nu - \omega_q) \sum_{j \in \text{wires}} \sigma^z_j

        The latter formulation is more common in neutral atom systems where we define the detuning from the atomic energy gap
        as :math:`\Delta = \nu - \omega_q`. This is because here all atoms have the same energy gap, whereas for superconducting
        qubits that is typically not the case.
        Note that a potential anharmonicity term, as is common for transmon systems when taking into account higher energy
        levels, is unaffected by this transformation.

        Further, note that the factor :math:`\frac{1}{2}` is a matter of convention. We keep it for ``drive()`` as well as :func:`~.rydberg_drive`,
        but ommit it in :func:`~.transmon_drive`, as is common in the respective fields.

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

        .. code-block:: python3

            amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
            phase = jnp.pi / 2

            H_d = qml.pulse.drive(amplitude, phase, wires)

            # detuning term
            H_z = qml.dot([-3*np.pi/4]*len(wires), [qml.Z(i) for i in wires])


        The total Hamiltonian of that evolution is given by

        .. math::
            \frac{1}{2} p \sin(\pi t) \sum_{j \in \text{wires}} \left(e^{i \pi/2} \sigma^+_j + e^{-i \pi/2} \sigma^-_j \right) -
            \frac{3 \pi}{4} \sum_{j \in \text{wires}} \sigma^z_j + \sum_{k<\ell} V_{k \ell} n_k n_\ell

        and can be executed and differentiated via the following code.

        .. code-block:: python3

            dev = qml.device("default.qubit.jax", wires=wires)
            @qml.qnode(dev, interface="jax")
            def circuit(params):
                qml.evolve(H_i + H_z + H_d)(params, t=[0, 10])
                return qml.expval(qml.Z(1))

        >>> params = [2.4]
        >>> circuit(params)
        Array(0.6962041, dtype=float64)
        >>> jax.grad(circuit)(params)
        [Array(1.75825695, dtype=float64)]
    """
    wires = Wires(wires)

    # TODO: use sigma+ and sigma- (not necessary as terms are the same, but for consistency)
    # We compute the `coeffs` and `observables` of the EM field
    coeffs = [
        amplitude_and_phase(qml.math.cos, amplitude, phase),
        amplitude_and_phase(qml.math.sin, amplitude, phase),
    ]

    drive_x_term = qml.Hamiltonian([0.5] * len(wires), [qml.X(wire) for wire in wires])
    drive_y_term = qml.Hamiltonian([-0.5] * len(wires), [qml.Y(wire) for wire in wires])

    observables = [drive_x_term, drive_y_term]

    return HardwareHamiltonian(coeffs, observables, _reorder_parameters)


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
        reorder_fn (callable): function for reordering the parameters before calling.
            This allows automatically copying parameters when they are used for different terms,
            as well as allowing single terms to depend on multiple parameters, as is the case for
            drive Hamiltonians. Note that in order to add two HardwareHamiltonians,
            the reorder_fn needs to be matching.
        settings Union[RydbergSettings, TransmonSettings]: Dataclass containing the hardware specific settings. Default is ``None``.
        pulses (list[HardwarePulse]): list of ``HardwarePulse`` dataclasses containing the information about the
            amplitude, phase, drive frequency and wires of each pulse

    Returns:
        HardwareHamiltonian: class representing the Hamiltonian of Rydberg or Transmon device.

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeffs,
        observables,
        reorder_fn: Callable = _reorder_parameters,
        pulses: List["HardwarePulse"] = None,
        settings: Union["RydbergSettings", "TransmonSettings"] = None,
    ):
        self.settings = settings
        self.pulses = [] if pulses is None else pulses
        self.reorder_fn = reorder_fn
        super().__init__(coeffs, observables)

    def __call__(self, params, t):
        params = self.reorder_fn(params, self.coeffs_parametrized)
        return super().__call__(params, t)

    def __repr__(self):
        return f"HardwareHamiltonian: terms={qml.math.shape(self.coeffs)[0]}"

    def __add__(self, other):  # pylint: disable=too-many-return-statements
        if isinstance(other, HardwareHamiltonian):
            if not self.reorder_fn == other.reorder_fn:
                raise ValueError(
                    "Cannot add two HardwareHamiltonians with different reorder functions. "
                    f"Received reorder_fns {self.reorder_fn} and {other.reorder_fn}. This is "
                    "likely due to an attempt to add hardware compatible Hamiltonians for "
                    "different target systems."
                )
            if self.settings is None and other.settings is None:
                new_settings = None
            else:
                new_settings = self.settings + other.settings

            new_pulses = self.pulses + other.pulses

            new_ops = self.ops + other.ops
            new_coeffs = self.coeffs + other.coeffs
            return HardwareHamiltonian(
                new_coeffs,
                new_ops,
                reorder_fn=self.reorder_fn,
                settings=new_settings,
                pulses=new_pulses,
            )

        ops = self.ops.copy()
        coeffs = self.coeffs.copy()
        settings = self.settings
        pulses = self.pulses

        if isinstance(
            other, (qml.ops.Hamiltonian, qml.ops.LinearCombination, ParametrizedHamiltonian)
        ):
            new_coeffs = coeffs + list(other.coeffs.copy())
            new_ops = ops + other.ops.copy()
            return HardwareHamiltonian(
                new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses
            )

        if isinstance(other, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = coeffs + [other.scalar]
            new_ops = ops + [other.base]
            return HardwareHamiltonian(
                new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses
            )

        if isinstance(other, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [other]
            return HardwareHamiltonian(
                new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses
            )

        if isinstance(other, (int, float)):
            if other in (0, 0.0):
                return HardwareHamiltonian(
                    coeffs, ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses
                )
            new_coeffs = coeffs + [other]
            with qml.queuing.QueuingManager.stop_recording():
                new_ops = ops + [qml.Identity(self.wires[0])]

            return HardwareHamiltonian(
                new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses
            )

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
                new_coeffs,
                new_ops,
                reorder_fn=self.reorder_fn,
                settings=self.settings,
                pulses=self.pulses,
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
        frequency (Union[float, Callable]): float or callable returning the frequency of a
            EM field. In the case of superconducting transmon systems this is the drive frequency.
            In the case of neutral atom rydberg systems this is the detuning between the drive frequency
            and energy gap.
        wires (Union[int, List[int]]): integer or list containing wire values that the EM field
            acts on
    """

    amplitude: Union[float, Callable]
    phase: Union[float, Callable]
    frequency: Union[float, Callable]
    wires: List[Wires]

    def __post_init__(self):
        self.wires = Wires(self.wires)

    def __eq__(self, other):
        return (
            self.amplitude == other.amplitude
            and self.phase == other.phase
            and self.frequency == other.frequency
            and self.wires == other.wires
        )


def amplitude_and_phase(trig_fn, amp, phase, hz_to_rads=2 * np.pi):
    r"""Wrapper function for combining amplitude and phase into a single callable
    (or constant if neither amplitude nor phase are callable). The factor of :math:`2 \pi` converts
    amplitude in Hz to amplitude in radians/second."""
    if not callable(amp) and not callable(phase):
        return hz_to_rads * amp * trig_fn(phase)
    return AmplitudeAndPhase(trig_fn, amp, phase, hz_to_rads=hz_to_rads)


# pylint:disable = too-few-public-methods
class AmplitudeAndPhase:
    """Class storing combined amplitude and phase callable if either or both
    of amplitude or phase are callable."""

    def __init__(self, trig_fn, amp, phase, hz_to_rads=2 * np.pi):
        # The factor of 2pi converts amplitude in Hz to amplitude in radians/second
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)

        def callable_amp_and_phase(params, t):
            return hz_to_rads * amp(params[0], t) * trig_fn(phase(params[1], t))

        def callable_amp(params, t):
            return hz_to_rads * amp(params, t) * trig_fn(phase)

        def callable_phase(params, t):
            return hz_to_rads * amp * trig_fn(phase(params, t))

        if self.amp_is_callable and self.phase_is_callable:
            self.func = callable_amp_and_phase

        elif self.amp_is_callable:
            self.func = callable_amp

        elif self.phase_is_callable:
            self.func = callable_phase

    def __call__(self, params, t):
        return self.func(params, t)
