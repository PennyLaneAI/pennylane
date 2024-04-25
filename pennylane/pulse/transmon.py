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
import numpy as np

import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires


# TODO ladder operators once there is qudit support
# pylint: disable=unused-argument
def a(wire, d=2):
    """creation operator"""
    return qml.s_prod(0.5, qml.X(wire)) + qml.s_prod(0.5j, qml.Y(wire))


def ad(wire, d=2):
    """annihilation operator"""
    return qml.s_prod(0.5, qml.X(wire)) + qml.s_prod(-0.5j, qml.Y(wire))


# pylint: disable=too-many-arguments
def transmon_interaction(
    qubit_freq: Union[float, list],
    connections: list,
    coupling: Union[float, list],
    wires: list,
    anharmonicity=None,
    d=2,
):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the circuit QED Hamiltonian of a
    superconducting transmon system.

    The Hamiltonian is given by

    .. math::

        H = \sum_{q\in \text{wires}} \omega_q b^\dagger_q b_q
        + \sum_{(i, j) \in \mathcal{C}} g_{ij} \left(b^\dagger_i b_j + b_j^\dagger b_i \right)
        + \sum_{q\in \text{wires}} \alpha_q b^\dagger_q b^\dagger_q b_q b_q

    where :math:`[b_p, b_q^\dagger] = \delta_{pq}` are creation and annihilation operators.
    The first term describes the effect of the dressed qubit frequencies ``qubit_freq`` :math:`= \omega_q/ (2\pi)`,
    the second term their ``coupling`` :math:`= g_{ij}/(2\pi)` and the last the
    ``anharmonicity`` :math:`= \alpha_q/(2\pi)`, which all can vary for
    different qubits. In practice, these operators are restricted to a finite dimension of the
    local Hilbert space (default ``d=2`` corresponds to qubits).
    In that case, the anharmonicity is set to :math:`\alpha=0` and ignored.

    The values of :math:`\omega` and :math:`\alpha` are typically around :math:`5 \times 2\pi \text{GHz}`
    and :math:`0.3 \times 2\pi \text{GHz}`, respectively.
    It is common for different qubits to be out of tune with different energy gaps. The coupling strength
    :math:`g` typically varies between :math:`[0.001, 0.1] \times 2\pi \text{GHz}`. For some example parameters,
    see e.g. `arXiv:1804.04073 <https://arxiv.org/abs/1804.04073>`_,
    `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`_, or `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`_.

    .. note:: Currently only supporting ``d=2`` with qudit support planned in the future. For ``d=2``, we have :math:`b:=\frac{1}{2}(\sigma^x + i \sigma^y)`.

    .. seealso::

        :func:`~.transmon_drive`

    Args:
        qubit_freq (Union[float, list[float], Callable]): List of dressed qubit frequencies. This should be in units
            of frequency (GHz), and will be converted to angular frequency :math:`\omega` internally where
            needed, i.e. multiplied by :math:`2 \pi`. When passing a single float all qubits are assumed to
            have that same frequency. When passing a parametrized function, it must have two
            arguments, the first one being the trainable parameters and the second one being time.
        connections (list[tuple(int)]): List of connections ``(i, j)`` between qubits i and j.
            When the wires in ``connections`` are not contained in ``wires``, a warning is raised.
        coupling (Union[float, list[float]]): List of coupling strengths. This should be in units
            of frequency (GHz), and will be converted to angular frequency internally where
            needed, i.e. multiplied by :math:`2 \pi`. Needs to match the length of ``connections``.
            When passing a single float need explicit ``wires``.
        anharmonicity (Union[float, list[float]]): List of anharmonicities. This should be in units
            of frequency (GHz), and will be converted to angular frequency internally where
            needed, i.e. multiplied by :math:`2 \pi`. Ignored when ``d=2``.
            When passing a single float all qubits are assumed to have that same anharmonicity.
        wires (list): Needs to be of the same length as qubit_freq. Note that there can be additional
            wires in the resulting operator from the ``connections``, which are treated independently.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can set up the transmon interaction Hamiltonian with uniform coefficients by passing ``float`` values.

    .. code-block::

        connections = [[0, 1], [1, 3], [2, 1], [4, 5]]
        H = qml.pulse.transmon_interaction(qubit_freq=0.5, connections=connections, coupling=1., wires=range(6))

    The resulting :class:`~.HardwareHamiltonian:` consists of ``4`` coupling terms and ``6`` qubits
    because there are six different wire indices in ``connections``.

    >>> print(H)
    HardwareHamiltonian: terms=10

    We can also provide individual values for each of the qubit energies and coupling strengths,
    here of order :math:`0.1 \times 2\pi\text{GHz}` and :math:`1 \times 2\pi\text{GHz}`, respectively.

    .. code-block::

        qubit_freqs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.]
        couplings= [1., 2., 3., 4.]
        H = qml.pulse.transmon_interaction(qubit_freq=qubit_freqs,
                                           connections=connections,
                                           coupling=couplings,
                                           wires=range(6))

    The interaction term is dependent only on the typically fixed transmon energies and coupling strengths.
    Executing this as a pulse program via :func:`~.evolve` would correspond to all driving fields being turned off.
    To add a driving field, see :func:`~.transmon_drive`.

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
    if callable(qubit_freq) or qml.math.ndim(qubit_freq) == 0:
        qubit_freq = [qubit_freq] * n_wires
    elif len(qubit_freq) != n_wires:
        raise ValueError(
            f"Number of qubit frequencies in {qubit_freq} does not match the provided wires = {wires}"
        )

    if qml.math.ndim(coupling) == 0:
        coupling = [coupling] * len(connections)
    if len(coupling) != len(connections):
        raise ValueError(
            f"Number of coupling terms {coupling} does not match the provided connections = {connections}"
        )

    settings = TransmonSettings(connections, qubit_freq, coupling, anharmonicity=anharmonicity)

    omega = [callable_freq_to_angular(f) if callable(f) else (2 * np.pi * f) for f in qubit_freq]
    g = [callable_freq_to_angular(c) if callable(c) else (2 * np.pi * c) for c in coupling]

    # qubit term
    coeffs = list(omega)
    observables = [ad(i, d) @ a(i, d) for i in wires]

    # coupling term
    coeffs += list(g)
    observables += [ad(i, d) @ a(j, d) + ad(j, d) @ a(i, d) for (i, j) in connections]

    # TODO Qudit support. Currently not supported but will be in the future.
    # if d>2:
    #     if qml.math.ndim(anharmonicity)==0:
    #         anharmonicity = [anharmonicity] * n_wires
    #     if len(anharmonicity) != n_wires:
    #         raise ValueError(f"Number of qubit anharmonicities anharmonicity = {anharmonicity} does not match the provided wires = {wires}")
    #     # anharmonicity term
    #     alpha = [2 * np.pi * a for a in anharmonicity]
    #     coeffs += list(alpha)
    #     observables += [ad(i, d) @ ad(i, d) @ a(i, d) @ a(i, d) for i in wires]

    return HardwareHamiltonian(
        coeffs, observables, settings=settings, reorder_fn=_reorder_AmpPhaseFreq
    )


def callable_freq_to_angular(fn):
    """Add a factor of 2pi to a callable result to convert from Hz to rad/s"""

    def angular_fn(p, t):
        return 2 * np.pi * fn(p, t)

    return angular_fn


@dataclass
class TransmonSettings:
    """Dataclass that contains the information of a Transmon setup.

    .. seealso:: :func:`transmon_interaction`

    Args:
            connections (List): List `[[idx_q0, idx_q1], ..]` of connected qubits (wires)
            qubit_freq (List[float, Callable]):
            coupling (List[list, TensorLike, Callable]):
            anharmonicity (List[float, Callable]):

    """

    connections: List
    qubit_freq: Union[float, Callable]
    coupling: Union[list, TensorLike, Callable]
    anharmonicity: Union[float, Callable]

    def __eq__(self, other):
        return (
            qml.math.all(self.connections == other.connections)
            and qml.math.all(self.qubit_freq == other.qubit_freq)
            and qml.math.all(self.coupling == other.coupling)
            and qml.math.all(self.anharmonicity == other.anharmonicity)
        )

    def __add__(self, other):
        if other is not None:
            new_connections = list(self.connections) + list(other.connections)
            new_qubit_freq = list(self.qubit_freq) + list(other.qubit_freq)
            new_coupling = list(self.coupling) + list(other.coupling)
            new_anh = list(self.anharmonicity) + list(other.anharmonicity)
            return TransmonSettings(
                new_connections, new_qubit_freq, new_coupling, anharmonicity=new_anh
            )

        return self


def transmon_drive(amplitude, phase, freq, wires, d=2):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the drive term of a transmon qubit.

    The Hamiltonian is given by

    .. math::

        \Omega(t) \sin\left(\phi(t) + \nu t\right) \sum_q Y_q

    where :math:`\{Y_q\}` are the Pauli-Y operators on ``wires`` :math:`\{q\}`.
    The arguments ``amplitude``, ``phase`` and ``freq`` correspond to :math:`\Omega / (2\pi)`, :math:`\phi`
    and :math:`\nu / (2\pi)`, respectively, and can all be either fixed numbers (``float``) or depend on time
    (``callable``). If they are time-dependent, they need to abide by the restrictions imposed
    in :class:`ParametrizedHamiltonian` and have a signature of two parameters, ``(params, t)``.

    Together with the qubit :math:`Z` terms in :func:`transmon_interaction`, driving with this term can generate
    :math:`X` and :math:`Y` rotations by setting :math:`\phi` accordingly and driving on resonance
    (see eqs. (79) - (92) in `1904.06560 <https://arxiv.org/abs/1904.06560>`_).
    Further, it can generate entangling gates by driving at cross-resonance with a coupled qubit
    (see eqs. (131) - (137) in `1904.06560 <https://arxiv.org/abs/1904.06560>`_).
    Such a coupling is described in :func:`transmon_interaction`.

    For realistic simulations, one may restrict the amplitude, phase and drive frequency parameters.
    For example, the authors in `2008.04302 <https://arxiv.org/abs/2008.04302>`_ impose the restrictions of
    a maximum amplitude :math:`\Omega_{\text{max}} = 20 \text{MHz}` and the carrier frequency to deviate at most
    :math:`\nu - \omega = \pm 1 \text{GHz}` from the qubit frequency :math:`\omega`
    (see :func:`~.transmon_interaction`).
    The phase :math:`\phi(t)` is typically a slowly changing function of time compared to :math:`\Omega(t)`.

    .. note:: Currently only supports ``d=2`` with qudit support planned in the future.
        For ``d>2``, we have :math:`Y \mapsto i (\sigma^- - \sigma^+)`
        with lowering and raising operators  :math:`\sigma^{\mp}`.

    .. note:: Due to convention in the respective fields, we omit the factor :math:`\frac{1}{2}` present in the related constructor :func:`~.rydberg_drive`

    .. seealso::

        :func:`~.rydberg_drive`, :func:`~.transmon_interaction`

    Args:
        amplitude (Union[float, callable]): Float or callable representing the amplitude of the driving field.
            This should be in units of frequency (GHz), and will be converted to angular frequency
            :math:`\Omega(t)` internally where needed, i.e. multiplied by :math:`2 \pi`.
        phase (Union[float, callable]): Float or callable returning phase :math:`\phi(t)` (in radians).
            Can be a fixed number (``float``) or depend on time (``callable``)
        freq (Union[float, callable]): Float or callable representing the frequency of the driving field.
            This should be in units of frequency (GHz), and will be converted to angular frequency
            :math:`\nu(t)` internally where needed, i.e. multiplied by :math:`2 \pi`.
        wires (Union[int, list[int]]): Label of the qubit that the drive acts upon. Can be a list of multiple wires.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can construct a drive term acting on qubit ``0`` in the following way. We parametrize the amplitude and phase
    via :math:`\Omega(t)/(2 \pi) = A \times \sin^2(\pi t)` and :math:`\phi(t) = \phi_0 (t - \frac{1}{2})`. The squared
    sine ensures that the amplitude will be strictly positive (a requirement for some hardware). For simplicity, we
    set the drive frequency to zero :math:`\nu=0`.

    .. code-block:: python3

        def amp(A, t):
            return A * jnp.exp(-t**2)

        def phase(phi0, t):
            return phi0

        freq = 0

        H = qml.pulse.transmon_drive(amp, phase, freq, 0)

        t = 0.
        A = 1.
        phi0 = jnp.pi/2
        params = [A, phi0]

    Evaluated at :math:`t = 0` with the parameters :math:`A = 1` and :math:`\phi_0 = \pi/2` we obtain
    :math:`2 \pi A \exp(0) \sin(\pi/2 + 0)\sigma^y = 2 \pi \sigma^y`.

    >>> H(params, t)
    6.283185307179586 * Y(0)

    We can combine ``transmon_drive()`` with :func:`~.transmon_interaction` to create a full driven transmon Hamiltonian.
    Let us look at a chain of three transmon qubits that are coupled with their direct neighbors. We provide all
    frequencies in GHz (conversion to angular frequency, i.e. multiplication by :math:`2 \pi`, is taken care of
    internally where needed).

    We use values around :math:`\omega = 5 \times 2\pi \text{GHz}` for resonant frequencies, and coupling strenghts
    on the order of around :math:`g = 0.01 \times 2\pi \text{GHz}`.

    We parametrize the drive Hamiltonians for the qubits with amplitudes as squared sinusodials of
    maximum amplitude :math:`A`, and constant drive frequencies of value :math:`\nu`. We set the
    phase to zero :math:`\phi=0`, and we make the parameters :math:`A` and :math:`\nu` trainable
    for every qubit. We simulate the evolution for a time window of :math:`[0, 5]\text{ns}`.

    .. code-block:: python3

        qubit_freqs = [5.1, 5., 5.3]
        connections = [[0, 1], [1, 2]]  # qubits 0 and 1 are coupled, as are 1 and 2
        g = [0.02, 0.05]
        H = qml.pulse.transmon_interaction(qubit_freqs, connections, g, wires=range(3))

        def amp(max_amp, t): return max_amp * jnp.sin(t) ** 2
        freq = qml.pulse.constant  # Parametrized constant frequency
        phase = 0.0
        time = 5

        for q in range(3):
            H += qml.pulse.transmon_drive(amp, phase, freq, q)  # Parametrized drive for each qubit

        dev = qml.device("default.qubit.jax", wires=range(3))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H)(params, time)
            return qml.expval(qml.Z(0) + qml.Z(1) + qml.Z(2))

    We evaluate the Hamiltonian with some arbitrarily chosen maximum amplitudes (here on the order of :math:`0.5 \times 2\pi \text{GHz}`)
    and set the drive frequency equal to the qubit frequencies. Note how the order of the construction
    of ``H`` determines the order with which the parameters need to be passed to
    :class:`~.ParametrizedHamiltonian` and :func:`~.evolve`. We made the drive frequencies
    trainable parameters by providing constant callables through :func:`~.pulse.constant` instead of fixed values (like the phase).
    This allows us to differentiate with respect to both the maximum amplitudes and the frequencies and optimize them.

    >>> max_amp0, max_amp1, max_amp2 = [0.5, 0.3, 0.6]
    >>> fr0, fr1, fr2 = qubit_freqs
    >>> params = [max_amp0, fr0, max_amp1, fr1, max_amp2, fr2]
    >>> qnode(params)
    Array(-1.57851962, dtype=float64)
    >>> jax.grad(qnode)(params)
    [Array(-13.50193649, dtype=float64),
     Array(3.1112141, dtype=float64),
     Array(16.40286521, dtype=float64),
     Array(-4.30485667, dtype=float64),
     Array(4.75813949, dtype=float64),
     Array(3.43272354, dtype=float64)]

    """
    if d != 2:
        raise NotImplementedError(
            "Currently only supports qubits (d=2). Qutrits and qudits support is planned in the future."
        )

    wires = Wires(wires)

    # TODO: use creation and annihilation operators when introducing qutrits
    # Note that exp(-iw)a* + exp(iw)a = cos(w)X - sin(w)Y for a=1/2(X+iY)
    # We compute the `coeffs` and `observables` of the EM field
    coeffs = [AmplitudeAndPhaseAndFreq(qml.math.sin, amplitude, phase, freq)]

    drive_y_term = sum(qml.Y(wire) for wire in wires)

    observables = [drive_y_term]

    pulses = [HardwarePulse(amplitude, phase, freq, wires)]

    return HardwareHamiltonian(coeffs, observables, pulses=pulses, reorder_fn=_reorder_AmpPhaseFreq)


# pylint:disable = too-few-public-methods,too-many-return-statements
class AmplitudeAndPhaseAndFreq:
    """Class storing combined amplitude, phase and freq callables"""

    def __init__(self, trig_fn, amp, phase, freq, hz_to_rads=2 * np.pi):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)
        self.freq_is_callable = callable(freq)

        # all 3 callable

        if self.amp_is_callable and self.phase_is_callable and self.freq_is_callable:

            def callable_amp_and_phase_and_freq(params, t):
                return (
                    hz_to_rads
                    * amp(params[0], t)
                    * trig_fn(phase(params[1], t) + hz_to_rads * freq(params[2], t) * t)
                )

            self.func = callable_amp_and_phase_and_freq
            return

        # 2 out of 3 callable

        if self.amp_is_callable and self.phase_is_callable:

            def callable_amp_and_phase(params, t):
                return (
                    hz_to_rads
                    * amp(params[0], t)
                    * trig_fn(phase(params[1], t) + hz_to_rads * freq * t)
                )

            self.func = callable_amp_and_phase
            return

        if self.amp_is_callable and self.freq_is_callable:

            def callable_amp_and_freq(params, t):
                return (
                    hz_to_rads
                    * amp(params[0], t)
                    * trig_fn(phase + hz_to_rads * freq(params[1], t) * t)
                )

            self.func = callable_amp_and_freq
            return

        if self.phase_is_callable and self.freq_is_callable:

            def callable_phase_and_freq(params, t):
                return (
                    hz_to_rads
                    * amp
                    * trig_fn(phase(params[0], t) + hz_to_rads * freq(params[1], t) * t)
                )

            self.func = callable_phase_and_freq
            return

        # 1 out of 3 callable

        if self.amp_is_callable:

            def callable_amp(params, t):
                return hz_to_rads * amp(params[0], t) * trig_fn(phase + hz_to_rads * freq * t)

            self.func = callable_amp
            return

        if self.phase_is_callable:

            def callable_phase(params, t):
                return hz_to_rads * amp * trig_fn(phase(params[0], t) + hz_to_rads * freq * t)

            self.func = callable_phase
            return

        if self.freq_is_callable:

            def callable_freq(params, t):
                return hz_to_rads * amp * trig_fn(phase + hz_to_rads * freq(params[0], t) * t)

            self.func = callable_freq
            return

        # 0 out of 3 callable
        # (the remaining coeff is still callable due to explicit time dependence)

        def no_callable(_, t):
            return hz_to_rads * amp * trig_fn(phase + hz_to_rads * freq * t)

        self.func = no_callable

    def __call__(self, params, t):
        return self.func(params, t)


def _reorder_AmpPhaseFreq(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude and/or callable freq.

    Consolidates amplitude, phase and freq parameters if they are callable,
    and duplicates parameters since they will be passed to two operators in the Hamiltonian"""

    reordered_params = []

    coeff_idx = 0
    params_idx = 0

    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhaseAndFreq):
                is_callables = [
                    coeff.phase_is_callable,
                    coeff.amp_is_callable,
                    coeff.freq_is_callable,
                ]

                num_callables = sum(is_callables)

                # package parameters according to how many coeffs are callable
                reordered_params.extend([params[params_idx : params_idx + num_callables]])

                coeff_idx += 1
                params_idx += num_callables

            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1

    return reordered_params
