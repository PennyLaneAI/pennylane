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
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

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

    return HardwareHamiltonian(
        coeffs, observables, settings=settings, reorder_fn=_reorder_AmpPhaseFreq
    )


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


def transmon_drive(amplitude, phase, freq, wires, d=2):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the drive term of a transmon qubit.

    The Hamiltonian is given by

    .. math::

        \Omega(t) \left(e^{i (\phi(t) + \nu t)} a_q + e^{-i (\phi(t) + \nu t)} a^\dagger_q\right)

    where :math:`[a^\dagger_p, a_q] = i \delta_{pq}` are bosonic creation and annihilation operators
    and :math:`q` is the qubit label (``wires``).
    The arguments ``amplitude``, ``phase`` and ``freq`` correspond to :math:`\Omega`, :math:`\phi`
    and :math:`\nu`, respectively, and can all be either fixed numbers (``float``) or depend on time
    (``callable``). In the case they are time-dependent, they need to abide by the restrictions imposed
    in :class:`ParametrizedHamiltonian` and have a signature of two parameters, ``(params, t)``.

    For realistic simulations, one may restrict the amplitude, phase and drive frequency parameters.
    For example, the authors in `2008.04302 <https://arxiv.org/abs/2008.04302>`_ impose the restrictions of
    a maximum amplitude :math:`\Omega_{\text{max}} = 20 \text{MHz}` and the carrier frequency to deviate at most
    :math:`\nu - \omega = \pm 1 \text{GHz}` from the qubit frequency :math:`\omega`
    (see :func:`~.transmon_interaction`).
    The phase :math:`\phi(t)` is typically a slowly changing function of time compared to :math:`\Omega(t)`.

    .. note:: Currently only supports ``d=2`` with qudit support planned in the future. For ``d=2`` we have :math:`a:=\frac{1}{2}(\sigma^x + i \sigma^y)`.

    .. seealso::

        :func:`~.drive`, :func:`~.rydberg_drive`, :func:`~.transmon_interaction`

    Args:
        amplitude (Union[float, callable]): The amplitude :math:`\Omega(t)`.
            Can be a fixed number (``float``) or depend on time (``callable``)
        phase (Union[float, callable]): The phase :math:`\phi(t)`.
            Can be a fixed number (``float``) or depend on time (``callable``)
        freq (Union[float, callable]): The drive frequency :math:`\nu`.
            Can be a fixed number (``float``) or ``callable``. Physically it does not make sense for the drive frequency
            to depend on time. The option for it to be ``callable`` is to allow the it to be a trainable parameter.
        wires (Union[int, list[int]]): Label of the qubit that the drive acts upon. Can be a list of multiple wires.
        d (int): Local Hilbert space dimension. Defaults to ``d=2`` and is currently the only supported value.

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the transmon interaction

    **Example**

    We can construct a drive term acting on qubit ``0`` in the following way. We parametrize the amplitude and phase
    via :math:`\Omega(t) = \Omega \sin(\pi t)` and :math:`\phi(t) = \phi (t - \frac{1}{2})`.

    .. code-block:: python3

        def amp(Omega, t): return Omega * jnp.sin(jnp.pi*t)
        def phase(phi, t): return phi * (t - 0.5)

        H = qml.pulse.transmon_drive(amp, phase, 0, 0)

        t = 0.5
        Omega = 0.1
        phi = 0.001
        params = [Omega, phi]

    Evaluated at :math:`t = \frac{1}{2}` with the parameters :math:`\Omega = 0.1` and :math:`\phi = 10^{-3}` we obtain
    :math:`\Omega \left(\frac{1}{2}(\sigma^x + i \sigma^y) + \frac{1}{2}(\sigma^x + i \sigma^y)\right) = \Omega \sigma^x`.

    >>> H(params, t)
    (0.1*(PauliX(wires=[0]))) + (0.0*(-1*(PauliY(wires=[0]))))

    We can combine ``transmon_drive()`` with :func:`~.transmon_interaction` to create a full driven transmon Hamiltonian.
    Let us look at a chain of three transmon qubits that are coupled with their direct neighbors. We provide all numbers in
    :math:`2\pi\text{GHz}`. We parametrize the amplitude as a sinusodial and make the maximum amplitude
    as well as the drive frequency trainable parameters. We simulate the evolution for a time window of :math:`[0, 5]\text{ns}`.

    .. code-block:: python3

        omega = [5.1, 5., 5.3]
        connections = [[0, 1], [1, 2]]
        g = [0.02, 0.05]
        H = qml.pulse.transmon_interaction(omega, connections, g, wires=range(3))

        def amp(Omega, t): return Omega * jnp.sin(t)
        def freq(fr, t): return fr
        phase = 0.
        t=2

        H += qml.pulse.transmon_drive(amp, phase, freq, 0)
        H += qml.pulse.transmon_drive(amp, phase, freq, 1)
        H += qml.pulse.transmon_drive(amp, phase, freq, 2)

        dev = qml.device("default.qubit.jax", wires=range(3))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(H)(params, t=5.)
            return qml.expval(qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliZ(2))

    We evaluate the Hamiltonian with some arbitrarily chosen maximum amplitudes and set
    the drive frequency equal to the qubit frequencies. Note how the order of the construction
    of ``H`` determines the order with which the parameters need to be passed to
    :class:`~.ParametrizedHamiltonian` and :func:`~.evolve`. By making the drive frequencies
    trainable parameters by providing a constant callable above instead of the fixed values,
    we can differentiate with respect to them.

    >>> Omega0, Omega1, Omega2 = [0.5, 0.3, 0.6]
    >>> fr0, fr1, fr2 = omega
    >>> params = [Omega0, fr0, Omega1, fr1, Omega2, fr2]
    >>> qnode(params)
    Array(2.25098131, dtype=float64)
    >>> jax.grad(qnode)(params)
    [Array(-0.96356123, dtype=float64),
     Array(-0.0189564, dtype=float64),
     Array(-0.58581467, dtype=float64),
     Array(0.24023855, dtype=float64),
     Array(-1.30009675, dtype=float64),
     Array(-0.10709503, dtype=float64)]

    """
    if d != 2:
        raise NotImplementedError(
            "Currently only supports qubits (d=2). Qutrits and qudits support is planned in the future."
        )

    wires = Wires(wires)

    # TODO: use creation and annihilation operators when introducing qutrits
    # We compute the `coeffs` and `observables` of the EM field
    coeffs = [
        AmplitudeAndPhaseAndFreq(qml.math.cos, amplitude, phase, freq),
        AmplitudeAndPhaseAndFreq(qml.math.sin, amplitude, phase, freq),
    ]

    drive_x_term = sum(qml.PauliX(wire) for wire in wires)
    drive_y_term = sum(-qml.PauliY(wire) for wire in wires)

    observables = [drive_x_term, drive_y_term]

    return HardwareHamiltonian(coeffs, observables, reorder_fn=_reorder_AmpPhaseFreq)


# pylint:disable = too-few-public-methods,too-many-return-statements
class AmplitudeAndPhaseAndFreq:
    """Class storing combined amplitude, phase and freq callables"""

    def __init__(self, trig_fn, amp, phase, freq):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)
        self.freq_is_callable = callable(freq)

        # all 3 callable

        if self.amp_is_callable and self.phase_is_callable and self.freq_is_callable:

            def callable_amp_and_phase_and_freq(params, t):
                return amp(params[0], t) * trig_fn(phase(params[1], t) + freq(params[2], t) * t)

            self.func = callable_amp_and_phase_and_freq
            return

        # 2 out of 3 callable

        if self.amp_is_callable and self.phase_is_callable:

            def callable_amp_and_phase(params, t):
                return amp(params[0], t) * trig_fn(phase(params[1], t) + freq * t)

            self.func = callable_amp_and_phase
            return

        if self.amp_is_callable and self.freq_is_callable:

            def callable_amp_and_freq(params, t):
                return amp(params[0], t) * trig_fn(phase + freq(params[1], t) * t)

            self.func = callable_amp_and_freq
            return

        if self.phase_is_callable and self.freq_is_callable:

            def callable_phase_and_freq(params, t):
                return amp * trig_fn(phase(params[0], t) + freq(params[1], t) * t)

            self.func = callable_phase_and_freq
            return

        # 1 out of 3 callable

        if self.amp_is_callable:

            def callable_amp(params, t):
                return amp(params, t) * trig_fn(phase + freq * t)

            self.func = callable_amp
            return

        if self.phase_is_callable:

            def callable_phase(params, t):
                return amp * trig_fn(phase(params, t) + freq * t)

            self.func = callable_phase
            return

        if self.freq_is_callable:

            def callable_freq(params, t):
                return amp * trig_fn(phase + freq(params, t) * t)

            self.func = callable_freq
            return

        # 0 out of 3 callable 
        # (the remaining coeff is still callable due to explicit time dependence)

        def no_callable(_, t):
            return amp * trig_fn(phase + freq * t)

        self.func = no_callable

    def __call__(self, params, t):
        return self.func(params, t)


def _reorder_AmpPhaseFreq(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude and/or callable freq.

    Consolidates amplitude, phase and freq parameters in they are callable,
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

                # all 3 parameters are callable
                if sum(is_callables) == 3:
                    reordered_params.append(
                        [params[params_idx], params[params_idx + 1], params[params_idx + 2]]
                    )
                    reordered_params.append(
                        [params[params_idx], params[params_idx + 1], params[params_idx + 2]]
                    )
                    coeff_idx += 2
                    params_idx += 3

                # 2 of 3 parameters are callable
                elif sum(is_callables) == 2:
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    coeff_idx += 2
                    params_idx += 2

                # 1 of 3 parameters is callable
                elif sum(is_callables) == 1:
                    reordered_params.append(params[params_idx])
                    reordered_params.append(params[params_idx])
                    coeff_idx += 2
                    params_idx += 1

                # in case of no callable, the coeff is still callable due to the explicit freq*t dependence
                elif sum(is_callables) == 0:
                    reordered_params.append(None)
                    reordered_params.append(None)
                    coeff_idx += 2
                    params_idx += 0

            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1

    return reordered_params
