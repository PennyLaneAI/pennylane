# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the quantum_monte_carlo transform.
"""
from copy import copy
from typing import Sequence, Callable

import pennylane as qml
from pennylane import PauliX, Hadamard, MultiControlledX, CZ, adjoint
from pennylane.wires import Wires
from pennylane.templates import QFT
from pennylane.transforms.core import transform


def _apply_controlled_z(wires, control_wire, work_wires):
    r"""Provides the circuit to apply a controlled version of the :math:`Z` gate defined in
    `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The multi-qubit gate :math:`Z = I - 2|0\rangle \langle 0|` can be performed using the
    conventional multi-controlled-Z gate with an additional bit flip on each qubit before and after.

    This function returns the multi-controlled-Z gate via a multi-controlled-X gate by picking an
    arbitrary target wire to perform the X and adding a Hadamard on that wire either side of the
    transformation.

    Additional control from ``control_wire`` is then included within the multi-controlled-X gate.

    Args:
        wires (Wires): the wires on which the Z gate is applied
        control_wire (Wires): the control wire from the register of phase estimation qubits
        work_wires (Wires): the work wires used in the decomposition
    """
    target_wire = wires[0]
    updated_operations = []
    updated_operations.append(PauliX(target_wire))
    updated_operations.append(Hadamard(target_wire))

    control_values = [0] * (len(wires) - 1) + [1]
    control_wires = wires[1:] + control_wire
    updated_operations.append(
        MultiControlledX(
            wires=[*control_wires, target_wire],
            control_values=control_values,
            work_wires=work_wires,
        )
    )

    updated_operations.append(Hadamard(target_wire))
    updated_operations.append(PauliX(target_wire))

    return updated_operations


def _apply_controlled_v(target_wire, control_wire):
    """Provides the circuit to apply a controlled version of the :math:`V` gate defined in
    `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The :math:`V` gate is simply a Pauli-Z gate applied to the ``target_wire``, i.e., the ancilla
    wire in which the expectation value is encoded.

    The controlled version of this gate is then a CZ gate.

    Args:
        target_wire (Wires): the ancilla wire in which the expectation value is encoded
        control_wire (Wires): the control wire from the register of phase estimation qubits
    """
    return [CZ(wires=[control_wire[0], target_wire[0]])]


@transform
def apply_controlled_Q(
    tape: qml.tape.QuantumTape, wires, target_wire, control_wire, work_wires
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Applies the transform that performs a controlled version of the :math:`\mathcal{Q}` unitary
    defined in `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The input ``tape`` should be the quantum circuit corresponding to the :math:`\mathcal{F}` unitary
    in the paper above. This function transforms this circuit into a controlled version of the
    :math:`\mathcal{Q}` unitary, which forms part of the quantum Monte Carlo algorithm. The
    :math:`\mathcal{Q}` unitary encodes the target expectation value as a phase in one of its
    eigenvalues. This phase can be estimated using quantum phase estimation (see
    :class:`~.QuantumPhaseEstimation` for more details).

    Args:
        tape (QNode or QuantumTape or Callable): the quantum circuit that applies quantum operations
            according to the :math:`\mathcal{F}` unitary used as part of quantum Monte Carlo estimation
        wires (Union[Wires or Sequence[int]]): the wires acted upon by the ``fn`` circuit
        target_wire (Union[Wires, int]): The wire in which the expectation value is encoded. Must be
            contained within ``wires``.
        control_wire (Union[Wires, int]): the control wire from the register of phase estimation
            qubits
        work_wires (Union[Wires, Sequence[int], or int]): additional work wires used when
            decomposing :math:`\mathcal{Q}`

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will perform control on :math:`\mathcal{Q}` unitary.

    Raises:
        ValueError: if ``target_wire`` is not in ``wires``
    """
    operations = tape.operations.copy()
    updated_operations = []

    with qml.queuing.QueuingManager.stop_recording():
        op_inv = [adjoint(copy(op)) for op in reversed(operations)]

        wires = Wires(wires)
        target_wire = Wires(target_wire)
        control_wire = Wires(control_wire)
        work_wires = Wires(work_wires) if work_wires is not None else Wires([])

        if not wires.contains_wires(target_wire):
            raise ValueError("The target wire must be contained within wires")

        updated_operations.extend(
            _apply_controlled_v(target_wire=target_wire, control_wire=control_wire)
        )
        updated_operations.extend(op_inv)
        updated_operations.extend(
            _apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires)
        )
        updated_operations.extend(operations)
        updated_operations.extend(
            _apply_controlled_v(target_wire=target_wire, control_wire=control_wire)
        )
        updated_operations.extend(op_inv)
        updated_operations.extend(
            _apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires)
        )
        updated_operations.extend(operations)

    tape = type(tape)(updated_operations, tape.measurements, shots=tape.shots)
    return [tape], lambda x: x[0]


@transform
def quantum_monte_carlo(
    tape: qml.tape.QuantumTape, wires, target_wire, estimation_wires
) -> (Sequence[qml.tape.QuantumTape], Callable):
    r"""Applies the transform
    `quantum Monte Carlo estimation <https://arxiv.org/abs/1805.00109>`__ algorithm.

    The input `tape`` should be the quantum circuit corresponding to the :math:`\mathcal{F}` unitary
    in the paper above. This unitary encodes the probability distribution and random variable onto
    ``wires`` so that measurement of the ``target_wire`` provides the expectation value to be
    estimated. The quantum Monte Carlo algorithm then estimates the expectation value using quantum
    phase estimation (check out :class:`~.QuantumPhaseEstimation` for more details), using the
    ``estimation_wires``.

    .. note::

        A complementary approach for quantum Monte Carlo is available with the
        :class:`~.QuantumMonteCarlo` template.

        The ``quantum_monte_carlo`` transform is intended for
        use when you already have the circuit for performing :math:`\mathcal{F}` set up, and is
        compatible with resource estimation and potential hardware implementation. The
        :class:`~.QuantumMonteCarlo` template is only compatible with
        simulators, but may perform faster and is suited to quick prototyping.

    Args:
        tape (QNode or QuantumTape or Callable): the quantum circuit that applies quantum operations according to the
            :math:`\mathcal{F}` unitary used as part of quantum Monte Carlo estimation
        wires (Union[Wires or Sequence[int]]): the wires acted upon by the ``fn`` circuit
        target_wire (Union[Wires, int]): The wire in which the expectation value is encoded. Must be
            contained within ``wires``.
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires used for phase estimation

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will perform the quantum Monte Carlo estimation.


    Raises:
        ValueError: if ``wires`` and ``estimation_wires`` share a common wire

    .. details::
        :title: Usage Details

        Consider an input quantum circuit ``fn`` that performs the unitary

        .. math::

            \mathcal{F} = \mathcal{R} \mathcal{A}.

        .. figure:: ../../_static/ops/f.svg
            :align: center
            :width: 15%
            :target: javascript:void(0);

        Here, the unitary :math:`\mathcal{A}` prepares a probability distribution :math:`p(i)` of
        dimension :math:`M = 2^{m}` over :math:`m \geq 1` qubits:

        .. math::

            \mathcal{A}|0\rangle^{\otimes m} = \sum_{i \in X} p(i) |i\rangle,

        where :math:`X = \{0, 1, \ldots, M - 1\}` and :math:`|i\rangle` is the basis state
        corresponding to :math:`i`. The :math:`\mathcal{R}` unitary imprints the
        result of a function :math:`f: X \rightarrow [0, 1]` onto an ancilla qubit:

        .. math::

            \mathcal{R}|i\rangle |0\rangle = |i\rangle \left(\sqrt{1 - f(i)} |0\rangle + \sqrt{f(i)}|1\rangle\right).

        Following `this <https://arxiv.org/abs/1805.00109>`__ paper,
        the probability of measuring the state :math:`|1\rangle` in the final
        qubit is

        .. math::

            \mu = \sum_{i \in X} p(i) f(i).

        However, it is possible to measure :math:`\mu` more efficiently using quantum Monte Carlo
        estimation. This function transforms an input quantum circuit ``fn`` that performs the
        unitary :math:`\mathcal{F}` to a larger circuit for measuring :math:`\mu` using the quantum
        Monte Carlo algorithm.

        .. figure:: ../../_static/ops/qmc.svg
            :align: center
            :width: 60%
            :target: javascript:void(0);

        The algorithm proceeds as follows:

        #. The probability distribution :math:`p(i)` is encoded using a unitary :math:`\mathcal{A}`
           applied to the first :math:`m` qubits specified by ``wires``.
        #. The function :math:`f(i)` is encoded onto the ``target_wire`` using a unitary
           :math:`\mathcal{R}`.
        #. The unitary :math:`\mathcal{Q}` is defined with eigenvalues
           :math:`e^{\pm 2 \pi i \theta}` such that the phase :math:`\theta` encodes the expectation
           value through the equation :math:`\mu = (1 + \cos (\pi \theta)) / 2`. The circuit in
           steps 1 and 2 prepares an equal superposition over the two states corresponding to the
           eigenvalues :math:`e^{\pm 2 \pi i \theta}`.
        #. The circuit returned by this function is applied so that :math:`\pm\theta` can be
           estimated by finding the probabilities of the :math:`n` estimation wires. This in turn
           allows for the estimation of :math:`\mu`.

        Visit `Rebentrost et al. (2018)
        <https://arxiv.org/abs/1805.00109>`__ for further details.
        In this algorithm, the number of applications :math:`N` of the :math:`\mathcal{Q}` unitary
        scales as :math:`2^{n}`. However, due to the use of quantum phase estimation, the error
        :math:`\epsilon` scales as :math:`\mathcal{O}(2^{-n})`. Hence,

        .. math::

            N = \mathcal{O}\left(\frac{1}{\epsilon}\right).

        This scaling can be compared to standard Monte Carlo estimation, where :math:`N` samples are
        generated from the probability distribution and the average over :math:`f` is taken. In that
        case,

        .. math::

            N =  \mathcal{O}\left(\frac{1}{\epsilon^{2}}\right).

        Hence, the quantum Monte Carlo algorithm has a quadratically improved time complexity with
        :math:`N`.

        **Example**

        Consider a standard normal distribution :math:`p(x)` and a function
        :math:`f(x) = \sin ^{2} (x)`. The expectation value of :math:`f(x)` is
        :math:`\int_{-\infty}^{\infty}f(x)p(x)dx \approx 0.432332`. This number can be approximated by
        discretizing the problem and using the quantum Monte Carlo algorithm.

        First, the problem is discretized:

        .. code-block:: python

            from scipy.stats import norm

            m = 5
            M = 2 ** m

            xmax = np.pi  # bound to region [-pi, pi]
            xs = np.linspace(-xmax, xmax, M)

            probs = np.array([norm().pdf(x) for x in xs])
            probs /= np.sum(probs)

            func = lambda i: np.sin(xs[i]) ** 2
            r_rotations = np.array([2 * np.arcsin(np.sqrt(func(i))) for i in range(M)])

        The ``quantum_monte_carlo`` transform can then be used:

        .. code-block::

            from pennylane.templates.state_preparations.mottonen import (
                _apply_uniform_rotation_dagger as r_unitary,
            )

            n = 6
            N = 2 ** n

            a_wires = range(m)
            wires = range(m + 1)
            target_wire = m
            estimation_wires = range(m + 1, n + m + 1)

            dev = qml.device("default.qubit", wires=(n + m + 1))

            def fn():
                qml.templates.MottonenStatePreparation(np.sqrt(probs), wires=a_wires)
                r_unitary(qml.RY, r_rotations, control_wires=a_wires[::-1], target_wire=target_wire)

            @qml.qnode(dev)
            def qmc():
                qml.quantum_monte_carlo(fn, wires, target_wire, estimation_wires)()
                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(qmc()[:int(N / 2)]) / N

        The estimated value can be retrieved using the formula :math:`\mu = (1-\cos(\pi \theta))/2`

        >>> (1 - np.cos(np.pi * phase_estimated)) / 2
        0.42663476277231915

        It is also possible to explore the resources required to perform the quantum Monte Carlo
        algorithm

        >>> qml.specs(qmc, expansion_strategy="device")()
        {'resources': Resources(
            num_wires=12,
            num_gates=31882,
            gate_types=defaultdict(<class 'int'>, {'RY': 7747, 'CNOT': 7874, 'Hadamard': 258, 'CZ': 126, 'Adjoint(CNOT)': 7812, 'Adjoint(RY)': 7686, 'PauliX': 252, 'MultiControlledX': 126, 'Adjoint(QFT)': 1}),
            gate_sizes=defaultdict(<class 'int'>, {1: 15943, 2: 15812, 7: 126, 6: 1}), depth=30610, shots=Shots(total_shots=None, shot_vector=()),
         ),
         'num_observables': 1,
         'num_diagonalizing_gates': 0,
         'num_trainable_params': 15433,
         'num_device_wires': 12,
         'device_name': 'default.qubit',
         'expansion_strategy': 'gradient',
         'gradient_options': {},
         'interface': 'auto',
         'diff_method': 'best',
         'gradient_fn': 'backprop'}
    """
    operations = tape.operations.copy()
    wires = Wires(wires)
    target_wire = Wires(target_wire)
    estimation_wires = Wires(estimation_wires)

    if Wires.shared_wires([wires, estimation_wires]):
        raise ValueError("No wires can be shared between the wires and estimation_wires registers")

    updated_operations = []
    with qml.queuing.QueuingManager.stop_recording():
        updated_operations.extend(operations)
        for i, control_wire in enumerate(estimation_wires):
            updated_operations.append(Hadamard(control_wire))

            # Find wires eligible to be used as helper wires
            work_wires = estimation_wires.toset() - {control_wire}
            n_reps = 2 ** (len(estimation_wires) - (i + 1))

            tapes_q, _ = apply_controlled_Q(
                tape,
                wires=wires,
                target_wire=target_wire,
                control_wire=control_wire,
                work_wires=work_wires,
            )
            tape_q = tapes_q[0]
            for _ in range(n_reps):
                updated_operations.extend(tape_q.operations)

        updated_operations.append(adjoint(QFT(wires=estimation_wires), lazy=False))
    updated_tape = type(tape)(updated_operations, tape.measurements, shots=tape.shots)
    return [updated_tape], lambda x: x[0]
