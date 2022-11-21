# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
This module contains the qml.classical_shadow measurement.
"""
from collections.abc import Iterable
from string import ascii_letters as ABC
from typing import Sequence

import numpy as np

import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.interfaces import set_shots
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from .measurements import CustomMeasurement, MeasurementShapeError, Shadow, ShadowExpval


def shadow_expval(H, k=1, seed_recipes=True):
    r"""Compute expectation values using classical shadows in a differentiable manner.

    The canonical way of computing expectation values is to simply average the expectation values for each local snapshot, :math:`\langle O \rangle = \sum_t \text{tr}(\rho^{(t)}O) / T`.
    This corresponds to the case ``k=1``. In the original work, `2002.08953 <https://arxiv.org/abs/2002.08953>`_, it has been proposed to split the ``T`` measurements into ``k`` equal
    parts to compute the median of means. For the case of Pauli measurements and Pauli observables, there is no advantage expected from setting ``k>1``.

    Args:
        H (Union[Iterable, :class:`~.pennylane.Hamiltonian`, :class:`~.pennylane.operation.Tensor`]): Observable or
            iterable of observables to compute the expectation value over.
        k (int): Number of equal parts to split the shadow's measurements to compute the median of means. ``k=1`` (default) corresponds to simply taking the mean over all measurements.
        seed_recipes (bool): If True, a seed will be generated that
            is used for the randomly sampled Pauli measurements. This is to
            ensure that the same recipes are used when a tape containing this
            measurement is copied. Different seeds are still generated for
            different constructed tapes.

    Returns:
        float: expectation value estimate.

    .. note::

        This measurement uses the measurement :func:`~.pennylane.classical_shadow` and the class :class:`~.pennylane.ClassicalShadow` for post-processing
        internally to compute expectation values. In order to compute correct gradients using PennyLane's automatic differentiation,
        you need to use this measurement.

    **Example**

    .. code-block:: python3

        H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

        dev = qml.device("default.qubit", wires=range(2), shots=10000)
        @qml.qnode(dev)
        def qnode(x, obs):
            qml.Hadamard(0)
            qml.CNOT((0,1))
            qml.RX(x, wires=0)
            return qml.shadow_expval(obs)

        x = np.array(0.5, requires_grad=True)

    We can compute the expectation value of H as well as its gradient in the usual way.

    >>> qnode(x, H)
    tensor(1.827, requires_grad=True)
    >>> qml.grad(qnode)(x, H)
    -0.44999999999999984

    In `shadow_expval`, we can pass a list of observables. Note that each qnode execution internally performs one quantum measurement, so be sure
    to include all observables that you want to estimate from a single measurement in the same execution.

    >>> Hs = [H, qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
    >>> qnode(x, Hs)
    [ 1.88586e+00,  4.50000e-03,  1.32000e-03, -1.92000e-03]
    >>> qml.jacobian(qnode)(x, Hs)
    [-0.48312, -0.00198, -0.00375,  0.00168]
    """
    seed = np.random.randint(2**30) if seed_recipes else None
    return ClassicalShadow(ShadowExpval, H=H, seed=seed, k=k)


def classical_shadow(wires, seed_recipes=True):
    """
    The classical shadow measurement protocol.

    The protocol is described in detail in the paper `Predicting Many Properties of a Quantum System from Very Few Measurements <https://arxiv.org/abs/2002.08953>`_.
    This measurement process returns the randomized Pauli measurements (the ``recipes``)
    that are performed for each qubit and snapshot as an integer:

    - 0 for Pauli X,
    - 1 for Pauli Y, and
    - 2 for Pauli Z.

    It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
    is sampled, and 1 if the -1 eigenvalue is sampled.

    The device shots are used to specify the number of snapshots. If ``T`` is the number
    of shots and ``n`` is the number of qubits, then both the measured bits and the
    Pauli measurements have shape ``(T, n)``.

    Args:
        wires (Sequence[int]): the wires to perform Pauli measurements on
        seed_recipes (bool): If True, a seed will be generated that
            is used for the randomly sampled Pauli measurements. This is to
            ensure that the same recipes are used when a tape containing this
            measurement is copied. Different seeds are still generated for
            different constructed tapes.

    **Example**

    Consider the following QNode that prepares a Bell state and performs a classical
    shadow measurement:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=5)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.classical_shadow(wires=[0, 1])

    Executing this QNode produces the sampled bits and the Pauli measurements used:

    >>> bits, recipes = circuit()
    >>> bits
    tensor([[0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 1]], dtype=uint8, requires_grad=True)
    >>> recipes
    tensor([[2, 2],
            [0, 2],
            [1, 0],
            [0, 2],
            [0, 2]], dtype=uint8, requires_grad=True)

    .. details::
        :title: Usage Details

        Consider again the QNode in the above example. Since the Pauli observables are
        randomly sampled, executing this QNode again would produce different bits and Pauli recipes:

        >>> bits, recipes = circuit()
        >>> bits
        tensor([[0, 1],
                [0, 1],
                [0, 0],
                [0, 1],
                [1, 1]], dtype=uint8, requires_grad=True)
        >>> recipes
        tensor([[1, 0],
                [2, 1],
                [2, 2],
                [1, 0],
                [0, 0]], dtype=uint8, requires_grad=True)

        To use the same Pauli recipes for different executions, the :class:`~.tape.QuantumTape`
        interface should be used instead:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2, shots=5)

            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.classical_shadow(wires=[0, 1])

        >>> bits1, recipes1 = qml.execute([tape], device=dev, gradient_fn=None)[0][0]
        >>> bits2, recipes2 = qml.execute([tape], device=dev, gradient_fn=None)[0][0]
        >>> np.all(recipes1 == recipes2)
        True
        >>> np.all(bits1 == bits2)
        False

        If using different Pauli recipes is desired for the :class:`~.tape.QuantumTape` interface,
        the ``seed_recipes`` flag should be explicitly set to ``False``:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2, shots=5)

            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.classical_shadow(wires=[0, 1], seed_recipes=False)

        >>> bits1, recipes1 = qml.execute([tape], device=dev, gradient_fn=None)[0][0]
        >>> bits2, recipes2 = qml.execute([tape], device=dev, gradient_fn=None)[0][0]
        >>> np.all(recipes1 == recipes2)
        False
        >>> np.all(bits1 == bits2)
        False
    """
    wires = Wires(wires)

    seed = np.random.randint(2**30) if seed_recipes else None
    return ClassicalShadow(Shadow, wires=wires, seed=seed)


class ClassicalShadow(CustomMeasurement):
    """Represents a classical shadow measurement process occurring at the end of a
    quantum variational circuit.

    This has the same arguments as the base class MeasurementProcess, plus other additional
    arguments specific to the classical shadow protocol.

    Args:
        args (tuple[Any]): Positional arguments passed to :class:`~.pennylane.measurements.MeasurementProcess`
        seed (Union[int, None]): The seed used to generate the random measurements
        H (:class:`~.pennylane.Hamiltonian` or :class:`~.pennylane.operation.Tensor`): Observable
            to compute the expectation value over. Only used when ``return_type`` is ``ShadowExpval``.
        k (int): Number of equal parts to split the shadow's measurements to compute the median of means.
            ``k=1`` corresponds to simply taking the mean over all measurements. Only used
            when ``return_type`` is ``ShadowExpval``.
        kwargs (dict[Any, Any]): Additional keyword arguments passed to :class:`~.pennylane.measurements.MeasurementProcess`
    """

    def __init__(self, *args, seed=None, H=None, k=1, **kwargs):
        self.seed = seed
        self.H = H
        self.k = k
        super().__init__(*args, **kwargs)

    def process(self, tape: QuantumScript, device):
        """
        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation leverages vectorization and offers a significant speed-up over
        the generic implementation.

        .. Note::

            This method internally calls ``np.einsum`` which supports at most 52 indices,
            thus the classical shadow measurement for this device supports at most 52
            qubits.

        .. seealso:: :func:`~.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadow): The classical shadow measurement process
            circuit (~.tapes.QuantumTape): The quantum tape that is being executed

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        if isinstance(device, DefaultQubit):
            return self._process_state(state=device.state, shots=device.shots)

        return self._process_tape(tape, device)

    def _process_tape(self, tape: QuantumScript, device):
        wires = self.wires
        n_snapshots = device.shots
        seed = self.seed

        with set_shots(self, shots=1):
            # slow implementation but works for all devices
            n_qubits = len(wires)
            mapped_wires = np.array(self.map_wires(wires))

            if seed is not None:
                # seed the random measurement generation so that recipes
                # are the same for different executions with the same seed
                rng = np.random.RandomState(seed)
                recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
            else:
                recipes = np.random.randint(0, 3, size=(n_snapshots, n_qubits))
            obs_list = [qml.PauliX, qml.PauliY, qml.PauliZ]

            outcomes = np.zeros((n_snapshots, n_qubits))

            for t in range(n_snapshots):
                # compute rotations for the Pauli measurements
                rotations = [
                    rot
                    for wire_idx, wire in enumerate(wires)
                    for rot in obs_list[recipes[t][wire_idx]].compute_diagonalizing_gates(
                        wires=wire
                    )
                ]

                device.reset()
                device.apply(tape.operations, rotations=tape.diagonalizing_gates + rotations)

                outcomes[t] = device.generate_samples()[0][mapped_wires]

        return qml.math.cast(qml.math.stack([outcomes, recipes]), dtype=np.int8)

    def _process_state(self, state: Sequence[complex], shots: int):
        wires = self.wires
        seed = self.seed

        n_qubits = len(wires)
        n_snapshots = shots
        device_qubits = len(self.wires)
        mapped_wires = np.array(self.map_wires(wires))

        if seed is not None:
            # seed the random measurement generation so that recipes
            # are the same for different executions with the same seed
            rng = np.random.RandomState(seed)
            recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
        else:
            recipes = np.random.randint(0, 3, size=(n_snapshots, n_qubits))

        obs_list = qml.math.stack(
            [
                qml.PauliX.compute_matrix(),
                qml.PauliY.compute_matrix(),
                qml.PauliZ.compute_matrix(),
            ]
        )
        uni_list = qml.math.stack(
            [
                qml.Hadamard.compute_matrix(),
                qml.Hadamard.compute_matrix() @ qml.RZ.compute_matrix(-np.pi / 2),
                qml.Identity.compute_matrix(),
            ]
        )
        obs = obs_list[recipes]
        uni = uni_list[recipes]

        # There's a significant speedup if we use the following iterative
        # process to perform the randomized Pauli measurements:
        #   1. Randomly generate Pauli observables for all snapshots for
        #      a single qubit (e.g. the first qubit).
        #   2. Compute the expectation of each Pauli observable on the first
        #      qubit by tracing out all other qubits.
        #   3. Sample the first qubit based on each Pauli expectation.
        #   4. For all snapshots, determine the collapsed state of the remaining
        #      qubits based on the sample result.
        #   4. Repeat iteratively until no qubits are remaining.
        #
        # Observe that after the first iteration, the second qubit will become the
        # "first" qubit in the process. The advantage to this approach as opposed to
        # simulataneously computing the Pauli expectations for each qubit is that
        # the partial traces are computed over iteratively smaller subsystems, leading
        # to a significant speed-up.

        # transpose the state so that the measured wires appear first
        unmeasured_wires = [i for i in range(len(self.wires)) if i not in mapped_wires]
        transposed_state = np.transpose(state, axes=mapped_wires.tolist() + unmeasured_wires)

        outcomes = np.zeros((n_snapshots, n_qubits))
        stacked_state = qml.math.stack([transposed_state for _ in range(n_snapshots)])

        for i in range(n_qubits):

            # trace out every qubit except the first
            first_qubit_state = qml.math.einsum(
                f"{ABC[device_qubits - i + 1]}{ABC[:device_qubits - i]},{ABC[device_qubits - i + 1]}{ABC[device_qubits - i]}{ABC[1:device_qubits - i]}"
                f"->{ABC[device_qubits - i + 1]}a{ABC[device_qubits - i]}",
                stacked_state,
                qml.math.conj(stacked_state),
            )

            # sample the observables on the first qubit
            probs = (qml.math.einsum("abc,acb->a", first_qubit_state, obs[:, i]) + 1) / 2
            samples = np.random.uniform(0, 1, size=probs.shape) > probs
            outcomes[:, i] = samples

            # collapse the state of the remaining qubits; the next qubit in line
            # becomes the first qubit for the next iteration
            rotated_state = qml.math.einsum("ab...,acb->ac...", stacked_state, uni[:, i])
            stacked_state = rotated_state[np.arange(n_snapshots), qml.math.cast(samples, np.int8)]

            # re-normalize the collapsed state
            norms = np.sqrt(
                np.sum(
                    np.abs(stacked_state) ** 2, tuple(range(1, device_qubits - i)), keepdims=True
                )
            )
            stacked_state /= norms

        return qml.math.cast(qml.math.stack([outcomes, recipes]), dtype=np.int8)

    @property
    def numeric_type(self):
        """The Python numeric type of the measurement result.

        Returns:
            type: This is ``int`` when the return type is ``Shadow``,
            and ``float`` when the return type is ``ShadowExpval``.
        """
        return int if self.return_type is Shadow else float

    def shape(self, device=None):
        """The expected output shape of the ClassicalShadow.

        Args:
            device (.Device): a PennyLane device to use for determining the shape

        Returns:
            tuple: the output shape; this is ``(2, T, n)`` when the return type
            is ``Shadow``, where ``T`` is the number of device shots and ``n`` is
            the number of measured wires, and is a scalar when the return type
            is ``ShadowExpval``

        Raises:
            MeasurementShapeError: when a device is not provided and the return
            type is ``Shadow``, since the output shape is dependent on the device.
        """
        # the return value of expval is always a scalar
        if self.return_type is ShadowExpval:
            return (1,)

        # otherwise, the return type requires a device
        if device is None:
            raise MeasurementShapeError(
                "The device argument is required to obtain the shape of a classical "
                "shadow measurement process."
            )

        # the first entry of the tensor represents the measured bits,
        # and the second indicate the indices of the unitaries used
        return (1, 2, device.shots, len(self.wires))

    @property
    def wires(self):
        r"""The wires the measurement process acts on.

        This is the union of all the Wires objects of the measurement.
        """
        if self.return_type is Shadow:
            return self._wires

        if isinstance(self.H, Iterable):
            return Wires.all_wires([h.wires for h in self.H])

        return self.H.wires

    def queue(self, context=qml.QueuingManager):
        """Append the measurement process to an annotated queue, making sure
        the observable is not queued"""
        if self.H is not None:
            Hs = self.H if isinstance(self.H, Iterable) else [self.H]
            for H in Hs:
                context.update_info(H, owner=self)
            context.append(self, owns=Hs)
        else:
            context.append(self)

        return self

    def __copy__(self):
        obj = super().__copy__()
        obj.seed = self.seed
        obj.H = self.H
        obj.k = self.k
        return obj
