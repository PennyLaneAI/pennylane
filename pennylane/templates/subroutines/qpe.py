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
"""
Contains the QuantumPhaseEstimation template.
"""
# pylint: disable=too-many-arguments,arguments-differ
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.operation import AnyWires, Operator
from pennylane.resource.error import ErrorOperation, SpectralNormError


class QuantumPhaseEstimation(ErrorOperation):
    r"""Performs the
    `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given a unitary matrix :math:`U`, this template applies the circuit for quantum phase
    estimation. The unitary is applied to the qubits specified by ``target_wires`` and :math:`n`
    qubits are used for phase estimation as specified by ``estimation_wires``.

    .. figure:: ../../_static/templates/subroutines/qpe.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        unitary (array or Operator): the phase estimation unitary, specified as a matrix or an
            :class:`~.Operator`
        target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary.
            If the unitary is specified as an operator, the target wires should already have been
            defined as part of the operator. In this case, target_wires should not be specified.
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Raises:
        QuantumFunctionError: if the ``target_wires`` and ``estimation_wires`` share a common
            element, or if ``target_wires`` are specified for an operator unitary.

    .. details::
        :title: Usage Details

        This circuit can be used to perform the standard quantum phase estimation algorithm, consisting
        of the following steps:

        #. Prepare ``target_wires`` in a given state. If ``target_wires`` are prepared in an eigenstate
           of :math:`U` that has corresponding eigenvalue :math:`e^{2 \pi i \theta}` with phase
           :math:`\theta \in [0, 1)`, this algorithm will measure :math:`\theta`. Other input states can
           be prepared more generally.
        #. Apply the ``QuantumPhaseEstimation`` circuit.
        #. Measure ``estimation_wires`` using :func:`~.probs`, giving a probability distribution over
           measurement outcomes in the computational basis.
        #. Find the index of the largest value in the probability distribution and divide that number by
           :math:`2^{n}`. This number will be an estimate of :math:`\theta` with an error that decreases
           exponentially with the number of qubits :math:`n`.

        Note that if :math:`\theta \in (-1, 0]`, we can estimate the phase by again finding the index
        :math:`i` found in step 4 and calculating :math:`\theta \approx \frac{1 - i}{2^{n}}`. An example
        of this case is below.

        Consider the matrix corresponding to a rotation from an :class:`~.RX` gate:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import QuantumPhaseEstimation
            from pennylane import numpy as np

            phase = 5
            target_wires = [0]
            unitary = qml.RX(phase, wires=0).matrix()

        The ``phase`` parameter can be estimated using ``QuantumPhaseEstimation``. An example is
        shown below using a register of five phase-estimation qubits:

        .. code-block:: python

            n_estimation_wires = 5
            estimation_wires = range(1, n_estimation_wires + 1)

            dev = qml.device("default.qubit", wires=n_estimation_wires + 1)

            @qml.qnode(dev)
            def circuit():
                # Start in the |+> eigenstate of the unitary
                qml.Hadamard(wires=target_wires)

                QuantumPhaseEstimation(
                    unitary,
                    target_wires=target_wires,
                    estimation_wires=estimation_wires,
                )

                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

            # Need to rescale phase due to convention of RX gate
            phase_estimated = 4 * np.pi * (1 - phase_estimated)

        We can also perform phase estimation on an operator. Note that since operators are defined
        with target wires, the target wires should not be provided for the QPE.

        .. code-block:: python


            # use the product to specify compound operators
            unitary = qml.RX(np.pi / 2, wires=[0]) @ qml.CNOT(wires=[0, 1])
            eigenvector = np.array([-1/2, -1/2, 1/2, 1/2])

            n_estimation_wires = 5
            estimation_wires = range(2, n_estimation_wires + 2)
            target_wires = [0, 1]

            dev = qml.device("default.qubit", wires=n_estimation_wires + 2)

            @qml.qnode(dev)
            def circuit():
                qml.StatePrep(eigenvector, wires=target_wires)
                QuantumPhaseEstimation(
                    unitary,
                    estimation_wires=estimation_wires,
                )
                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

    """

    num_wires = AnyWires
    grad_method = None

    # pylint: disable=no-member
    def _flatten(self):
        data = (self.hyperparameters["unitary"],)
        metadata = (self.hyperparameters["estimation_wires"],)
        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata) -> "QuantumPhaseEstimation":
        return cls(data[0], estimation_wires=metadata[0])

    def __init__(self, unitary, target_wires=None, estimation_wires=None, id=None):
        if isinstance(unitary, Operator):
            # If the unitary is expressed in terms of operators, do not provide target wires
            if target_wires is not None:
                raise qml.QuantumFunctionError(
                    "The unitary is expressed as an operator, which already has target wires "
                    "defined, do not additionally specify target wires."
                )
            target_wires = unitary.wires

        elif target_wires is None:
            raise qml.QuantumFunctionError(
                "Target wires must be specified if the unitary is expressed as a matrix."
            )

        else:
            unitary = qml.QubitUnitary(unitary, wires=target_wires)

        # Estimation wires are required, but kept as an optional argument so that it can be
        # placed after target_wires for backwards compatibility.
        if estimation_wires is None:
            raise qml.QuantumFunctionError("No estimation wires specified.")

        target_wires = qml.wires.Wires(target_wires)
        estimation_wires = qml.wires.Wires(estimation_wires)
        wires = target_wires + estimation_wires

        if any(wire in target_wires for wire in estimation_wires):
            raise qml.QuantumFunctionError(
                "The target wires and estimation wires must not overlap."
            )

        self._hyperparameters = {
            "unitary": unitary,
            "target_wires": target_wires,
            "estimation_wires": estimation_wires,
        }

        super().__init__(wires=wires, id=id)

    @property
    def target_wires(self):
        """The target wires of the QPE"""
        return self._hyperparameters["target_wires"]

    @property
    def estimation_wires(self):
        """The estimation wires of the QPE"""
        return self._hyperparameters["estimation_wires"]

    def error(self):
        """The QPE error computed from the spectral norm error of the input unitary operator.

        **Example**

        >>> class CustomOP(qml.resource.ErrorOperation):
        ...    def error(self):
        ...       return qml.resource.SpectralNormError(0.005)
        >>> Op = CustomOP(wires=[0])
        >>> QPE = QuantumPhaseEstimation(Op, estimation_wires = range(1, 5))
        >>> QPE.error()
        SpectralNormError(0.075)

        """
        base_unitary = self._hyperparameters["unitary"]
        if not isinstance(base_unitary, ErrorOperation):
            return SpectralNormError(0.0)

        unitary_error = base_unitary.error().error

        sequence_error = qml.math.array(
            [unitary_error * (2**i) for i in range(len(self.estimation_wires) - 1, -1, -1)],
            like=qml.math.get_interface(unitary_error),
        )

        additive_error = qml.math.sum(sequence_error)

        return SpectralNormError(additive_error)

    # pylint: disable=protected-access
    def map_wires(self, wire_map: dict):
        new_op = super().map_wires(wire_map)
        new_op._hyperparameters["unitary"] = qml.map_wires(
            new_op._hyperparameters["unitary"], wire_map
        )

        new_op._hyperparameters["estimation_wires"] = [
            wire_map.get(wire, wire) for wire in self.estimation_wires
        ]
        new_op._hyperparameters["target_wires"] = [
            wire_map.get(wire, wire) for wire in self.target_wires
        ]

        return new_op

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters["unitary"])
        context.append(self)
        return self

    @staticmethod
    def compute_decomposition(
        wires, unitary, target_wires, estimation_wires
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the QPE circuit as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.QuantumPhaseEstimation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the QPE circuit acts on
            unitary (Operator): the phase estimation unitary, specified as an operator
            target_wires (Any or Iterable[Any]): the target wires to apply the unitary
            estimation_wires (Any or Iterable[Any]): the wires to be used for phase estimation

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = [qml.Hadamard(w) for w in estimation_wires]
        pow_ops = (qml.pow(unitary, 2**i) for i in range(len(estimation_wires) - 1, -1, -1))
        op_list.extend(qml.ctrl(op, w) for op, w in zip(pow_ops, estimation_wires))
        op_list.append(qml.adjoint(qml.templates.QFT(wires=estimation_wires)))

        return op_list
