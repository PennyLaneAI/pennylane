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
This module contains the qml.vn_entanglement_entropy measurement.
"""
from copy import copy
from typing import Optional, Sequence

import pennylane as qml
from pennylane.wires import Wires

from .measurements import VnEntanglementEntropy, StateMeasurement


def vn_entanglement_entropy(wires0, wires1, log_base=None):
    r"""Measures the `Von Neumann entanglement entropy <https://en.wikipedia.org/wiki/Entropy_of_entanglement>`_
    of a quantum state:

    .. math::

        S(\rho_A) = -\text{Tr}[\rho_A \log \rho_A] = -\text{Tr}[\rho_B \log \rho_B] = S(\rho_B)

    where :math:`S` is the Von Neumann entropy; :math:`\rho_A = \text{Tr}_B [\rho_{AB}]` and
    :math:`\rho_B = \text{Tr}_A [\rho_{AB}]` are the reduced density matrices for each partition.

    The Von Neumann entanglement entropy is a measure of the degree of quantum entanglement between
    two subsystems constituting a pure bipartite quantum state. The entropy of entanglement is the
    Von Neumann entropy of the reduced density matrix for any of the subsystems. If it is non-zero,
    it indicates the two subsystems are entangled.

    Args:
        wires0 (Sequence[int] or int): the wires of the first subsystem
        wires1 (Sequence[int] or int): the wires of the second subsystem
        log_base (float): Base for the logarithm.

    Returns:
        VnEntanglementEntropyMP: measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, 0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.vn_entanglement_entropy([0], [1])

    Executing this QNode:

    >>> circuit(1.967)
    0.16389850379003218

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit)(param)
    tensor(-0.62322524, requires_grad=True)

    .. note::

        Calculating the derivative of :func:`~.vn_entanglement_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`~.vn_entropy` and :func:`pennylane.math.vn_entanglement_entropy`
    """
    wires0 = qml.wires.Wires(wires0)
    wires1 = qml.wires.Wires(wires1)

    # the subsystems cannot overlap
    if not any(qml.math.is_abstract(w) for w in wires0 + wires1) and [
        wire for wire in wires0 if wire in wires1
    ]:
        raise qml.QuantumFunctionError(
            "Subsystems for computing entanglement entropy must not overlap."
        )
    return VnEntanglementEntropyMP(wires=(wires0, wires1), log_base=log_base)


class VnEntanglementEntropyMP(StateMeasurement):
    """Measurement process that computes the Von Neumann entanglement entropy between the provided wires.

    Please refer to :func:`~.vn_entanglement_entropy` for detailed documentation.

    Args:
        wires (Sequence[.Wires]): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): base for the logarithm

    """

    def _flatten(self):
        metadata = (("wires", tuple(self.raw_wires)), ("log_base", self.log_base))
        return (None, None), metadata

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Optional[Sequence[Wires]] = None,
        id: Optional[str] = None,
        log_base: Optional[float] = None,
    ):
        self.log_base = log_base
        super().__init__(wires=wires, id=id)

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, wires: Sequence, **kwargs):
        if cls._wires_primitive is None:  # pragma: no cover
            # just a safety check
            return type.__call__(cls, wires=wires, **kwargs)  # pragma: no cover
        return cls._wires_primitive.bind(*wires[0], *wires[1], n_wires0=len(wires[0]), **kwargs)

    def __repr__(self):
        return f"VnEntanglementEntropy(wires0={self.raw_wires[0].tolist()}, wires1={self.raw_wires[1].tolist()}, log_base={self.log_base})"

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            tuple(self.raw_wires[0].tolist()),
            tuple(self.raw_wires[1].tolist()),
            self.log_base,
        )

        return hash(fingerprint)

    @property
    def return_type(self):
        return VnEntanglementEntropy

    @property
    def numeric_type(self):
        return float

    def map_wires(self, wire_map: dict):
        new_measurement = copy(self)
        new_measurement._wires = [
            Wires([wire_map.get(wire, wire) for wire in wires]) for wires in self.raw_wires
        ]
        return new_measurement

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        state = qml.math.dm_from_state_vector(state)
        return qml.math.vn_entanglement_entropy(
            state,
            indices0=list(self._wires[0]),
            indices1=list(self._wires[1]),
            c_dtype=state.dtype,
            base=self.log_base,
        )


if VnEntanglementEntropyMP._wires_primitive is not None:

    @VnEntanglementEntropyMP._wires_primitive.def_impl
    def _(*all_wires, n_wires0, **kwargs):
        wires0 = all_wires[:n_wires0]
        wires1 = all_wires[n_wires0:]
        return type.__call__(VnEntanglementEntropyMP, wires=(wires0, wires1), **kwargs)
