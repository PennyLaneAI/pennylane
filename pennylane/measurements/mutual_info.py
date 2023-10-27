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
This module contains the qml.mutual_info measurement.
"""
from copy import copy
from typing import Sequence, Optional

import pennylane as qml
from pennylane.wires import Wires

from .measurements import MutualInfo, StateMeasurement


def mutual_info(wires0, wires1, log_base=None):
    r"""Mutual information between the subsystems prior to measurement:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        wires0 (Sequence[int] or int): the wires of the first subsystem
        wires1 (Sequence[int] or int): the wires of the second subsystem
        log_base (float): Base for the logarithm.

    Returns:
        MutualInfoMP: measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_mutual(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

    Executing this QNode:

    >>> circuit_mutual(np.pi/2)
    1.3862943611198906

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(circuit_mutual)(param)
    tensor(1.24645048, requires_grad=True)

    .. note::

        Calculating the derivative of :func:`~.mutual_info` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`~.vn_entropy`, :func:`pennylane.qinfo.transforms.mutual_info` and :func:`pennylane.math.mutual_info`
    """
    wires0 = qml.wires.Wires(wires0)
    wires1 = qml.wires.Wires(wires1)

    # the subsystems cannot overlap
    if [wire for wire in wires0 if wire in wires1]:
        raise qml.QuantumFunctionError(
            "Subsystems for computing mutual information must not overlap."
        )
    return MutualInfoMP(wires=(wires0, wires1), log_base=log_base)


class MutualInfoMP(StateMeasurement):
    """Measurement process that computes the mutual information between the provided wires.

    Please refer to :func:`mutual_info` for detailed documentation.

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

    def __repr__(self):
        return f"MutualInfo(wires0={self.raw_wires[0].tolist()}, wires1={self.raw_wires[1].tolist()}, log_base={self.log_base})"

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
        return MutualInfo

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
        return qml.math.mutual_info(
            state,
            indices0=list(self._wires[0]),
            indices1=list(self._wires[1]),
            c_dtype=state.dtype,
            base=self.log_base,
        )
