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
import copy
from typing import Sequence

import pennylane as qml
from pennylane.operation import Operator
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
    1.2464504802804612

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
    return MutualInfoMP(wires=[wires0, wires1], log_base=log_base)


class MutualInfoMP(StateMeasurement):
    """Measurement process that computes the mutual information between the provided wires.

    Please refer to :func:`mutual_info` for detailed documentation.

    Args:
        obs (.Observable): The observable that is to be measured as part of the
            measurement process. Not all measurement processes require observables (for
            example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): base for the logarithm

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        obs: Operator = None,
        wires=None,
        eigvals=None,
        id=None,
        log_base=None,
    ):
        self.log_base = log_base
        super().__init__(obs=obs, wires=wires, eigvals=eigvals, id=id)

    @property
    def return_type(self):
        return MutualInfo

    @property
    def numeric_type(self):
        return float

    def shape(self, device=None):
        if qml.active_return():
            return self._shape_new(device)
        if device is None or device.shot_vector is None:
            return (1,)
        num_shot_elements = sum(s.copies for s in device.shot_vector)
        return (num_shot_elements,)

    def _shape_new(self, device=None):
        if device is None or device.shot_vector is None:
            return ()
        num_shot_elements = sum(s.copies for s in device.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        return qml.math.mutual_info(
            state,
            indices0=list(self._wires[0]),
            indices1=list(self._wires[1]),
            c_dtype=state.dtype,
            base=self.log_base,
        )

    def __copy__(self):
        return self.__class__(
            obs=copy.copy(self.obs),
            wires=self._wires,
            eigvals=self._eigvals,
            log_base=self.log_base,
        )
