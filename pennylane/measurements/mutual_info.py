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
from collections.abc import Sequence
from copy import copy

from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .measurements import StateMeasurement


class MutualInfoMP(StateMeasurement):
    """Measurement process that computes the mutual information between the provided wires.

    Please refer to :func:`pennylane.mutual_info` for detailed documentation.

    Args:
        wires (Sequence[.Wires]): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): base for the logarithm

    """

    def __str__(self):
        return "mutualinfo"

    _shortname = "mutualinfo"

    def _flatten(self):
        metadata = (("wires", tuple(self.raw_wires)), ("log_base", self.log_base))
        return (None, None), metadata

    def __init__(
        self,
        wires: Sequence[Wires] | None = None,
        id: str | None = None,
        log_base: float | None = None,
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
    def numeric_type(self):
        return float

    def map_wires(self, wire_map: dict):
        new_measurement = copy(self)
        new_measurement._wires = [
            Wires([wire_map.get(wire, wire) for wire in wires]) for wires in self.raw_wires
        ]
        return new_measurement

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()

    def process_state(self, state: TensorLike, wire_order: Wires):
        state = math.dm_from_state_vector(state)
        return math.mutual_info(
            state,
            indices0=list(self._wires[0]),
            indices1=list(self._wires[1]),
            c_dtype=state.dtype,
            base=self.log_base,
        )

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        return math.mutual_info(
            density_matrix,
            indices0=list(self._wires[0]),
            indices1=list(self._wires[1]),
            c_dtype=density_matrix.dtype,
            base=self.log_base,
        )


if MutualInfoMP._wires_primitive is not None:

    @MutualInfoMP._wires_primitive.def_impl
    def _(*all_wires, n_wires0, **kwargs):
        wires0 = all_wires[:n_wires0]
        wires1 = all_wires[n_wires0:]
        return type.__call__(MutualInfoMP, wires=(wires0, wires1), **kwargs)


def mutual_info(wires0, wires1, log_base=None) -> MutualInfoMP:
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

    .. seealso:: :func:`~pennylane.vn_entropy`, :func:`pennylane.math.mutual_info`
    """
    wires0 = Wires(wires0)
    wires1 = Wires(wires1)

    # the subsystems cannot overlap
    if not any(math.is_abstract(w) for w in wires0 + wires1) and [
        wire for wire in wires0 if wire in wires1
    ]:
        raise QuantumFunctionError("Subsystems for computing mutual information must not overlap.")
    return MutualInfoMP(wires=(wires0, wires1), log_base=log_base)
