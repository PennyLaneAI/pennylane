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
This module contains the qml.state measurement.
"""
from typing import Sequence, Optional

import pennylane as qml
from pennylane.wires import Wires, WireError

from .measurements import State, StateMeasurement


def state() -> "StateMP":
    r"""Quantum state in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its state. A
    ``wires`` argument should *not* be provided since ``state()`` always returns a pure state
    describing all wires in the device.

    Note that the output shape of this measurement process depends on the
    number of wires defined for the device.

    Returns:
        StateMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            return qml.state()

    Executing this QNode:

    >>> circuit()
    array([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

    The returned array is in lexicographic order. Hence, we have a :math:`1/\sqrt{2}` amplitude
    in both :math:`|00\rangle` and :math:`|01\rangle`.

    .. note::

        Differentiating :func:`~pennylane.state` is currently only supported when using the
        classical backpropagation differentiation method (``diff_method="backprop"``) with a
        compatible device.

    .. details::
        :title: Usage Details

        A QNode with the ``qml.state`` output can be used in a cost function which
        is then differentiated:

        >>> dev = qml.device('default.qubit', wires=2)
        >>> @qml.qnode(dev, diff_method="backprop")
        ... def test(x):
        ...     qml.RY(x, wires=[0])
        ...     return qml.state()
        >>> def cost(x):
        ...     return np.abs(test(x)[0])
        >>> cost(x)
        0.9987502603949663
        >>> qml.grad(cost)(x)
        tensor(-0.02498958, requires_grad=True)
    """
    return StateMP()


def density_matrix(wires) -> "DensityMatrixMP":
    r"""Quantum density matrix in the computational basis.

    This function accepts no observables and instead instructs the QNode to return its density
    matrix or reduced density matrix. The ``wires`` argument gives the possibility
    to trace out a part of the system. It can result in obtaining a mixed state, which can be
    only represented by the reduced density matrix.

    Args:
        wires (Sequence[int] or int): the wires of the subsystem

    Returns:
        DensityMatrixMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Y(0)
            qml.Hadamard(wires=1)
            return qml.density_matrix([0])

    Executing this QNode:

    >>> circuit()
    array([[0.+0.j 0.+0.j]
        [0.+0.j 1.+0.j]])

    The returned matrix is the reduced density matrix, where system 1 is traced out.

    .. note::

        Calculating the derivative of :func:`~pennylane.density_matrix` is currently only supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device.
    """
    wires = Wires(wires)
    return DensityMatrixMP(wires=wires)


class StateMP(StateMeasurement):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`state` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Optional[Wires] = None, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @property
    def return_type(self):
        return State

    @property
    def numeric_type(self):
        return complex

    def shape(self, device, shots):
        num_shot_elements = (
            sum(s.copies for s in shots.shot_vector) if shots.has_partitioned_shots else 1
        )
        dim = 2 ** len(self.wires) if self.wires else 2 ** len(device.wires)
        return (dim,) if num_shot_elements == 1 else tuple((dim,) for _ in range(num_shot_elements))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # pylint:disable=redefined-outer-name
        is_tf_interface = qml.math.get_deep_interface(state) == "tensorflow"
        wires = self.wires
        if not wires or wire_order == wires:
            return qml.math.cast(state, "complex128") if is_tf_interface else state + 0.0j

        if set(wires) != set(wire_order):
            raise WireError(
                f"Unexpected unique wires {Wires.unique_wires([wires, wire_order])} found. "
                f"Expected wire order {wire_order} to be a rearrangement of {wires}"
            )

        shape = (2,) * len(wires)
        flat_shape = (2 ** len(wires),)
        desired_axes = [wire_order.index(w) for w in wires]
        if qml.math.ndim(state) == 2:  # batched state
            batch_size = qml.math.shape(state)[0]
            shape = (batch_size,) + shape
            flat_shape = (batch_size,) + flat_shape
            desired_axes = [0] + [i + 1 for i in desired_axes]

        state = qml.math.reshape(state, shape)
        state = qml.math.transpose(state, desired_axes)
        state = qml.math.reshape(state, flat_shape)
        return qml.math.cast(state, "complex128") if is_tf_interface else state + 0.0j


class DensityMatrixMP(StateMP):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`density_matrix` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Wires, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    def shape(self, device, shots):
        num_shot_elements = (
            sum(s.copies for s in shots.shot_vector) if shots.has_partitioned_shots else 1
        )

        dim = 2 ** len(self.wires)
        return (
            (dim, dim)
            if num_shot_elements == 1
            else tuple((dim, dim) for _ in range(num_shot_elements))
        )

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # pylint:disable=redefined-outer-name
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        kwargs = {"indices": mapped_wires, "c_dtype": "complex128"}
        if not qml.math.is_abstract(state) and qml.math.any(qml.math.iscomplex(state)):
            kwargs["c_dtype"] = state.dtype
        return qml.math.reduce_statevector(state, **kwargs)
