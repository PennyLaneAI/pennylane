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
from collections.abc import Sequence

from pennylane import math
from pennylane.exceptions import WireError
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .measurements import StateMeasurement


class StateMP(StateMeasurement):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`pennylane.state` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    _shortname = "state"

    def __init__(self, wires: Wires | None = None, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ):
        n_wires = n_wires or num_device_wires
        shape = (2**n_wires,)
        return shape, complex

    @property
    def numeric_type(self):
        return complex

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple[int]:
        num_wires = len(self.wires) if self.wires else num_device_wires
        return (2**num_wires,)

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # pylint:disable=redefined-outer-name
        def cast_to_complex(state):
            dtype = str(state.dtype)
            if "complex" in dtype:
                return state
            if (
                math.get_interface(state) == "tensorflow"
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                return math.cast(state, "complex128")
            floating_single = "float32" in dtype or "complex64" in dtype
            return math.cast(state, "complex64" if floating_single else "complex128")

        if not self.wires or wire_order == self.wires:
            return cast_to_complex(state)

        if not all(w in self.wires for w in wire_order):
            bad_wires = [w for w in wire_order if w not in self.wires]
            raise WireError(
                f"State wire order has wires {bad_wires} not present in "
                f"measurement with wires {self.wires}. StateMP.process_state cannot trace out wires."
            )

        shape = (2,) * len(wire_order)
        batch_size = None if math.ndim(state) == 1 else math.shape(state)[0]
        shape = (batch_size,) + shape if batch_size else shape
        state = math.reshape(state, shape)

        if wires_to_add := Wires(set(self.wires) - set(wire_order)):
            for _ in wires_to_add:
                state = math.stack([state, math.zeros_like(state)], axis=-1)
            wire_order = wire_order + wires_to_add

        desired_axes = [wire_order.index(w) for w in self.wires]
        if batch_size:
            desired_axes = [0] + [i + 1 for i in desired_axes]
        state = math.transpose(state, desired_axes)

        flat_shape = (2 ** len(self.wires),)
        if batch_size:
            flat_shape = (batch_size,) + flat_shape
        state = math.reshape(state, flat_shape)
        return cast_to_complex(state)

    def process_density_matrix(self, density_matrix: Sequence[complex], wire_order: Wires):
        # pylint:disable=redefined-outer-name
        raise ValueError("Processing from density matrix to state is not supported.")


class DensityMatrixMP(StateMP):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`density_matrix` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Wires, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: int | None = None,
        has_eigvals=False,
        shots: int | None = None,
        num_device_wires: int = 0,
    ):
        n_wires = n_wires or num_device_wires
        shape = (2**n_wires, 2**n_wires)
        return shape, complex

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple[int, int]:
        dim = 2 ** len(self.wires)
        return (dim, dim)

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # pylint:disable=redefined-outer-name
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        kwargs = {"indices": mapped_wires, "c_dtype": "complex128"}
        if not math.is_abstract(state) and math.any(math.iscomplex(state)):
            kwargs["c_dtype"] = state.dtype
        return math.reduce_statevector(state, **kwargs)

    def process_density_matrix(self, density_matrix: TensorLike, wire_order: Wires):
        # pylint:disable=redefined-outer-name
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        kwargs = {"indices": mapped_wires, "c_dtype": "complex128"}
        if not math.is_abstract(density_matrix) and math.any(math.iscomplex(density_matrix)):
            kwargs["c_dtype"] = density_matrix.dtype
        return math.reduce_dm(density_matrix, **kwargs)


def state() -> StateMP:
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


def density_matrix(wires) -> DensityMatrixMP:
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
