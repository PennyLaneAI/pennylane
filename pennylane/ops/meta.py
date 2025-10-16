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
This submodule contains the discrete-variable quantum operations that do
not depend on any parameters.
"""
from collections.abc import Hashable, Sequence

# pylint: disable=arguments-differ
from copy import copy
from typing import Literal

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires, WiresLike


class Barrier(Operation):
    r"""Barrier(wires)
    The Barrier operator, used to separate the compilation process into blocks or as a visual tool.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        only_visual (bool): True if we do not want it to have an impact on the compilation process. Default is False.
        wires (Sequence[int] or int): the wires the operation acts on
    """

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def __init__(self, wires: WiresLike = (), only_visual=False, id=None):
        wires = Wires(wires)
        self.only_visual = only_visual
        self.hyperparameters["only_visual"] = only_visual
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires, only_visual=False):  # pylint: disable=unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Barrier.decomposition`.

        ``Barrier`` decomposes into an empty list for all arguments.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            only_visual (Bool): True if we do not want it to have an impact on the compilation process. Default is False.

        Returns:
            list: decomposition of the operator

        **Example:**

        >>> print(qml.Barrier.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "||"

    def _controlled(self, _):
        return copy(self).queue()

    def adjoint(self):
        return copy(self)

    def pow(self, z):
        return [copy(self)]

    def simplify(self):
        if self.only_visual:
            if len(self.wires) == 1:
                return qml.Identity(self.wires[0])
            return qml.prod(*(qml.Identity(w) for w in self.wires))
        return self


class WireCut(Operation):
    r"""WireCut(wires)
    The wire cut operation, used to manually mark locations for wire cuts.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wires the operation acts on
    """

    num_params = 0
    grad_method = None

    def __init__(self, wires: WiresLike = (), id=None):
        wires = Wires(wires)
        super().__init__(wires=wires, id=id)
        if not self._wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. At least one wire has to be provided."
            )

    @staticmethod
    def compute_decomposition(wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method).

        Since this operator is a placeholder inside a circuit, it decomposes into an empty list.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.WireCut.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "//"

    def adjoint(self):
        return WireCut(wires=self.wires)

    def pow(self, z):
        return [copy(self)]


class Snapshot(Operation):
    r"""
    The Snapshot operation saves the internal execution state of the quantum function
    at a specific point in the execution pipeline. As such, it is a pseudo operation
    with no effect on the quantum state. Arbitrary measurements are supported
    in snapshots via the keyword argument ``measurement``.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        tag (str or None): An optional custom tag for the snapshot, used to index it
            in the snapshots dictionary.

        measurement (MeasurementProcess or None): An optional argument to record arbitrary
            measurements during execution. If None, the measurement defaults to `qml.state`
            on the available wires.

        shots (Literal["workflow"], None, int, Sequence[int]): shots to use for the snapshot.
            ``"workflow"`` indicates the same number of shots as for the final measurement.

    .. warning::

        ``Snapshot`` captures the internal execution state at a point in the circuit, but compilation transforms
        (e.g., ``combine_global_phases``, ``merge_rotations``) may reorder or modify operations across the snapshot.
        As a result, the captured state may differ from the original intent.

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", seed=42)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            m = qml.Snapshot("samples", qml.sample(), shots=5)
            return qml.expval(qml.X(0))

    >>> from pprint import pprint
    >>> pprint(qml.snapshots(circuit)())
    {0: np.float64(1.0),
     2: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
     'execution_results': np.float64(0.0),
     'samples': array([[1, 1],
                       [0, 0],
                       [1, 1],
                       [1, 1],
                       [0, 0]]),
     'very_important_state': array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j])}

    .. seealso:: :func:`~.snapshots`
    """

    num_params = 0
    grad_method = None

    @classmethod
    def _primitive_bind_call(cls, tag=None, measurement=None, shots="workflow"):
        if measurement is None:
            return cls._primitive.bind(measurement=measurement, tag=tag, shots=shots)
        return cls._primitive.bind(measurement, tag=tag, shots=shots)

    def __init__(
        self,
        tag: str | None = None,
        measurement=None,
        shots: Literal["workflow"] | None | int | Sequence[int] = "workflow",
    ):
        if tag is not None and not isinstance(tag, (str, int)):
            # ints are validated in snapshot transform, as the snapshot
            # transform adds int tags
            raise ValueError("Snapshot tags can only be of type 'str'")

        if measurement is None:
            measurement = qml.state()
        if isinstance(measurement, qml.measurements.StateMP) and shots == "workflow":
            shots = None  # always use analytic with state
        if isinstance(measurement, qml.measurements.MidMeasureMP):
            raise ValueError("Mid-circuit measurements can not be used in snapshots.")
        if isinstance(measurement, qml.measurements.MeasurementProcess):
            qml.queuing.QueuingManager.remove(measurement)
        else:
            raise ValueError(
                f"The measurement {measurement.__class__.__name__} is not supported as it is not "
                f"an instance of {qml.measurements.MeasurementProcess}"
            )

        self.hyperparameters["tag"] = tag
        self.hyperparameters["measurement"] = measurement
        self.hyperparameters["shots"] = (
            shots if shots == "workflow" else qml.measurements.Shots(shots)
        )
        super().__init__(wires=measurement.wires)

    def __repr__(self):
        return f"<Snapshot: tag={self.tag}, measurement={self.hyperparameters['measurement']}, shots={self.hyperparameters['shots']}>"

    @property
    def tag(self) -> None | str | int:
        """The tag for the snapshot."""
        return self.hyperparameters["tag"]

    def update_tag(self, new_tag: int | None | str):
        """Create a new snapshot with an updated tag."""
        new_op = copy(self)
        new_op.hyperparameters["tag"] = new_tag
        return new_op

    def label(self, decimals=None, base_label=None, cache=None):
        return "|Snap|"

    def _flatten(self):
        return (self.hyperparameters["measurement"],), (self.tag, self.hyperparameters["shots"])

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(tag=metadata[0], measurement=data[0], shots=metadata[1])

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return []

    def _controlled(self, _):
        return Snapshot(**self.hyperparameters)

    def adjoint(self):
        return Snapshot(**self.hyperparameters)

    def map_wires(self, wire_map: dict[Hashable, Hashable]) -> "Snapshot":
        new_measurement = self.hyperparameters["measurement"].map_wires(wire_map)
        return Snapshot(
            tag=self.tag, measurement=new_measurement, shots=self.hyperparameters["shots"]
        )


# Since measurements are captured as variables in plxpr with the capture module,
# the measurement is treated as a traceable argument.
# This step is mandatory for fixing the order of arguments overwritten by ``Snapshot._primitive_bind_call``.
if Snapshot._primitive:  # pylint: disable=protected-access

    @Snapshot._primitive.def_impl  # pylint: disable=protected-access
    def _(measurement, tag=None, shots="workflow"):
        return type.__call__(Snapshot, tag=tag, measurement=measurement, shots=shots)
