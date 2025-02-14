# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes and functions for creating and diagonalizing
mid-circuit measurements with a parameterized measurement axis."""

from typing import Optional

import pennylane as qml
from pennylane.drawer.tape_mpl import _add_operation_to_drawer
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP
from pennylane.wires import Wires


class ParametricMidMeasureMP(MidMeasureMP):
    """Parametric mid-circuit measurement. The basis for the measurement is parametrized by
    a plane ("XY", "YZ" or "ZX"), and an angle within the plane.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit are assumed.

    .. warning::
        Measurements should be diagonalized before execution for any device that only natively supports
        mid-circuit measurements in the computational basis. To diagonalize, the :func:`diagonalize_mcms <pennylane.ftqc.diagonalize_mcms>`
        transform can be applied.

        Skipping diagonalization for a circuit containing parametric mid-circuit measurements may result
        in a completed execution with incorrect results.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.

    Keyword Args:
        angle (float): The angle in radians
        plane (str): The plane the measurement basis lies in. Options are "XY", "ZX" and "YZ"
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    _shortname = "measure"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Optional[Wires],
        *,
        angle: Optional[float],
        plane: Optional[str],
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), reset=reset, postselect=postselect, id=id)
        self.plane = plane
        self.angle = angle

    def _flatten(self):
        metadata = (
            ("angle", self.angle),
            ("wires", self.raw_wires),
            ("plane", self.plane),
            ("reset", self.reset),
            ("id", self.id),
        )
        return (None, None), metadata

    # pylint: disable=arguments-renamed, arguments-differ
    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            self.plane,
            self.angle,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}_{self.plane.lower()}(wires={self.wires.tolist()}, angle={self.angle})"

    @property
    def has_diagonalizing_gates(self):
        """Whether there are gates that need to be applied to diagonalize the measurement"""
        return True

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        if self.plane == "XY":
            return [qml.PhaseShift(-self.angle, self.wires), qml.H(self.wires)]
        if self.plane == "ZX":
            return [qml.RY(-self.angle, self.wires)]
        if self.plane == "YZ":
            return [qml.RX(-self.angle, self.wires)]

        raise NotImplementedError(
            f"{self.plane} plane not implemented. Available plans are 'XY' 'ZX' and 'YZ'."
        )

    def label(self, decimals=None, base_label=None, cache=None):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Required to match general call signature. Not used.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing. Required to match general call signature. Not used.

        Returns:
            str: label to use in drawings
        """
        superscripts = {"X": "ˣ", "Y": "ʸ", "Z": "ᶻ"}
        _plane = "".join([superscripts[i] for i in self.plane])

        _label = f"┤↗{_plane}"

        if decimals is not None:
            _label += f"({self.angle:.{decimals}f})"

        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label


@_add_operation_to_drawer.register
def _(op: ParametricMidMeasureMP, drawer, layer, _):
    text = op.plane
    drawer.measure(layer, op.wires[0], text=text)  # assume one wire

    if op.reset:
        drawer.erase_wire(layer, op.wires[0], 1)
        drawer.box_gate(
            layer + 1,
            op.wires[0],
            "|0⟩",
            box_options={"zorder": 4},
            text_options={"zorder": 5},
        )


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@qml.transform
def diagonalize_mcms(tape):
    """Diagonalize any mid-circuit measurements in a parameterized basis into the computational basis.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to modify the mid-circuit measurements of.

    Returns:
        qnode (QNode) or tuple[List[QuantumScript], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Examples:**

    This transform allows us to transform mid-circuit measurements into the measurement basis by adding
    the relevant diagonalizing gates to the tape just before the measurement is performed.

    .. code-block:: python3

        from pennylane.ftqc import diagonalize_mcms, ParametricMidMeasureMP

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            ParametricMidMeasureMP(0, angle=x[1], plane="XY")
            return qml.expval(qml.Z(0))

    Applying the transform inserts the relevant gates before the measurement to allow
    measurements to be in the Z basis, so the original circuit

    >>> print(qml.draw(circuit, level=0)([np.pi/4, np.pi]))
    0: ──RY(0.79)──┤↗ˣʸ(3.14)├─┤  <Z>

    becomes

    >>> print(qml.draw(circuit)([np.pi/4, np.pi]))
    ──RY(0.79)──Rϕ(-3.14)──H──┤↗├─┤  <Z>
    """

    new_operations = []
    mps_mapping = {}

    for op in tape.operations:
        if isinstance(op, ParametricMidMeasureMP):

            # add diagonalizing gates to tape
            diag_gates = op.diagonalizing_gates()
            new_operations.extend(diag_gates)

            # add computational basis MCM to tape
            with qml.QueuingManager.stop_recording():
                new_mp = MidMeasureMP(op.wires, reset=op.reset, postselect=op.postselect, id=op.id)
            new_operations.append(new_mp)

            # track mapping from original to computational basis MCMs
            mps_mapping[op] = new_mp

        elif isinstance(op, qml.ops.Conditional):

            # from MCM mapping, map any MCMs in the condition if needed
            processing_fn = op.meas_val.processing_fn
            mps = [mps_mapping.get(op, op) for op in op.meas_val.measurements]
            expr = MeasurementValue(mps, processing_fn=processing_fn)

            if isinstance(op.base, ParametricMidMeasureMP):
                # add conditional diagonalizing gates + conditional MCM to the tape
                with qml.QueuingManager.stop_recording():
                    diag_gates = [
                        qml.ops.Conditional(expr=expr, then_op=gate)
                        for gate in op.diagonalizing_gates()
                    ]
                    new_mp = MidMeasureMP(
                        op.wires, reset=op.base.reset, postselect=op.base.postselect, id=op.base.id
                    )
                    new_cond = qml.ops.Conditional(expr=expr, then_op=new_mp)

                new_operations.extend(diag_gates)
                new_operations.append(new_cond)

                # track mapping from original to computational basis MCMs
                mps_mapping[op.base] = new_mp
            else:
                with qml.QueuingManager.stop_recording():
                    new_cond = qml.ops.Conditional(expr=expr, then_op=op.base)
                new_operations.append(new_cond)

        else:
            new_operations.append(op)

    new_tape = tape.copy(operations=new_operations)

    return (new_tape,), null_postprocessing
