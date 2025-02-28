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

import uuid
from collections.abc import Hashable
from typing import Iterable, Optional, Union

import numpy as np

import pennylane as qml
from pennylane.drawer.tape_mpl import _add_operation_to_drawer
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP, measure
from pennylane.wires import Wires


def measure_arbitrary_basis(
    wires: Union[Hashable, Wires],
    angle: float,
    plane: str,
    reset: bool = False,
    postselect: Optional[int] = None,
):
    r"""Perform a mid-circuit measurement in the basis defined by the plane and angle on the
    supplied qubit.

    The measurements are performed using the 0, 1 convention rather than the ±1 convention.

    If a device doesn't support mid-circuit measurements natively, then the desired ``mcm_method`` for
    executing mid-circuit measurements should be passed to the QNode.

    .. warning::
        Measurements should be diagonalized before execution for any device that only natively supports
        mid-circuit measurements in the computational basis. To diagonalize, the :func:`diagonalize_mcms <pennylane.ftqc.diagonalize_mcms>`
        transform can be applied.

        Skipping diagonalization for a circuit containing parametric mid-circuit measurements may result
        in a completed execution with incorrect results.

    Args:
        wires (Wires): The wire to measure.
        angle (float): The angle of rotation defining the axis, specified in radians.
        plane (str): The plane the measurement basis lies in. Options are "XY", "YZ" and "ZX"
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0 \rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.

    Returns:
        MeasurementValue: The mid-circuit measurement result linked to the created ``MidMeasureMP``.

    Raises:
        QuantumFunctionError: if multiple wires were specified

    .. note::
        Reset behaviour will depend on the execution method for mid-circuit measurements,
        and may not work for all configurations.

    **Example:**

    .. code-block:: python3

        import pennylane as qml
        from pennylane.ftqc import diagonalize_mcms, measure_arbitrary_basis

        dev = qml.device("default.qubit", wires=3)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = measure_arbitrary_basis(1, angle=np.pi/3, plane="XY")

            qml.cond(m_0, qml.RY)(y, wires=0)
            return qml.probs(wires=[0])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246])
    >>> func(*pars)
    array([0.91237915, 0.08762085])

    .. details::
        :title: Plane and angle

        The plane and angle are related to the axis of measurement by the following formulas:

        .. math:: M_{XY}(\phi) =\frac{1}{\sqrt{2}} (|0\rangle + e^{i\phi} |1\rangle),

        .. math:: M_{YZ}(\theta) =\cos{\frac{\theta}{2}}|0\rangle + i \sin{\frac{\theta}{2}} |1\rangle,\text{ and}

        .. math:: M_{ZX}(\theta) = \cos{\frac{\theta}{2}}|0\rangle + \sin{\frac{\theta}{2}} |1\rangle

        where, in terms of `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_ in
        the physics convention, the angles :math:`\phi` and :math:`\theta` are the azimuthal and polar angles,
        respectively.

    .. details::
        :title: Using mid-circuit measurements

        Measurement outcomes can be used to conditionally apply operations, and measurement
        statistics can be gathered and returned by a quantum function. Measurement outcomes can
        also be manipulated using arithmetic operators like ``+``, ``-``, ``*``, ``/``, etc. with
        other mid-circuit measurements or scalars.

        See the :func:`qml.measure <pennylane.measurements.measure>` function
        for details on the available arithmetic operators for mid-circuit measurement results.

        Mid-circuit measurement results can be processed with the usual measurement functions such as
        :func:`~.expval`. For QNodes with finite shots, :func:`~.sample` applied to a mid-circuit measurement
        result will return a binary sequence of samples.
        See :ref:`here <mid_circuit_measurements_statistics>` for more details.
    """

    # ToDo: if capture is enabled, create and bind primitive here and return primitive instead (subsequent PR)

    return _measure_impl(
        wires, ParametricMidMeasureMP, angle=angle, plane=plane, reset=reset, postselect=postselect
    )


def measure_x(
    wires: Union[Hashable, Wires],
    reset: bool = False,
    postselect: Optional[int] = None,
):
    r"""Perform a mid-circuit measurement in the X basis. The measurements are performed using the 0, 1
    convention rather than the ±1 convention.

    For more details on the results of mid-circuit measurements and how to use them,
    see :func:`qml.measure <pennylane.measure>`.

    For more details on mid-circuit measurements in an arbitrary basis (besides the computational basis),
    see :func:`measure_arbitrary_basis <pennylane.ftqc.measure_arbitrary_basis>`.

    .. warning::
        Measurements should be diagonalized before execution for any device that only natively supports
        mid-circuit measurements in the computational basis. To diagonalize, the :func:`diagonalize_mcms <pennylane.ftqc.diagonalize_mcms>`
        transform can be applied.

        Skipping diagonalization for a circuit containing parametric mid-circuit measurements may result
        in a completed execution with incorrect results.

    Args:
        wires (Wires): The wire to measure.
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0 \rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.

    Returns:
        MeasurementValue: The mid-circuit measurement result linked to the created ``MidMeasureMP``.

    Raises:
        QuantumFunctionError: if multiple wires were specified

    """

    # ToDo: if capture is enabled, create and bind primitive here and return primitive instead (subsequent PR)

    return _measure_impl(wires, XMidMeasureMP, reset=reset, postselect=postselect)


def measure_y(
    wires: Union[Hashable, Wires],
    reset: bool = False,
    postselect: Optional[int] = None,
):
    r"""Perform a mid-circuit measurement in the Y basis. The measurements are performed using the 0, 1
    convention rather than the ±1 convention.

    For more details on the results of mid-circuit measurements and how to use them,
    see :func:`qml.measure <pennylane.measure>`.

    For more details on mid-circuit measurements in an arbitrary basis (besides the computational basis),
    see :func:`measure_arbitrary_basis <pennylane.ftqc.measure_arbitrary_basis>`.

    .. warning::
        Measurements should be diagonalized before execution for any device that only natively supports
        mid-circuit measurements in the computational basis. To diagonalize, the :func:`diagonalize_mcms <pennylane.ftqc.diagonalize_mcms>`
        transform can be applied.

        Skipping diagonalization for a circuit containing parametric mid-circuit measurements may result
        in a completed execution with incorrect results.

    Args:
        wires (Wires): The wire to measure.
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0 \rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.

    Returns:
        MeasurementValue: The mid-circuit measurement result linked to the created ``MidMeasureMP``.

    Raises:
        QuantumFunctionError: if multiple wires were specified

    """

    # ToDo: if capture is enabled, create and bind primitive here and return primitive instead (subsequent PR)

    return _measure_impl(wires, YMidMeasureMP, reset=reset, postselect=postselect)


def measure_z(
    wires: Union[Hashable, Wires],
    reset: bool = False,
    postselect: Optional[int] = None,
):
    r"""Perform a mid-circuit measurement in the Z basis. The measurements are performed using the 0, 1
    convention rather than the ±1 convention.

    .. note::
        This function dispatches to :func:`qml.measure <pennylane.measure>`

    For more details on the results of mid-circuit measurements and how to use them,
    see :func:`qml.measure <pennylane.measure>`.

    Args:
        wires (Wires): The wire to measure.
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0 \rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.

    Returns:
        MeasurementValue: The mid-circuit measurement result linked to the created ``MidMeasureMP``.

    Raises:
        QuantumFunctionError: if multiple wires were specified

    """
    return measure(wires, reset=reset, postselect=postselect)


def _measure_impl(
    wires: Union[Hashable, Wires],
    measurement_class=MidMeasureMP,
    **kwargs,
):
    """Concrete implementation of qml.measure"""
    wires = Wires(wires)
    if len(wires) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())
    mp = measurement_class(wires=wires, id=measurement_id, **kwargs)
    return MeasurementValue([mp], processing_fn=lambda v: v)


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

    def label(
        self, decimals: int = None, base_label: Iterable[str] = None, cache: dict = None
    ):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals: If ``None``, no parameters are included. Else,
                how to round the parameters. Defaults to None.
            base_label: overwrite the non-parameter component of the label.
                Required to match general call signature. Not used.
            cache: dictionary that carries information between label calls in the
                same drawing. Required to match general call signature. Not used.

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


class XMidMeasureMP(ParametricMidMeasureMP):
    """A subclass of ParametricMidMeasureMP that uses the X measurement basis
    (angle=0, plane="XY"). For labels and visualizations, this will be represented
    as a X measurement. It is otherwise identical to the parent class."""

    _shortname = "measure_x"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Optional[Wires],
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            wires=Wires(wires), angle=0, plane="XY", reset=reset, postselect=postselect, id=id
        )

    def _flatten(self):
        metadata = (
            ("wires", self.raw_wires),
            ("reset", self.reset),
            ("id", self.id),
        )
        return (None, None), metadata

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}(wires={self.wires.tolist()})"

    def label(
        self, decimals: int = None, base_label: Iterable[str] = None, cache: dict = None
    ):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals: If ``None``, no parameters are included. Else, how to round
                the parameters. Required to match general call signature. Not used.
            base_label: overwrite the non-parameter component of the label.
                Required to match general call signature. Not used.
            cache: dictionary that carries information between label calls in the
                same drawing. Required to match general call signature. Not used.

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗ˣ"

        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        return [qml.H(self.wires)]


class YMidMeasureMP(ParametricMidMeasureMP):
    """A subclass of ParametricMidMeasureMP that uses the Y measurement basis
    (angle=pi/2, plane="XY"). For labels and visualizations, this will be represented
    as a Y measurement. It is otherwise identical to the parent class."""

    _shortname = "measure_y"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Optional[Wires],
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            wires=Wires(wires),
            angle=np.pi / 2,
            plane="XY",
            reset=reset,
            postselect=postselect,
            id=id,
        )

    def _flatten(self):
        metadata = (
            ("wires", self.raw_wires),
            ("reset", self.reset),
            ("id", self.id),
        )
        return (None, None), metadata

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}(wires={self.wires.tolist()})"

    def label(
        self, decimals: int = None, base_label: str = None, cache: dict = None
    ):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals: If ``None``, no parameters are included. Else, how to round
                the parameters. Required to match general call signature. Not used.
            base_label: overwrite the non-parameter component of the label.
                Required to match general call signature. Not used.
            cache: dictionary that carries information between label calls in the
                same drawing. Required to match general call signature. Not used.

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗ʸ"

        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        # alternatively we could apply (Z, S) instead of adjoint(S)
        return [qml.adjoint(qml.S(self.wires)), qml.H(self.wires)]


@_add_operation_to_drawer.register
def _(op: ParametricMidMeasureMP, drawer, layer, _):
    if isinstance(op, XMidMeasureMP):
        text = "X"
    elif isinstance(op, YMidMeasureMP):
        text = "Y"
    else:
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
