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

import hashlib
import uuid
from collections.abc import Hashable, Iterable
from copy import copy
from functools import lru_cache

import numpy as np

from pennylane import capture
from pennylane.drawer.tape_mpl import _add_operation_to_drawer
from pennylane.exceptions import QuantumFunctionError
from pennylane.math import is_abstract, isscalar, ndim, unwrap
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP, measure
from pennylane.ops.op_math import Conditional, adjoint
from pennylane.ops.qubit import RX, RY, H, PhaseShift, S
from pennylane.queuing import QueuingManager
from pennylane.transforms import transform
from pennylane.wires import Wires


@lru_cache(maxsize=1)
def _create_parametrized_mid_measure_primitive():
    """Create a primitive corresponding to a parametrized mid-circuit measurement type.

    Called when using a parametrized mid-circuit measurement, such as
    :func:`~pennylane.measure_arbitrary_basis`.

    Returns:
        jax.core.Primitive: A new jax primitive corresponding to a mid-circuit
        measurement.

    """
    # pylint: disable=import-outside-toplevel
    import jax

    from pennylane.capture.custom_primitives import QmlPrimitive

    measure_in_basis_p = QmlPrimitive("measure_in_basis")

    @measure_in_basis_p.def_impl
    def _(wires, angle=0.0, plane="ZX", reset=False, postselect=None):
        return _measure_impl(
            wires,
            measurement_class=ParametricMidMeasureMP,
            angle=angle,
            plane=plane,
            reset=reset,
            postselect=postselect,
        )

    @measure_in_basis_p.def_abstract_eval
    def _(*_, **__):
        return jax.core.ShapedArray((), jax.numpy.bool)

    return measure_in_basis_p


def measure_arbitrary_basis(
    wires: Hashable | Wires,
    angle: float,
    plane: str,
    reset: bool = False,
    postselect: int | None = None,
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

    .. code-block:: python

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
    if len(Wires(wires)) > 1:
        raise QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    if capture.enabled():
        primitive = _create_parametrized_mid_measure_primitive()
        return primitive.bind(angle, wires, plane=plane, reset=reset, postselect=postselect)

    return _measure_impl(
        wires, ParametricMidMeasureMP, angle=angle, plane=plane, reset=reset, postselect=postselect
    )


def measure_x(
    wires: Hashable | Wires,
    reset: bool = False,
    postselect: int | None = None,
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
    if len(Wires(wires)) > 1:
        raise QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    if capture.enabled():
        primitive = _create_parametrized_mid_measure_primitive()
        return primitive.bind(0.0, wires, plane="XY", reset=reset, postselect=postselect)

    return _measure_impl(wires, XMidMeasureMP, reset=reset, postselect=postselect)


def measure_y(
    wires: Hashable | Wires,
    reset: bool = False,
    postselect: int | None = None,
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
    if len(Wires(wires)) > 1:
        raise QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    if capture.enabled():
        primitive = _create_parametrized_mid_measure_primitive()
        return primitive.bind(np.pi / 2, wires, plane="XY", reset=reset, postselect=postselect)

    return _measure_impl(wires, YMidMeasureMP, reset=reset, postselect=postselect)


def measure_z(
    wires: Hashable | Wires,
    reset: bool = False,
    postselect: int | None = None,
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
    # capture is already handled inside qml.measure
    return measure(wires, reset=reset, postselect=postselect)


def _measure_impl(
    wires: Hashable | Wires,
    measurement_class=MidMeasureMP,
    **kwargs,
):
    """Concrete implementation of qml.measure"""
    wires = Wires(wires)

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())
    mp = measurement_class(wires=wires, id=measurement_id, **kwargs)
    return MeasurementValue([mp])


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
        wires: Wires | None,
        *,
        angle: float | None,
        plane: str | None,
        reset: bool | None = False,
        postselect: int | None = None,
        id: str | None = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), reset=reset, postselect=postselect, id=id)
        self.hyperparameters["plane"] = plane
        self.hyperparameters["angle"] = angle

    @property
    def plane(self) -> str | None:
        """The plane the measurement basis lies in. Options are "XY", "ZX" and "YZ"""
        return self.hyperparameters["plane"]

    @property
    def angle(self):
        """The angle in radians"""
        return self.hyperparameters["angle"]

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        if is_abstract(self.angle):  # pragma: no cover
            # no unique value from tracer to values, hash based on object string
            param_hash = hashlib.sha256(str(self).encode()).digest()
        elif isscalar(self.angle) or ndim(self.angle) == 0:
            # Values are 0-dim arrays or scalars, array-ify
            param_hash = hashlib.sha256(unwrap(self.angle)).digest()
        else:
            # otherwise, use the existing array structure
            param_hash = hashlib.sha256(self.angle).digest()

        fingerprint = (
            self.__class__.__name__,
            self.plane,
            param_hash,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)

    # pylint: disable=too-many-positional-arguments, arguments-differ
    @classmethod
    def _primitive_bind_call(
        cls, angle=0.0, wires=None, plane="ZX", reset=False, postselect=None, id=None
    ):
        wires = () if wires is None else wires
        return cls._primitive.bind(
            *wires, angle=angle, plane=plane, reset=reset, postselect=postselect, id=id
        )

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}_{self.plane.lower()}(wires={self.wires.tolist()}, angle={self.angle})"

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        if self.plane == "XY":
            return [PhaseShift(-self.angle, self.wires), H(self.wires)]
        if self.plane == "ZX":
            return [RY(-self.angle, self.wires)]
        if self.plane == "YZ":
            return [RX(-self.angle, self.wires)]

        raise NotImplementedError(
            f"{self.plane} plane not implemented. Available plans are 'XY' 'ZX' and 'YZ'."
        )

    def label(self, decimals: int = None, base_label: Iterable[str] = None, cache: dict = None):
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

    def _flatten(self):
        metadata = (("reset", self.reset), ("postselect", self.postselect), ("id", self.id))
        return (), (self.wires, metadata)

    def __init__(
        self,
        wires: Wires | None,
        reset: bool | None = False,
        postselect: int | None = None,
        id: str | None = None,
    ):
        super().__init__(
            wires=Wires(wires), angle=0, plane="XY", reset=reset, postselect=postselect, id=id
        )

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}(wires={self.wires.tolist()})"

    def label(self, decimals: int = None, base_label: Iterable[str] = None, cache: dict = None):
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
        return [H(self.wires)]


class YMidMeasureMP(ParametricMidMeasureMP):
    """A subclass of ParametricMidMeasureMP that uses the Y measurement basis
    (angle=pi/2, plane="XY"). For labels and visualizations, this will be represented
    as a Y measurement. It is otherwise identical to the parent class."""

    _shortname = "measure_y"

    def _flatten(self):
        metadata = (("reset", self.reset), ("postselect", self.postselect), ("id", self.id))
        return (), (self.wires, metadata)

    def __init__(
        self,
        wires: Wires | None,
        reset: bool | None = False,
        postselect: int | None = None,
        id: str | None = None,
    ):
        super().__init__(
            wires=Wires(wires),
            angle=np.pi / 2,
            plane="XY",
            reset=reset,
            postselect=postselect,
            id=id,
        )

    def __repr__(self):
        """Representation of this class."""
        return f"{self._shortname}(wires={self.wires.tolist()})"

    def label(self, decimals: int = None, base_label: str = None, cache: dict = None):
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
        return [adjoint(S(self.wires)), H(self.wires)]


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


@transform
def diagonalize_mcms(tape):
    """Diagonalize any mid-circuit measurements in a parameterized basis into the computational basis.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to modify the mid-circuit measurements of.

    Returns:
        qnode (QNode) or tuple[List[QuantumScript], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Examples:**

    This transform allows us to transform mid-circuit measurements into the measurement basis by adding
    the relevant diagonalizing gates to the tape just before the measurement is performed.

    .. code-block:: python

        from functools import partial

        from pennylane.ftqc import ParametricMidMeasureMP, diagonalize_mcms

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @partial(qml.set_shots, shots=1000)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RX(x, wires=0)
            m = measure_y(0)
            qml.cond(m, qml.X)(1)
            return qml.expval(qml.Z(1))

    Applying the transform inserts the relevant gates before the measurement to allow
    measurements to be in the Z basis, so the original circuit

    >>> print(qml.draw(circuit, level=0)(np.pi/4))
    0: ──RX(0.79)──┤↗ʸ├────┤
    1: ─────────────║────X─┤  <Z>
                    ╚════╝

    becomes

    >>> print(qml.draw(circuit)(np.pi/4))
    0: ──RX(0.79)──S†──H──┤↗├────┤
    1: ────────────────────║───X─┤  <Z>
                           ╚═══╝


    .. details::
        :title: Conditional measurements

        The transform can also handle diagonalization of conditional measurements created by
        :func:`qml.ftqc.cond_measure <pennylane.ftqc.cond_measure>`. This is done by replacing the
        measurements for the true and false condition with conditional diagonalizing gates,
        and a single measurement in the computational basis:

        .. code-block:: python

            from pennylane.ftqc import cond_measure, diagonalize_mcms, measure_x

            dev = qml.device("default.qubit")

            @diagonalize_mcms
            @qml.qnode(dev)
            def circuit(x):
                qml.RY(x[0], wires=0)
                qml.RX(x[1], wires=1)
                m = qml.measure(0)
                m2 = cond_measure(m, measure_x, measure_y)(1)
                qml.cond(m2, qml.X)(1)
                return qml.expval(qml.Z(1))

        The :func:`cond_measure <pennylane.ftqc.cond_measure>` function adds a conditional X-basis
        measurement and a conditional Y basis measurement to the circuit, with opposite conditions.
        When the transform is applied, the diagonalizing gates of the measurements are conditional.
        The two conditional measurements then become equivalent measurements in the computational basis
        with opposite conditions, and can be simplified to a single, unconditional measurement in the
        computational basis.

        This circuit thus diagonalizes to:

        >>> print(qml.draw(circuit)([np.pi, np.pi/4]))
        0: ──RY(3.14)──┤↗├───────────────────┤
        1: ──RX(0.79)───║───H──S†──H──┤↗├──X─┤  <Z>
                        ╚═══╩══╩═══╝   ╚═══╝

        where the initial Hadamard gate on wire 1 has the same condition as the original X-basis
        measurement, and the adjoint S gate and second Hadamard share a condition with the Y-basis
        measurement.
    """

    new_operations = []
    mps_mapping = {}

    curr_idx = 0

    for i, op in enumerate(tape.operations):

        if i != curr_idx:
            continue

        if isinstance(op, ParametricMidMeasureMP):

            # add diagonalizing gates to tape
            diag_gates = op.diagonalizing_gates()
            new_operations.extend(diag_gates)

            # add computational basis MCM to tape
            with QueuingManager.stop_recording():
                new_mp = MidMeasureMP(op.wires, reset=op.reset, postselect=op.postselect, id=op.id)
            new_operations.append(new_mp)

            # track mapping from original to computational basis MCMs
            mps_mapping[op] = new_mp

        elif isinstance(op, Conditional):

            # from MCM mapping, map any MCMs in the condition if needed
            mps = [mps_mapping.get(op, op) for op in op.meas_val.measurements]

            if isinstance(op.base, MidMeasureMP):
                # the only user-facing API for creating Conditionals with MCMs is meas_cond,
                # which ensures both and true_fn and false_fn are included, so here we assume the
                # expected format (i.e. conditional mcms are found pairwise with opposite conditions)
                true_cond, false_cond = (op, tape.operations[i + 1])
                # we process both the true_cond and the false_cond together, so we skip an index in the ops
                curr_idx += 1

                # add conditional diagonalizing gates + computational basis MCM to the tape
                p_fn = (
                    true_cond.meas_val.processing_fn if true_cond.meas_val.has_processing else None
                )
                expr_true = MeasurementValue(mps, processing_fn=p_fn)
                f_fn = (
                    false_cond.meas_val.processing_fn
                    if false_cond.meas_val.has_processing
                    else None
                )
                expr_false = MeasurementValue(mps, processing_fn=f_fn)

                with QueuingManager.stop_recording():
                    diag_gates_true = [
                        Conditional(expr=expr_true, then_op=gate)
                        for gate in true_cond.diagonalizing_gates()
                    ]

                    diag_gates_false = [
                        Conditional(expr=expr_false, then_op=gate)
                        for gate in false_cond.diagonalizing_gates()
                    ]

                    new_mp = MidMeasureMP(
                        op.wires, reset=op.base.reset, postselect=op.base.postselect, id=op.base.id
                    )

                new_operations.extend(diag_gates_true)
                new_operations.extend(diag_gates_false)
                new_operations.append(new_mp)

                # track mapping from original to computational basis MCMs
                mps_mapping[true_cond.base] = new_mp
                mps_mapping[false_cond.base] = new_mp
            else:
                processing_fn = op.meas_val.processing_fn
                expr = MeasurementValue(mps, processing_fn=processing_fn)

                with QueuingManager.stop_recording():
                    new_cond = Conditional(expr=expr, then_op=op.base)
                new_operations.append(new_cond)

        else:
            new_operations.append(op)

        curr_idx += 1

    new_measurements = []
    for mp in tape.measurements:
        if mp.mv is None:
            new_measurements.append(mp)
        else:
            new_mp = copy(mp)
            mps = [mps_mapping.get(m, m) for m in mp.mv.measurements]
            new_mp.mv.measurements = mps
            new_measurements.append(new_mp)

    new_tape = tape.copy(operations=new_operations, measurements=new_measurements)
    return (new_tape,), null_postprocessing
