# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes and functions for midcircuit measurements with a
parameterized measurement axis."""

import uuid
from functools import lru_cache, partial
from typing import Hashable, Optional, Union

import numpy as np

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementValue, MidMeasureMP
from pennylane.wires import Wires


def measure_xy(
    angle, wires: Union[Hashable, Wires], reset: bool = False, postselect: Optional[int] = None
):
    """measure in the XY plane"""
    if qml.capture.enabled():
        primitive = _create_parametric_mid_measure_primitive()
        return primitive.bind(wires, angle=angle, plane="XY", reset=reset, postselect=postselect)

    return _measure_impl(wires, angle=angle, plane="XY", reset=reset, postselect=postselect)


@lru_cache
def _create_parametric_mid_measure_primitive():
    """Create a primitive corresponding to an parametrized mid-circuit measurement type.

    Called when using :func:`~pennylane.measure_xy`.

    Returns:
        jax.core.Primitive: A new jax primitive corresponding to a mid-circuit
        measurement.

    """
    # pylint: disable=import-outside-toplevel
    import jax

    from pennylane.capture.custom_primitives import NonInterpPrimitive

    mid_measure_p = NonInterpPrimitive("parametrized_measure")

    @mid_measure_p.def_impl
    def _(wires, angle=None, plane=None, reset=False, postselect=None):
        return _measure_impl(angle, wires, reset=reset, postselect=postselect, plane=plane)

    @mid_measure_p.def_abstract_eval
    def _(*_, **__):
        dtype = jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        return jax.core.ShapedArray((), dtype)

    return mid_measure_p


parametric_measure_prim = _create_parametric_mid_measure_primitive()


def _measure_impl(
    angle,
    wires: Union[Hashable, Wires],
    reset: Optional[bool] = False,
    postselect: Optional[int] = None,
    plane=None,
):
    """Concrete implementation of qml.measure"""
    wires = Wires(wires)
    if len(wires) > 1:
        raise qml.QuantumFunctionError(
            "Only a single qubit can be measured in the middle of the circuit"
        )

    # Create a UUID and a map between MP and MV to support serialization
    measurement_id = str(uuid.uuid4())[:8]
    mp = ParametricMidMeasureMP(
        wires=wires,
        angle=angle,
        plane=plane,
        reset=reset,
        postselect=postselect,
        id=measurement_id,
    )
    return MeasurementValue([mp], processing_fn=lambda v: v)


# ToDo: should some of the info be data instead of metadata?
# ToDo: does this need its own custom label? Or is it fine that it looks like a normal MCM in diagrams?
class ParametricMidMeasureMP(MidMeasureMP):
    """Parametric mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.

    Keyword Args:
        angle (float): The angle in radians
        plane (str): The plane the measurement basis lies in. Options are "XY", "XZ" and "YZ"
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (
            ("angle", self.angle),
            ("wires", self.raw_wires),
            ("plane", self.plane),
            ("reset", self.reset),
            ("id", self.id),
        )
        return (None, None), metadata

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wires: Optional[Wires],
        *,
        angle: Optional[float],
        plane: Optional[str] = "XY",
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), reset=reset, postselect=postselect, id=id)
        self.plane = plane
        self.angle = angle

    @classmethod
    def _primitive_bind_call(
        cls,
        wires=None,
        angle=None,
        plane=None,
        reset=False,
        postselect=None,
        id=None,
    ):
        wires = () if wires is None else wires
        return cls._wires_primitive.bind(
            *wires, angle=angle, plane=plane, reset=reset, postselect=postselect, id=id
        )

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

    @property
    def has_diagonalizing_gates(self):
        """Whether there are gates that need to be applied for to diagonalize the measurement"""
        return True

    @staticmethod
    def compute_diagonalizing_gates(wires, angle, plane):
        if plane == "XY":
            return [qml.QubitUnitary(_xy_to_z(angle), wires=wires)]
        if plane == "XZ":
            return [qml.RY(-angle, wires)]
        if plane == "YZ":
            return [qml.RX(-angle, wires)]

        raise NotImplementedError(
            f"{plane} plane not implemented. Available plans are 'XY' 'XZ' and 'YZ'."
        )

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        return self.compute_diagonalizing_gates(self.wires, self.angle, self.plane)

    # ToDo: is this needed anymore?
    @property
    def has_matrix(self):
        """The name of the measurement. Needed to match the Operator API."""
        return False


def _xy_to_z(angle):
    """Project XY basis states onto computational basis states"""
    return np.array([[1, np.exp(-1j * angle)], [1, -np.exp(-1j * angle)]]) / np.sqrt(2)


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@lru_cache
def _get_plxpr_diagonalize_mcms():  # pylint: disable=missing-docstring
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture.primitives import ctrl_transform_prim
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class DiagonalizeMCMInterpreter(qml.capture.PlxprInterpreter):
        """Plxpr Interpreter for applying the ``diagonalize_mcms`` transform to callables or jaxpr
        when program capture is enabled.
        """

        def __init__(self):
            super().__init__()

            # honestly not sure if I'm using this right, ask Mudit if it makes sense
            self.state = {"cur_idx": 0}

        def cleanup(self) -> None:
            """Perform any final steps after iterating through all equations.
            Blank by default, this method can clean up instance variables. Particularly,
            this method can be used to deallocate qubits and registers when converting to
            a Catalyst variant jaxpr.
            """
            self.state = {"cur_idx": 0}

    @DiagonalizeMCMInterpreter.register_primitive(parametric_measure_prim)
    def _(self, wires, angle=None, plane=None, reset=False, postselect=None):

        ParametricMidMeasureMP.compute_diagonalizing_gates(wires=wires, angle=angle, plane=plane)

        meas = type.__call__(
            ParametricMidMeasureMP,
            Wires(wires),
            angle=angle,
            plane=plane,
            reset=reset,
            postselect=postselect,
            id=self.state["cur_idx"],
        )

        # will this be specific to the mcm method? (I would think so)
        # Should mcm transform be applied first, and will that impact what needs to happen here?

        # if postselect is not None:
        #     qml.Projector(jax.numpy.array([postselect]), wires=wires)

        # if reset:
        #     if postselect is None:
        #         qml.CNOT(wires=cnot_wires[::-1])
        #     elif postselect == 1:
        #         qml.PauliX(wires=wires)

        self.state["cur_idx"] += 1
        return MeasurementValue([meas], lambda x: x)

    def diagonalize_mcms_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        """Function from decomposing jaxpr."""
        diagonalizer = DiagonalizeMCMInterpreter()

        def wrapper(*inner_args):
            return diagonalizer.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return DiagonalizeMCMInterpreter, diagonalize_mcms_plxpr_to_plxpr


DiagonalizeMCMInterpreter, diagonalize_mcms_plxpr_to_plxpr = _get_plxpr_diagonalize_mcms()


@partial(qml.transform, plxpr_transform=diagonalize_mcms_plxpr_to_plxpr)
def diagonalize_mcms(tape):
    """transform diagonalizing the parametrized MCMs"""

    new_operations = []

    for op in tape.operations:
        if isinstance(op, ParametricMidMeasureMP):
            new_operations.extend(op.diagonalizing_gates() + [op])
        elif isinstance(op, qml.ops.Conditional) and isinstance(op.base, ParametricMidMeasureMP):
            diag_gate = qml.ops.Conditional(
                expr=op.hyperparameters["meas_val"], then_op=op.diagonalizing_gates()[0]
            )
            new_operations.extend([diag_gate, op])
        else:
            new_operations.append(op)

    new_tape = tape.copy(operations=new_operations)

    return (new_tape,), null_postprocessing
