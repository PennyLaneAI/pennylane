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
from typing import Hashable, Optional, Union

import numpy as np

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementProcess, MeasurementValue
from pennylane.wires import Wires


def measure_xy(
    angle, wires: Union[Hashable, Wires], reset: bool = False, postselect: Optional[int] = None
):
    """measure in the XY plane"""
    if qml.capture.enabled():
        raise NotImplementedError
        # primitive = _create_mid_measure_primitive()
        # return primitive.bind(wires, reset=reset, postselect=postselect)

    return _measure_impl(angle, wires, reset=reset, postselect=postselect)


def _measure_impl(
    angle,
    wires: Union[Hashable, Wires],
    reset: Optional[bool] = False,
    postselect: Optional[int] = None,
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
        angle, wires=wires, reset=reset, postselect=postselect, id=measurement_id, plane="XY"
    )
    return MeasurementValue([mp], processing_fn=lambda v: v)


# ToDo: should this really be a MeasurementProcess, or is it just an operation?
#  Does it matter in practice? Conceptually?

# ToDo: generalize to other planes. Should plane options also include "ZX", "ZY" and "YX"?

# ToDo: should some of the info be data instead of metadata?


# ToDo: program capture compatibility (_create_midmeasure_primitive equivalent)
class ParametricMidMeasureMP(MeasurementProcess):
    """Parametric mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`pennylane.measure` for detailed documentation.

    Args:
        angle (float): The angle in radians
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.

    Keyword Args:
        plane (str): The plane the measurement basis lies in. Options are "XY", "XZ" and "YZ"
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (
            ("plane", self.plane),
            ("angle", self.angle),
            ("wires", self.raw_wires),
            ("reset", self.reset),
            ("id", self.id),
        )
        return (None, None), metadata

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        angle: Optional[float],
        wires: Optional[Wires],
        *,
        plane: Optional[str] = "XY",
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), id=id)
        self.plane = plane
        self.angle = angle
        self.reset = reset
        self.postselect = postselect

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

    def decomposition(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        if self.plane == "XY":
            U = qml.QubitUnitary(_xy_to_z(self.angle), wires=self.wires)
        else:
            raise NotImplementedError(f"{self.plane} plane not implemented")

        return [U, qml.measurements.MidMeasureMP(self.wires, self.reset, self.postselect, self.id)]

    @property
    def _queue_category(self):
        return "_ops"

    @property
    def data(self):
        """The data of the measurement. Needed to match the Operator API."""
        return []

    @property
    def name(self):
        """The name of the measurement. Needed to match the Operator API."""
        return self.__class__.__name__

    # ToDo: _primitive_bind_call
    #
    # ToDo: _abstract_eval
    #
    # ToDo: label


def _xy_to_z(angle):
    """Project XY basis states onto computational basis states"""
    return np.array([[1, np.exp(-1j * angle)], [1, -np.exp(-1j * angle)]]) / np.sqrt(2)
