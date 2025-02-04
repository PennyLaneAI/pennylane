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

from typing import Optional

import numpy as np

import pennylane as qml
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.wires import Wires


# ToDo: should some of the info be data instead of metadata?
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

    @property
    def has_diagonalizing_gates(self):
        """Whether there are gates that need to be applied for to diagonalize the measurement"""
        return True

    def diagonalizing_gates(self):
        """Decompose to a diagonalizing gate and a standard MCM in the computational basis"""
        if self.plane == "XY":
            return [qml.QubitUnitary(_xy_to_z(self.angle), wires=self.wires)]
        if self.plane == "XZ":
            return [qml.RY(-self.angle, self.wires)]
        if self.plane == "YZ":
            return [qml.RX(-self.angle, self.wires)]

        raise NotImplementedError(
            f"{self.plane} plane not implemented. Available plans are 'XY' 'XZ' and 'YZ'."
        )

    # ToDo: is this needed anymore?
    @property
    def has_matrix(self):
        """The name of the measurement. Needed to match the Operator API."""
        return False

    def label(self, decimals=None, base_label=None, cache=None):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗ᶿ"
        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        if decimals is not None:

            def _format(x):
                try:
                    return format(qml.math.toarray(x), f".{decimals}f")
                except ValueError:
                    # If the parameter can't be displayed as a float
                    return format(x)

            data = (_format(self.angle), self.plane)
            _label += data

        _label += "├" if not self.reset else "│  │0⟩"

        return _label


def _xy_to_z(angle):
    """Project XY basis states onto computational basis states"""
    return np.array([[1, np.exp(-1j * angle)], [1, -np.exp(-1j * angle)]]) / np.sqrt(2)
