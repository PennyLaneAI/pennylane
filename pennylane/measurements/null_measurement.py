# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains the qml.mutual_info measurement.
"""
import numpy as np

from .measurements import SampleMeasurement, StateMeasurement


class NullMeasurement(SampleMeasurement, StateMeasurement):
    """A measurement that strictly returns an array with one nan.

    This measurement is for profiling problems without the overhead of performing a measurement.

    >>> @qml.qnode(qml.device('default.qubit', wires=1), diff_method="parameter-shift")
    ... def circuit():
    ...     return qml.measurements.NullMeasurement()
    ...
    >>> circuit()
    array(nan)

    ``np.array(np.nan)`` is chosen so the result still has a shape and data type for integration
    with jax, catalyst, and program capture.

    """

    _shortname = "null"
    numeric_type = float

    @classmethod
    def _abstract_eval(cls, *_, **__):
        return (), float

    def shape(self, *_, **__):
        return ()

    def process_density_matrix(self, *_, **__):
        return np.array(np.nan)

    def process_samples(self, *_, **__):
        return np.array(np.nan)

    def process_counts(self, *_, **__):
        return np.array(np.nan)

    def process_state(self, *_, **__):
        return np.array(np.nan)
