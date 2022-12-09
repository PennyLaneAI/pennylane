# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""module docstring"""
from typing import Sequence, Union, Callable

from pennylane.operation import Operator
from pennylane.measurements import MeasurementProcess
from pennylane.tape import QuantumScript


class DefaultPreprocessor:
    """PROTOTYPE"""

    def __init__(
        self, operations: set = None, observables: set = None, measurement_processes: set = None
    ):
        self.supported_operations = operations or {}
        self.supported_observables = observables or {}
        self.supported_measurement_processes = measurement_processes or {}

        self.processing_steps = [self.expand_till_supported]

    # pylint: disable=unused-argument
    def __call__(
        self, qscript: Union[QuantumScript, Sequence[QuantumScript]], execution_config=None
    ) -> tuple[Sequence[QuantumScript], Callable]:
        if isinstance(qscript, QuantumScript):
            qscript = [qscript]

        def post_processing_fn(res):
            """Identity post processing function to start the DefaultProcessor with."""
            return res

        for step in self.processing_steps:
            qscript, new_fn = step(qscript, execution_config)

        return qscript, lambda res: new_fn(post_processing_fn(res))

    @property
    def processing_steps_names(self):
        """docstring"""
        return [fn.__name__ for fn in self.processing_steps]

    def _is_supported_object(self, obj):
        if isinstance(obj, Operator):
            return obj.name in self.supported_operations
        if isinstance(obj, MeasurementProcess):
            if obj.obs is not None:
                return obj.obs in self.supported_observables
            if self.supported_measurement_processes:
                return obj.return_type in self.supported_measurement_processes
        return False

    def expand_till_supported(self, qscripts: Sequence[QuantumScript], execution_config=None):
        """Uses ``qml.tape.tape_expand`` to decompose operations till they are supported."""

        new_qscripts = [
            qs.expand(depth=1000, stop_at=self._is_supported_object, expand_measurements=False)
            for qs in qscripts
        ]

        for qs in new_qscripts:
            for op in qs.operations:
                if not self._is_supported_object(op):
                    raise NotImplementedError(f"{op} is not supported by the requested device.")

        def identity_post_processing_fn(res):
            """Identity post processing function produced by DefaultProcessor.expand_till_supported"""
            return res

        return new_qscripts, identity_post_processing_fn
