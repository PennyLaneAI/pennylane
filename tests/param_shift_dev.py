# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This file provides a device that calculates derivatives via parameter shift.
"""

import dataclasses

import pennylane as qml


# pylint: disable=unused-argument
class ParamShiftDerivativesDevice(qml.devices.DefaultQubit):
    """This device provides derivatives via parameter shift."""

    name = "param_shift.qubit"

    def preprocess(self, execution_config=qml.devices.DefaultExecutionConfig):
        if config.gradient_method in {"device", "parameter-shift"}:
            config = dataclasses.replace(config, use_device_gradient=True)
        return super().preprocess(config)

    def supports_derivatives(self, execution_config=None, circuit=None):
        if execution_config is None:
            return True
        return execution_config.gradient_method in {"device", "parameter-shift"}

    def compute_derivatives(self, circuits, execution_config=None):
        is_single_circuit = False
        if isinstance(circuits, qml.tape.QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)

        if self.tracker.active:
            self.tracker.update(derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()

        diff_batch, fn = qml.transforms.map_batch_transform(qml.gradients.param_shift, circuits)
        diff_results = self.execute(diff_batch)

        jacs = fn(diff_results)
        return jacs[0] if is_single_circuit else jacs

    def execute_and_compute_derivatives(self, circuits, execution_config=None):
        is_single_circuit = False
        if isinstance(circuits, qml.tape.QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_derivative_batches=1,
                derivatives=len(circuits),
            )
            self.tracker.record()

        diff_batch, fn = qml.transforms.map_batch_transform(qml.gradients.param_shift, circuits)
        combined_batch = tuple(circuits) + tuple(diff_batch)
        all_results = self.execute(combined_batch)
        results = all_results[: len(circuits)]
        jacs = fn(all_results[len(circuits) :])
        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)
