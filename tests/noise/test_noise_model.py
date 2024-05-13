# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the available conditional utitlities for noise models.
"""

# pylint: disable = too-few-public-methods
import pennylane as qml


class TestNoiseModels:
    """Test for Noise Models class and its methods"""

    def test_building_noise_model(self):
        """Test that noise models are built correctly using NoiseModels"""
        fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)
        noise = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

        noise_model = qml.NoiseModel({fcond: noise}, t1=0.04, t2=0.02)

        assert fcond in noise_model.model
        assert noise in noise_model.model.values()
        assert all(t in noise_model.metadata for t in ["t1", "t2"])
        assert list(noise_model.metadata.values()) == [0.04, 0.02]
        assert (
            repr(noise_model) == "NoiseModel({\n"
            "    Or(OpEq('PauliX'), OpEq('PauliY')) = AmplitudeDamping(gamma=0.4)\n"
            "}, t1 = 0.04, t2 = 0.02)"
        )

    # pylint: disable=comparison-with-callable
    def test_add_noise_models(self):
        """Test that noise models can be added and manipulated"""

        fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)
        noise = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)
        noise_model = qml.NoiseModel({fcond: noise}, t1=0.04)

        @qml.BooleanFn
        def fcond1(op):
            return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

        noise1 = qml.noise.partial_wires(qml.PhaseDamping, 0.9)
        add_model = noise_model + {fcond1: noise1}
        assert add_model.model[fcond1] == noise1

        radd_model = {fcond1: noise1} + noise_model
        assert radd_model == add_model

        noise_model1 = qml.NoiseModel({fcond1: noise1}, t2=0.02)
        nadd_model = noise_model + noise_model1
        assert nadd_model.model[fcond1] == noise1
        assert nadd_model.metadata["t2"] == noise_model1.metadata["t2"]

    # pylint: disable=comparison-with-callable
    def test_sub_noise_models(self):
        """Test that noise models can be subtracted and manipulated"""

        fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)
        noise = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

        @qml.BooleanFn
        def fcond1(op):
            return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

        noise1 = qml.noise.partial_wires(qml.PhaseDamping, 0.9)
        noise_model = qml.NoiseModel({fcond: noise, fcond1: noise1}, t1=0.04, t2=0.02)

        sub_model = noise_model - {fcond1: noise1}
        assert qml.NoiseModel({fcond: noise}, t1=0.04, t2=0.02) == sub_model

        sub_model1 = noise_model - qml.NoiseModel({fcond1: noise1}, t2=0.02)
        assert qml.NoiseModel({fcond: noise}, t1=0.04) == sub_model1
