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
import pytest

import pennylane as qml
from pennylane.noise import op_eq, op_in, partial_wires, wires_eq, wires_in


class TestNoiseModels:
    """Test for Noise Models class and its methods"""

    @pytest.mark.parametrize(
        ("fcond", "noise"),
        [
            (op_eq(qml.X) | op_eq(qml.Y), partial_wires(qml.AmplitudeDamping, 0.4)),
            (op_eq(qml.X) & op_eq(qml.Y), partial_wires(qml.RX, 0.4)),
            (wires_eq(qml.X(0)) & op_eq(qml.X(0)), partial_wires(qml.PauliRot(1.2, "XY", [0, 1]))),
            (~wires_eq(qml.X(0)) ^ op_eq(qml.X(0)), lambda op, **kwargs: qml.RZ(0.2, op.wires)),
            (
                wires_in(qml.CNOT([0, 1])) | ~op_in(["RY", "RZ"]),
                lambda op, **kwargs: qml.RX(0.2, op.wires),
            ),
            (wires_in(["a", "b"]) & op_eq("RY"), lambda op, **kwargs: qml.RX(0.2, op.wires)),
        ],
    )
    def test_building_noise_model(self, fcond, noise):
        """Test that noise models are built correctly using NoiseModels"""

        noise_model = qml.NoiseModel({fcond: noise}, t1=0.04, t2=0.02)

        assert fcond in noise_model.model_map
        assert noise in noise_model.model_map.values()
        assert all(t in noise_model.metadata for t in ["t1", "t2"])
        assert list(noise_model.metadata.values()) == [0.04, 0.02]
        assert (
            repr(noise_model) == "NoiseModel({\n"
            f"    {fcond}: {noise.__name__}" + "\n"
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
        assert add_model.model_map[fcond1] == noise1

        radd_model = {fcond1: noise1} + noise_model
        assert radd_model == add_model

        noise_model1 = qml.NoiseModel({fcond1: noise1}, t2=0.02)
        nadd_model = noise_model + noise_model1
        assert nadd_model.model_map[fcond1] == noise1
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

    # pylint: disable=comparison-with-callable, unused-argument
    def test_eq_noise_models(self):
        """Test that noise models can be subtracted and manipulated"""

        fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)

        def noise(op, **kwargs):
            qml.RX(op.parameters[0] * 0.05, op.wires)

        # explicit construction
        noise_model = qml.NoiseModel({fcond: noise})
        noise_model2 = qml.NoiseModel({fcond: noise})
        assert noise_model == noise_model2

        # implicit construction
        noise_model = qml.NoiseModel({qml.noise.op_eq(qml.X): noise})
        noise_model2 = qml.NoiseModel({qml.noise.op_eq(qml.X): noise})
        assert noise_model == noise_model2

        # check inequality
        @qml.BooleanFn
        def fcond1(op):
            return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

        noise_model = qml.NoiseModel({fcond1: noise})
        noise_model2 = qml.NoiseModel({fcond: noise})
        assert noise_model != noise_model2

        noise1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)
        noise_model = qml.NoiseModel({fcond: noise})
        noise_model2 = qml.NoiseModel({fcond: noise1})
        assert noise_model != noise_model2

    def test_build_model_errors(self):
        """Test for checking building noise models raise correct error when signatures are not proper"""

        with pytest.raises(ValueError, match="must be a boolean conditional"):
            qml.NoiseModel({lambda x: x: qml.noise.partial_wires(qml.X)})

        with pytest.raises(
            ValueError, match=r"must accept \*\*kwargs as the last argument in its signature"
        ):
            qml.NoiseModel({qml.noise.op_eq(qml.X): lambda x: x})
