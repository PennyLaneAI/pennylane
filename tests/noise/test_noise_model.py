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

import pennylane as qp
from pennylane.noise import meas_eq, op_eq, op_in, partial_wires, wires_eq, wires_in


class TestNoiseModels:
    """Test for Noise Models class and its methods"""

    @pytest.mark.parametrize(
        ("fcond", "noise"),
        [
            (op_eq(qp.X) | op_eq(qp.Y), partial_wires(qp.AmplitudeDamping, 0.4)),
            (op_eq(qp.X) & op_eq(qp.Y), partial_wires(qp.RX, 0.4)),
            (wires_eq(qp.X(0)) & op_eq(qp.X(0)), partial_wires(qp.PauliRot(1.2, "XY", [0, 1]))),
            (~wires_eq(qp.X(0)) ^ op_eq(qp.X(0)), lambda op, **kwargs: qp.RZ(0.2, op.wires)),
            (
                wires_in(qp.CNOT([0, 1])) | ~op_in(["RY", "RZ"]),
                lambda op, **kwargs: qp.RX(0.2, op.wires),
            ),
            (wires_in(["a", "b"]) & op_eq("RY"), lambda op, **kwargs: qp.RX(0.2, op.wires)),
        ],
    )
    def test_building_noise_model(self, fcond, noise):
        """Test that noise models are built correctly using NoiseModels"""

        noise_model = qp.NoiseModel({fcond: noise}, t1=0.04, t2=0.02)

        assert fcond in noise_model.model_map
        assert noise in noise_model.model_map.values()
        assert all(t in noise_model.metadata for t in ["t1", "t2"])
        assert list(noise_model.metadata.values()) == [0.04, 0.02]
        assert (
            repr(noise_model) == "NoiseModel({\n"
            f"    {fcond}: {noise.__name__}" + "\n"
            "}, t1 = 0.04, t2 = 0.02)"
        )

    @pytest.mark.parametrize(
        ("model", "meas", "metadata"),
        [
            (
                {op_eq(qp.X) | op_eq(qp.Y): partial_wires(qp.AmplitudeDamping, 0.4)},
                {meas_eq(qp.expval) & wires_in([0, 1]): partial_wires(qp.PhaseFlip, 0.2)},
                {"t1": 0.04, "t2": 0.02},
            ),
            (
                {
                    wires_eq(qp.X(0))
                    & op_eq(qp.X(0)): partial_wires(qp.PauliRot(1.2, "XY", [0, 1]))
                },
                {meas_eq(qp.vn_entropy([0])) & op_eq(qp.X): partial_wires(qp.BitFlip, 0.2)},
                {"log_base": 4},
            ),
            (
                {~wires_eq(qp.X(0)) ^ op_eq(qp.X(0)): lambda op, **kwargs: qp.RZ(0.2, op.wires)},
                {
                    meas_eq(qp.sample(qp.X(0) @ qp.Y(1)))
                    | meas_eq(qp.counts(wires=[0])): lambda op, **kwargs: qp.RX(0.2, op.wires)
                },
                {},
            ),
        ],
    )
    def test_building_noise_model_with_meas(self, model, meas, metadata):
        """Test that noise models are built correctly using NoiseModels with readout noise"""

        noise_model = qp.NoiseModel(model, meas_map=meas, **metadata)

        assert model == noise_model.model_map
        assert meas == noise_model.meas_map
        assert metadata == noise_model.metadata

        model_str = "\n".join(["    " + f"{key}: {val.__name__}" for key, val in model.items()])
        meas_str = "\n".join(["    " + f"{key}: {val.__name__}" for key, val in meas.items()])
        meta_str = ", ".join([f"{key} = {val}" for key, val in metadata.items()])

        assert (
            repr(noise_model)
            == "NoiseModel({\n"
            + model_str
            + "\n},\nmeas_map = {\n"
            + meas_str
            + "\n}"
            + (", " + meta_str if meta_str else "")
            + ")"
        )

    # pylint: disable=comparison-with-callable
    def test_add_noise_models(self):
        """Test that noise models can be added and manipulated"""

        fcond = qp.noise.op_eq(qp.X) | qp.noise.op_eq(qp.Y)
        noise = qp.noise.partial_wires(qp.AmplitudeDamping, 0.4)
        noise_model = qp.NoiseModel({fcond: noise}, t1=0.04)

        @qp.BooleanFn
        def fcond1(op):
            return isinstance(op, qp.RY) and op.parameters[0] >= 0.5

        noise1 = qp.noise.partial_wires(qp.PhaseDamping, 0.9)
        add_model = noise_model + {fcond1: noise1}
        assert add_model.model_map[fcond1] == noise1

        radd_model = {fcond1: noise1} + noise_model
        assert radd_model == add_model

        meas_fcond = meas_eq(qp.expval) & wires_in([0, 1])
        meas_noise = partial_wires(qp.PhaseFlip, 0.2)

        noise_model1 = qp.NoiseModel({fcond1: noise1}, {meas_fcond: meas_noise}, t2=0.02)
        nadd_model = noise_model + noise_model1
        assert nadd_model.model_map[fcond1] == noise1
        assert nadd_model.meas_map[meas_fcond] == meas_noise
        assert nadd_model.metadata["t2"] == noise_model1.metadata["t2"]

    # pylint: disable=comparison-with-callable
    def test_sub_noise_models(self):
        """Test that noise models can be subtracted and manipulated"""

        fcond = qp.noise.op_eq(qp.X) | qp.noise.op_eq(qp.Y)
        noise = qp.noise.partial_wires(qp.AmplitudeDamping, 0.4)

        @qp.BooleanFn
        def fcond1(op):
            return isinstance(op, qp.RY) and op.parameters[0] >= 0.5

        noise1 = qp.noise.partial_wires(qp.PhaseDamping, 0.9)

        m_fcond = meas_eq(qp.expval) & wires_in([0, 1])
        m_noise = partial_wires(qp.PhaseFlip, 0.2)

        noise_model = qp.NoiseModel(
            {fcond: noise, fcond1: noise1}, {m_fcond: m_noise}, t1=0.04, t2=0.02
        )

        sub_model = noise_model - {fcond1: noise1}
        assert qp.NoiseModel({fcond: noise}, {m_fcond: m_noise}, t1=0.04, t2=0.02) == sub_model

        sub_model1 = noise_model - qp.NoiseModel(
            {fcond1: noise1}, meas={m_fcond: m_noise}, t2=0.02
        )
        assert qp.NoiseModel({fcond: noise}, t1=0.04) == sub_model1

    # pylint: disable=comparison-with-callable, unused-argument
    def test_eq_noise_models(self):
        """Test that noise models can be subtracted and manipulated"""

        fcond = qp.noise.op_eq(qp.X) | qp.noise.op_eq(qp.Y)

        def noise(op, **kwargs):
            qp.RX(op.parameters[0] * 0.05, op.wires)

        # explicit construction
        noise_model = qp.NoiseModel({fcond: noise})
        noise_model2 = qp.NoiseModel({fcond: noise})
        assert noise_model == noise_model2

        # implicit construction
        noise_model = qp.NoiseModel({qp.noise.op_eq(qp.X): noise})
        noise_model2 = qp.NoiseModel({qp.noise.op_eq(qp.X): noise})
        assert noise_model == noise_model2

        m_fcond = meas_eq(qp.expval) & wires_in([0, 1])
        m_noise = partial_wires(qp.PhaseFlip, 0.2)
        noise_model = qp.NoiseModel({fcond: noise}, {m_fcond: m_noise})
        noise_model2 = qp.NoiseModel({fcond: noise}, {m_fcond: m_noise})
        assert noise_model == noise_model2

        # check inequality
        @qp.BooleanFn
        def fcond1(op):
            return isinstance(op, qp.RY) and op.parameters[0] >= 0.5

        noise_model = qp.NoiseModel({fcond1: noise})
        noise_model2 = qp.NoiseModel({fcond: noise})
        assert noise_model != noise_model2

        noise1 = qp.noise.partial_wires(qp.AmplitudeDamping, 0.4)
        noise_model = qp.NoiseModel({fcond: noise})
        noise_model2 = qp.NoiseModel({fcond: noise1})
        assert noise_model != noise_model2

        noise_model = qp.NoiseModel({fcond: noise}, {m_fcond: noise1})
        noise_model2 = qp.NoiseModel({fcond: noise}, {m_fcond: m_noise})
        assert noise_model != noise_model2

    def test_build_model_errors(self):
        """Test for checking building noise models raise correct error when signatures are not proper"""

        with pytest.raises(ValueError, match="must be a boolean conditional"):
            qp.NoiseModel({lambda x: x: qp.noise.partial_wires(qp.X)})

        with pytest.raises(
            ValueError, match=r"must accept \*\*kwargs as the last argument in its signature"
        ):
            qp.NoiseModel({qp.noise.op_eq(qp.X): lambda x: x})
