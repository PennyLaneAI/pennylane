# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains tests for 'get_compile_pipeline'."""

import re

import pytest

import pennylane as qml
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.transforms.core.compile_pipeline import CompilePipeline
from pennylane.transforms.core.transform import BoundTransform
from pennylane.workflow import get_compile_pipeline


class TestValidation:
    """Tests for validation errors."""

    @pytest.mark.parametrize(
        "unsupported_level",
        (
            [0],
            [0, 1],
            (0,),
            (0, 1),
        ),
    )
    def test_incorrect_level_type(self, unsupported_level):
        """Tests validation for incorrect level types."""

        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'level={unsupported_level}' of type '{type(unsupported_level)}' is not supported."
            ),
        ):
            _ = get_compile_pipeline(circuit, level=unsupported_level)()

    def test_missing_marker_label(self):
        """Tests that a string that doesn't have a marker raises an exception."""

        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=re.escape("level my_marker not found in compile pipeline. "),
        ):
            _ = get_compile_pipeline(circuit, level="my_marker")()

    def test_gradient_level_with_final_transform(self):
        """Tests that a final transform causes an exception to be raised"""

        dev = qml.device("reference.qubit")

        @qml.gradients.metric_tensor
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(ValueError, match="Cannot retrieve compile pipeline"):
            _ = get_compile_pipeline(circuit, level="gradient")()

    def test_device_level_with_final_transform(self):
        """Tests that a final transform is correctly re-appended."""

        dev = qml.device("reference.qubit")

        @qml.gradients.metric_tensor
        @qml.transforms.merge_rotations(atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(ValueError, match="Cannot retrieve compile pipeline"):
            _ = get_compile_pipeline(circuit, level="device")()


class TestUserLevel:
    """Tests 'user' level transforms."""

    def test_user_level_pipeline(self):
        """Tests the contents of a user level pipeline."""

        dev = qml.device("reference.qubit")

        @qml.transforms.merge_rotations(atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()
        assert len(cp) == 2
        assert cp[0].tape_transform == qml.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qml.transforms.merge_rotations.tape_transform

    def test_user_level_with_final_transform(self):
        """Tests that a final transform is correctly re-appended."""

        dev = qml.device("reference.qubit")

        @qml.gradients.metric_tensor
        @qml.transforms.merge_rotations(atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()

        assert len(cp) == 4
        assert cp[0].tape_transform == qml.transforms.cancel_inverses.tape_transform
        assert cp[1].tape_transform == qml.transforms.merge_rotations.tape_transform
        assert cp[2].tape_transform == qml.gradients.metric_tensor.expand_transform
        assert cp[3].tape_transform == qml.gradients.metric_tensor.tape_transform

    def test_no_user_levels(self):
        """Ensures an empty compile pipeline if no user transforms."""
        dev = qml.device("reference.qubit")

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="user")()

        assert cp == CompilePipeline()


class TestGradientLevel:
    """Tests 'device' level transforms."""

    def test_gradient_level_pipeline(self):
        """Tests the contents of a gradient level pipeline."""

        dev = qml.device("reference.qubit")

        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": 2})
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="gradient")()

        assert len(cp) == 1
        assert cp[0] == BoundTransform(
            qml.transform(qml.gradients.param_shift.expand_transform), kwargs={"shifts": 2}
        )

    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_no_gradient_levels(self, diff_method):
        """Ensures an empty compile pipeline if no gradient transforms."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="gradient")()

        assert cp == CompilePipeline()


class TestDeviceLevel:
    """Tests 'device' level transforms."""

    @pytest.mark.parametrize("device", ["reference.qubit", "default.qubit"])
    def test_reference_qubit(self, device):
        """Tests the contents of a device level pipeline."""

        dev = qml.device(device)

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.expval(qml.Z(0))

        cp = get_compile_pipeline(circuit, level="device")()

        expected_cp = CompilePipeline()

        # User transforms
        expected_cp.add_transform(qml.transforms.cancel_inverses)
        expected_cp.add_transform(qml.transforms.merge_rotations)

        # Gradient expansion
        expected_cp.add_transform(qml.transform(qml.gradients.param_shift.expand_transform))

        # Device transforms
        device_cp, _ = dev.preprocess()
        expected_cp += device_cp

        assert cp == expected_cp

    def test_device_derivative(self):
        """Tests the compile pipeline if a device derivative exists."""

        dev = qml.device("default.qubit")

        @qml.transforms.split_non_commuting
        @qml.qnode(dev, diff_method="adjoint", device_vjp=False)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        cp = get_compile_pipeline(circuit, level="device")(1.0)

        expected_cp = CompilePipeline()
        # User transforms
        expected_cp.add_transform(qml.transforms.split_non_commuting)

        # Device transforms
        basic_config = ExecutionConfig(gradient_method="adjoint", use_device_jacobian_product=False)
        resolved_config = dev.setup_execution_config(basic_config)
        expected_cp += dev.preprocess_transforms(resolved_config)

        assert cp == expected_cp


def test_marker_level():
    """Tests that a string corresponding to a marker level can be used."""

    dev = qml.device("reference.qubit")

    @qml.transforms.merge_rotations
    @qml.marker("blah")
    @qml.transforms.cancel_inverses
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit():
        return qml.expval(qml.Z(0))

    cp = get_compile_pipeline(circuit, level="blah")()
    assert len(cp) == 1
    assert cp[0].tape_transform == qml.transforms.cancel_inverses.tape_transform


def test_level_is_top():
    """Tests that level is top returns an empty pipeline."""

    dev = qml.device("reference.qubit")

    @qml.transforms.merge_rotations(atol=1e-5)
    @qml.transforms.cancel_inverses
    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.Z(0))

    cp = get_compile_pipeline(circuit, level="top")()
    assert cp == CompilePipeline()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_level_is_integer(level):
    """Tests that levels can be integers corresponding to their position
    in the compile pipeline."""

    dev = qml.device("reference.qubit")

    @qml.transforms.merge_rotations
    @qml.transforms.cancel_inverses
    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.Z(0))

    cp = get_compile_pipeline(circuit, level=level)()

    if level == 0:
        assert cp == CompilePipeline()
    elif level == 1:
        assert cp == CompilePipeline(qml.transforms.cancel_inverses)
    elif level == 2:
        assert cp == CompilePipeline(qml.transforms.cancel_inverses, qml.transforms.merge_rotations)
