# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Test the resource configuration class.
"""
import pytest

from pennylane.estimator.ops.op_math.controlled_ops import CRX, CRY, CRZ
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import RX, RY, RZ
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_operator import ResourceOperator

# pylint: disable=protected-access, too-few-public-methods


class DummyOp(ResourceOperator):
    """Dummy ResourceOperator for testing."""

    num_wires = 1


# pylint: disable=unused-argument
def dummy_decomp_func(**kwargs):
    """Dummy decomposition function for testing."""
    return []


class TestResourceConfig:
    """Test the ResourceConfig class and its methods."""

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_initialization_of_custom_decomps(self):
        """Test that the custom decomposition dictionaries initialize as empty."""
        config = ResourceConfig()
        assert config.pow_custom_decomps == {}
        assert config.custom_decomps == {}
        assert config.adj_custom_decomps == {}
        assert config.ctrl_custom_decomps == {}

    # pylint: disable=use-implicit-booleaness-not-comparison
    @pytest.mark.parametrize(
        "decomp_type, target_dict_name",
        [
            (None, "custom_decomps"),
            ("adj", "adj_custom_decomps"),
            ("ctrl", "ctrl_custom_decomps"),
            ("pow", "pow_custom_decomps"),
        ],
    )
    def test_set_decomp(self, decomp_type, target_dict_name):
        """Test that set_decomp correctly registers a custom decomposition for various types."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type=decomp_type)

        target_dict = getattr(config, target_dict_name)
        assert len(target_dict) == 1
        assert target_dict[DummyOp] is dummy_decomp_func

        all_dicts = [
            "custom_decomps",
            "adj_custom_decomps",
            "ctrl_custom_decomps",
            "pow_custom_decomps",
        ]
        for dict_name in all_dicts:
            if dict_name != target_dict_name:
                assert getattr(config, dict_name) == {}

    def test_public_accessors_for_decomps(self):
        """Test that the public properties correctly return the custom decomposition dictionaries."""
        config = ResourceConfig()

        # Set one of each type
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="base")
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="adj")
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="ctrl")
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="pow")

        # Check properties
        assert config.custom_decomps == {DummyOp: dummy_decomp_func}
        assert config.adj_custom_decomps == {DummyOp: dummy_decomp_func}
        assert config.ctrl_custom_decomps == {DummyOp: dummy_decomp_func}
        assert config.pow_custom_decomps == {DummyOp: dummy_decomp_func}

    def test_str_representation(self):
        """Test the user-friendly string representation of the ResourceConfig class."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func)
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="adj")
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="ctrl")
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="pow")

        dict_items_str = ",\n".join(
            f"    {key.__name__}: {value!r}" for key, value in config.resource_op_precisions.items()
        )
        formatted_dict = f"{{\n{dict_items_str}\n}}"
        op_names = "DummyOp, Adjoint(DummyOp), Controlled(DummyOp), Pow(DummyOp)"

        expected_str = (
            f"ResourceConfig(\n"
            f"  precisions = {formatted_dict},\n"
            f"  custom decomps = [{op_names}]\n)"
        )

        assert str(config) == expected_str

    def test_repr_representation(self):
        """Test the representation of the ResourceConfig class."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type="ctrl")

        repr_str = repr(config)
        expected_repr_str = (
            f"ResourceConfig(precisions = {config.resource_op_precisions}, "
            f"custom_decomps = {config.custom_decomps}, adj_custom_decomps = {config.adj_custom_decomps}, "
            f"ctrl_custom_decomps = {config.ctrl_custom_decomps}, "
            f"pow_custom_decomps = {config.pow_custom_decomps})"
        )

        assert repr_str == expected_repr_str

    def test_set_precision_raises_error_for_unknown_op(self):
        """Test that set_precision raises ValueError for an unknown operator."""
        config = ResourceConfig()
        match_str = "DummyOp is not a configurable operator. Configurable operators are:"
        with pytest.raises(ValueError, match=match_str):
            config.set_precision(DummyOp, 0.123)

    def test_set_single_qubit_rot_precision_raises_for_negative_value(self):
        """Test that set_single_qubit_rot_precision raises a ValueError for a negative precision."""
        config = ResourceConfig()
        negative_precision = -0.1

        with pytest.raises(ValueError, match="Precision must be a non-negative value"):
            config.set_single_qubit_rot_precision(negative_precision)

    def test_set_single_qubit_rot_precision(self):
        """Test that the set_single_qubit_rot_precision works as expected"""
        config = ResourceConfig()
        custom_precision = 1.23 * 1e-4

        for single_qubit_rot_op in [RX, RY, RZ, CRX, CRY, CRZ]:
            assert (
                config.resource_op_precisions[single_qubit_rot_op]["precision"] != custom_precision
            )

        config.set_single_qubit_rot_precision(precision=custom_precision)
        for single_qubit_rot_op in [RX, RY, RZ, CRX, CRY, CRZ]:
            assert (
                config.resource_op_precisions[single_qubit_rot_op]["precision"] == custom_precision
            )

    def test_set_precision_raises_for_negative_value(self):
        """Test that set_precision raises a ValueError for a negative precision."""
        config = ResourceConfig()
        config.resource_op_precisions[DummyOp] = {"precision": 1e-9}
        negative_precision = -0.1
        with pytest.raises(ValueError, match="Precision must be a non-negative value"):
            config.set_precision(DummyOp, negative_precision)

    def test_set_precision_raises_error_for_unsupported_op(self):
        """Test that set_precision raises ValueError for an unsupported operator."""
        config = ResourceConfig()
        config.resource_op_precisions[DummyOp] = {"bits": 10}
        match_str = "Setting precision for DummyOp is not supported."
        with pytest.raises(ValueError, match=match_str):
            config.set_precision(DummyOp, 0.123)

    def test_set_precision_sets_value(self):
        """Test that set_precision correctly sets the precision for a supported operator."""
        config = ResourceConfig()
        config.resource_op_precisions[DummyOp] = {"precision": 1e-9}
        new_precision = 1e-5
        config.set_precision(DummyOp, new_precision)
        assert config.resource_op_precisions[DummyOp]["precision"] == new_precision
