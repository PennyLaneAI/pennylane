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

from pennylane.labs.resource_estimation.ops.op_math.controlled_ops import (
    ResourceCRX,
    ResourceCRY,
    ResourceCRZ,
)
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    ResourceRX,
    ResourceRY,
    ResourceRZ,
)
from pennylane.labs.resource_estimation.resource_config import ResourceConfig
from pennylane.labs.resource_estimation.resource_operator import ResourceOperator
from pennylane.labs.resource_estimation.templates import ResourceSelectPauliRot

# pylint: disable=protected-access, unused-argument


class DummyOp(ResourceOperator):
    """Dummy ResourceOperator for testing."""

    num_wires = 1


def dummy_decomp_func(**kwargs):
    """Dummy decomposition function for testing."""
    return []


class TestResourceConfig:
    """Test the ResourceConfig class and its methods."""

    def test_initialization(self):
        """Test that the ResourceConfig class initializes correctly with default values."""
        config = ResourceConfig()

        assert isinstance(config.errors_and_precisions, dict)
        assert config.errors_and_precisions[ResourceRX] == {"precision": 1e-9}
        assert config.errors_and_precisions[ResourceCRZ] == {"precision": 1e-9}

        assert config._pow_custom_decomps == {}

    def test_set_single_qubit_rotation_error(self):
        """Test that the single qubit rotation error is set correctly across all relevant gates."""
        config = ResourceConfig()
        new_error = 1e-5
        config.set_single_qubit_rotation_error(new_error)

        rotation_ops = [ResourceRX, ResourceRY, ResourceRZ, ResourceCRX, ResourceCRY, ResourceCRZ]
        for op in rotation_ops:
            assert config.errors_and_precisions[op]["precision"] == new_error

        assert config.errors_and_precisions[ResourceSelectPauliRot]["precision"] == 1e-9

    @pytest.mark.parametrize(
        "decomp_type, target_dict_name",
        [
            (None, "_custom_decomps"),
            ("adj", "_adj_custom_decomps"),
            ("ctrl", "_ctrl_custom_decomps"),
            ("pow", "_pow_custom_decomps"),
        ],
    )
    def test_set_decomp(self, decomp_type, target_dict_name):
        """Test that set_decomp correctly registers a custom decomposition for various types."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func, type=decomp_type)

        target_dict = getattr(config, target_dict_name)
        assert len(target_dict) == 1
        assert target_dict[DummyOp] is dummy_decomp_func

        all_dicts = [
            "_custom_decomps",
            "_adj_custom_decomps",
            "_ctrl_custom_decomps",
            "_pow_custom_decomps",
        ]
        for dict_name in all_dicts:
            if dict_name != target_dict_name:
                assert not getattr(config, dict_name)

    def test_str_representation(self):
        """Test the user-friendly string representation of the ResourceConfig class."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func)
        config.set_decomp(DummyOp, dummy_decomp_func, type="adj")
        config.set_decomp(DummyOp, dummy_decomp_func, type="ctrl")
        config.set_decomp(DummyOp, dummy_decomp_func, type="pow")

        # Recreate the expected formatting from the __str__ method
        dict_items_str = ",\n".join(
            f"    {key.__name__}: {value!r}" for key, value in config.errors_and_precisions.items()
        )
        formatted_dict = f"{{\n{dict_items_str}\n}}"
        op_names = "DummyOp, Adjoint(DummyOp), Controlled(DummyOp), Pow(DummyOp)"

        expected_str = (
            f"ResourceConfig(\n"
            f"  errors and precisions = {formatted_dict},\n"
            f"  custom decomps = [{op_names}]\n)"
        )

        assert str(config) == expected_str

    def test_repr_representation(self):
        """Test the developer-focused representation of the ResourceConfig class."""
        config = ResourceConfig()
        config.set_decomp(DummyOp, dummy_decomp_func, type="ctrl")

        repr_str = repr(config)
        expected_repr_str = (
            f"ResourceConfig(errors_and_precisions = {config.errors_and_precisions}, "
            f"custom_decomps = {{}}, adj_custom_decomps = {{}}, "
            f"ctrl_custom_decomps = {{{DummyOp}: {dummy_decomp_func}}}, "
            f"pow_custom_decomps = {{}})"
        )

        assert repr_str == expected_repr_str
