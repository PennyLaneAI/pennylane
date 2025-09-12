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
from pennylane.labs.resource_estimation.templates.qubitize import ResourceQubitizeTHC
from pennylane.labs.resource_estimation.templates.select import ResourceSelectTHC
from pennylane.labs.resource_estimation.templates.stateprep import (
    ResourceAliasSampling,
    ResourceMPSPrep,
    ResourcePrepTHC,
    ResourceQROMStatePreparation,
)
from pennylane.labs.resource_estimation.templates.subroutines import ResourceQubitUnitary

# pylint: disable=protected-access, unused-argument


class DummyOp(ResourceOperator):
    """Dummy ResourceOperator for testing."""

    num_wires = 1


def dummy_decomp_func(**kwargs):
    """Dummy decomposition function for testing."""
    return []


class TestResourceConfig:
    """Test the ResourceConfig class and its methods."""

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_initialization_of_custom_decomps(self):
        """Test that the custom decomposition dictionaries initialize as empty."""
        config = ResourceConfig()
        assert config._pow_custom_decomps == {}
        assert config._custom_decomps == {}
        assert config._adj_custom_decomps == {}
        assert config._ctrl_custom_decomps == {}

    @pytest.mark.parametrize(
        "op_with_precision",
        [
            ResourceRX,
            ResourceRY,
            ResourceRZ,
            ResourceCRX,
            ResourceCRY,
            ResourceCRZ,
            ResourceSelectPauliRot,
            ResourceQubitUnitary,
            ResourceQROMStatePreparation,
            ResourceMPSPrep,
            ResourceAliasSampling,
        ],
    )
    def test_initialization_sets_default_precision(self, op_with_precision):
        """Test that default precision is set for standard operations on initialization."""
        config = ResourceConfig()
        assert op_with_precision in config.resource_op_precisions
        assert config.resource_op_precisions[op_with_precision].get("precision") > 0

    @pytest.mark.parametrize(
        "op_with_rotation_and_coeff_precision",
        [
            ResourceQubitizeTHC,
            ResourceSelectTHC,
            ResourcePrepTHC,
        ],
    )
    def test_initialization_sets_default_rotation_precision(
        self, op_with_rotation_and_coeff_precision
    ):
        """Test that default precision is set for standard operations on initialization."""
        config = ResourceConfig()
        assert op_with_rotation_and_coeff_precision in config.resource_op_precisions
        assert (
            config.resource_op_precisions[op_with_rotation_and_coeff_precision].get(
                "rotation_precision"
            )
            > 0
        )
        assert (
            config.resource_op_precisions[op_with_rotation_and_coeff_precision].get(
                "coeff_precision"
            )
            > 0
        )

    @pytest.mark.parametrize(
        "rotation_op",
        [
            ResourceRX,
            ResourceRY,
            ResourceRZ,
            ResourceCRX,
            ResourceCRY,
            ResourceCRZ,
        ],
    )
    def test_set_single_qubit_rotation_precision_for_rotation_ops(self, rotation_op):
        """Test that the single qubit rotation error is set correctly for a given rotation gate."""
        config = ResourceConfig()
        new_error = 1e-5
        config.set_single_qubit_rot_precision(new_error)

        assert config.resource_op_precisions[rotation_op]["precision"] == new_error

    # pylint: disable=use-implicit-booleaness-not-comparison
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
        config.set_decomp(DummyOp, dummy_decomp_func, decomp_type=decomp_type)

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
                assert getattr(config, dict_name) == {}

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
            f"custom_decomps = {{}}, adj_custom_decomps = {{}}, "
            f"ctrl_custom_decomps = {{{DummyOp}: {dummy_decomp_func}}}, "
            f"pow_custom_decomps = {{}})"
        )

        assert repr_str == expected_repr_str

    def test_set_decomp_invalid_type_raises_error(self):
        """Test that set_decomp raises a ValueError for an invalid decomposition type."""
        config = ResourceConfig()
        invalid_type_string = "this_is_not_a_valid_type"

        def dummy_decomp(**kwargs):
            """A placeholder decomposition function."""
            return []

        with pytest.raises(
            ValueError, match=f"'{invalid_type_string}' is not a valid DecompositionType"
        ):
            config.set_decomp(ResourceRX, dummy_decomp, decomp_type=invalid_type_string)

    @pytest.mark.parametrize(
        "op_type_to_set",
        [
            ResourceSelectPauliRot,
            ResourceRX,
            ResourceQubitUnitary,
        ],
    )
    def test_set_precision_updates_correctly(self, op_type_to_set):
        """Test that set_precision correctly updates the precision for a valid operator."""
        config = ResourceConfig()
        new_precision = 1e-5

        assert config.resource_op_precisions[op_type_to_set]["precision"] != new_precision

        config.set_precision(op_type_to_set, new_precision)
        assert config.resource_op_precisions[op_type_to_set]["precision"] == new_precision

    def test_set_precision_raises_error_for_unknown_op(self):
        """Test that set_precision raises ValueError for an unknown operator."""
        config = ResourceConfig()
        match_str = "DummyOp is not a configurable operator. Configurable operators are:"
        with pytest.raises(ValueError, match=match_str):
            config.set_precision(DummyOp, 0.123)

    @pytest.mark.parametrize(
        "unsupported_op",
        [
            ResourceQubitizeTHC,
            ResourceSelectTHC,
            ResourcePrepTHC,
        ],
    )
    def test_set_precision_raises_error_for_unsupported_op(self, unsupported_op):
        """Test ValueError for known operators that do not support single-precision setting."""
        config = ResourceConfig()
        match_str = f"Setting precision for {unsupported_op.__name__} is not supported."
        with pytest.raises(ValueError, match=match_str):
            config.set_precision(unsupported_op, 0.456)

    def test_set_single_qubit_rot_precision_raises_for_negative_value(self):
        """Test that set_single_qubit_rot_precision raises a ValueError for a negative precision."""
        config = ResourceConfig()
        negative_precision = -0.1

        with pytest.raises(ValueError, match="Precision must be a non-negative value"):
            config.set_single_qubit_rot_precision(negative_precision)

    def test_set_precision_raises_for_negative_value(self):
        """Test that set_precision raises a ValueError for a negative precision."""
        config = ResourceConfig()
        with pytest.raises(ValueError, match="Precision must be a non-negative value"):
            config.set_precision(ResourceRX, precision=-1e-5)
