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
Tests for qchem resource operators.
"""
import pytest

import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.resource_config import ResourceConfig

# pylint: disable=no-self-use,too-many-arguments


class TestResourceQubitUnitary:
    """Test the ResourceQubitUnitary template"""

    def test_init_no_num_wires(self):
        """Test that we can instantiate the operator without providing num_wires"""
        op = qre.QubitUnitary(wires=range(3))
        assert op.resource_params == {"num_wires": 3, "precision": None}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide atleast one of"):
            qre.QubitUnitary()

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_params(self, num_wires, precision):
        """Test that the resource params are correct."""
        op = qre.QubitUnitary(num_wires, precision) if precision else qre.QubitUnitary(num_wires)
        assert op.resource_params == {"num_wires": num_wires, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_rep(self, num_wires, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.QubitUnitary, num_wires, {"num_wires": num_wires, "precision": precision}
        )
        assert qre.QubitUnitary.resource_rep(num_wires=num_wires, precision=precision) == expected

    @pytest.mark.parametrize(
        "num_wires, precision, expected_res",
        (
            (
                1,
                None,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-9})),
                ],
            ),
            (
                2,
                1e-3,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-3}), 4),
                    GateCount(resource_rep(qre.CNOT), 3),
                ],
            ),
            (
                5,
                1e-5,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-5}), (4**3) * 4),
                    GateCount(resource_rep(qre.CNOT), (4**3) * 3),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Z",
                                "num_ctrl_wires": 2,
                                "precision": 1e-5,
                            },
                        ),
                        2 * 4**2,
                    ),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Y",
                                "num_ctrl_wires": 2,
                                "precision": 1e-5,
                            },
                        ),
                        4**2,
                    ),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Z",
                                "num_ctrl_wires": 3,
                                "precision": 1e-5,
                            },
                        ),
                        2 * 4,
                    ),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Y",
                                "num_ctrl_wires": 3,
                                "precision": 1e-5,
                            },
                        ),
                        4,
                    ),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Z",
                                "num_ctrl_wires": 4,
                                "precision": 1e-5,
                            },
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
                            {
                                "rot_axis": "Y",
                                "num_ctrl_wires": 4,
                                "precision": 1e-5,
                            },
                        ),
                        1,
                    ),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_wires, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qre.QubitUnitary]
            assert qre.QubitUnitary.resource_decomp(num_wires=num_wires, **kwargs) == expected_res
        else:
            assert (
                qre.QubitUnitary.resource_decomp(num_wires=num_wires, precision=precision)
                == expected_res
            )
