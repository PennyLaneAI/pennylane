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
"""
Tests for quantum algorithmic subroutines resource operators.
"""

import re

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount
from pennylane.labs.tests.estimator_beta.utils import assert_decomp_equal
from pennylane.wires import Wires

# pylint: disable=no-self-use,too-many-arguments


class TestResourcePhaseAdder:
    """Test the PhaseAdder resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, mod, wires",
        (
            (5, 1, [1, 2, 3, 4, 5]),
            (5, 2, ["a", "b", "c", 1, 2]),
            (5, 5, None),
            (5, 2**5, None),
            (5, None, None),
        ),
    )
    def test_init(self, num_x_wires, mod, wires):
        """Test the init method of the class."""
        phase_add = qre.PhaseAdder(num_x_wires, mod, wires)
        resource_params = phase_add.resource_params

        if mod is None:
            mod = 2**num_x_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert phase_add.wires == (Wires(wires) if wires is not None else None)
        assert phase_add.num_wires == num_x_wires

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.PhaseAdder(num_x_wires=3, mod=mod)

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 3 wires,"):
            qre.PhaseAdder(num_x_wires=3, mod=3, wires=[0, 1])

    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.PhaseAdder(num_x_wires, mod)

        if mod is None:
            mod = 2**num_x_wires
        expected = qre.CompressedResourceOp(
            qre.PhaseAdder, num_x_wires, params={"num_x_wires": num_x_wires, "mod": mod}
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.PhaseAdder(num_x_wires=5, mod=2),
                [
                    qre.Allocate(2),
                    GateCount(qre.PhaseShift.resource_rep(), 5 + 1),
                    GateCount(qre.PhaseShift.resource_rep(), 5 + 1),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(5 + 1),
                            cmpr_target_op=qre.CNOT.resource_rep(),
                            num_wires=5 + 2,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 5 + 1),),
                                5 + 1,
                            ),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.Adjoint.resource_rep(
                                qre.Prod.resource_rep(
                                    (
                                        (qre.QFT.resource_rep(5 + 1), 1),
                                        (qre.PhaseShift.resource_rep(), 5 + 1),
                                    ),
                                    num_wires=5 + 1,
                                )
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(qre.X.resource_rep(), 1, 1),
                            num_wires=5 + 2,
                        )
                    ),
                    qre.Deallocate(2),
                ],
            ),
            (
                qre.PhaseAdder(num_x_wires=7, mod=31),
                [
                    qre.Allocate(2),
                    GateCount(qre.PhaseShift.resource_rep(), 7 + 1),
                    GateCount(qre.PhaseShift.resource_rep(), 7 + 1),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(7 + 1),
                            cmpr_target_op=qre.CNOT.resource_rep(),
                            num_wires=7 + 2,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 7 + 1),),
                                7 + 1,
                            ),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.Adjoint.resource_rep(
                                qre.Prod.resource_rep(
                                    (
                                        (qre.QFT.resource_rep(7 + 1), 1),
                                        (qre.PhaseShift.resource_rep(), 7 + 1),
                                    ),
                                    num_wires=7 + 1,
                                )
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(qre.X.resource_rep(), 1, 1),
                            num_wires=7 + 2,
                        )
                    ),
                    qre.Deallocate(2),
                ],
            ),
            (
                qre.PhaseAdder(num_x_wires=5, mod=32),
                [GateCount(qre.PhaseShift.resource_rep(), 5)],
            ),
            (
                qre.PhaseAdder(num_x_wires=5, mod=None),
                [GateCount(qre.PhaseShift.resource_rep(), 5)],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert_decomp_equal(op.resource_decomp(**op.resource_params), expected_decomp)

    @pytest.mark.parametrize("z", (1, 2, 3, 5))
    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_pow_resources(self, num_x_wires, mod, z):
        """Test the pow_resource_decomp works as expected."""
        op = qre.PhaseAdder(num_x_wires, mod)
        cmpr_op = op.resource_rep_from_op()
        assert op.pow_resource_decomp(z, op.resource_params) == [GateCount(cmpr_op)]


class TestResourceAdder:
    """Test the Adder resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, mod, wires",
        (
            (5, 1, [1, 2, 3, 4, 5]),
            (5, 2, ["a", "b", "c", 1, 2]),
            (5, 5, None),
            (5, 2**5, None),
            (5, None, None),
        ),
    )
    def test_init(self, num_x_wires, mod, wires):
        """Test the init method of the class."""
        adder = qre.Adder(num_x_wires, mod, wires)
        resource_params = adder.resource_params

        if mod is None:
            mod = 2**num_x_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert adder.wires == (Wires(wires) if wires is not None else None)
        assert adder.num_wires == num_x_wires

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 3 wires,"):
            qre.Adder(num_x_wires=3, mod=3, wires=[0, 1])

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.Adder(num_x_wires=3, mod=mod)

    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.Adder(num_x_wires, mod)

        if mod is None:
            mod = 2**num_x_wires
        expected = qre.CompressedResourceOp(
            qre.Adder, num_x_wires, params={"num_x_wires": num_x_wires, "mod": mod}
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.Adder(num_x_wires=5, mod=2),
                [
                    qre.Allocate(2),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(num_wires=5 + 1),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=(
                                    (qre.PhaseShift.resource_rep(), 5 + 1),
                                    (qre.PhaseShift.resource_rep(), 5 + 1),
                                    (
                                        qre.ChangeOpBasis.resource_rep(
                                            cmpr_compute_op=qre.QFT.resource_rep(5 + 1),
                                            cmpr_target_op=qre.CNOT.resource_rep(),
                                            num_wires=5 + 2,
                                        ),
                                        1,
                                    ),
                                    (
                                        qre.Controlled.resource_rep(
                                            base_cmpr_op=qre.Prod.resource_rep(
                                                ((qre.PhaseShift.resource_rep(), 5 + 1),),
                                                5 + 1,
                                            ),
                                            num_ctrl_wires=1,
                                            num_zero_ctrl=0,
                                        ),
                                        1,
                                    ),
                                    (
                                        qre.ChangeOpBasis.resource_rep(
                                            cmpr_compute_op=qre.Adjoint.resource_rep(
                                                qre.Prod.resource_rep(
                                                    (
                                                        (qre.QFT.resource_rep(5 + 1), 1),
                                                        (qre.PhaseShift.resource_rep(), 5 + 1),
                                                    ),
                                                    num_wires=5 + 1,
                                                )
                                            ),
                                            cmpr_target_op=qre.Controlled.resource_rep(
                                                qre.X.resource_rep(), 1, 1
                                            ),
                                            num_wires=5 + 2,
                                        ),
                                        1,
                                    ),
                                ),
                                num_wires=5 + 2,
                            ),
                            num_wires=5 + 2,
                        )
                    ),
                    qre.Deallocate(2),
                ],
            ),
            (
                qre.Adder(num_x_wires=7, mod=31),
                [
                    qre.Allocate(2),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(num_wires=7 + 1),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=(
                                    (qre.PhaseShift.resource_rep(), 7 + 1),
                                    (qre.PhaseShift.resource_rep(), 7 + 1),
                                    (
                                        qre.ChangeOpBasis.resource_rep(
                                            cmpr_compute_op=qre.QFT.resource_rep(7 + 1),
                                            cmpr_target_op=qre.CNOT.resource_rep(),
                                            num_wires=7 + 2,
                                        ),
                                        1,
                                    ),
                                    (
                                        qre.Controlled.resource_rep(
                                            base_cmpr_op=qre.Prod.resource_rep(
                                                ((qre.PhaseShift.resource_rep(), 7 + 1),),
                                                7 + 1,
                                            ),
                                            num_ctrl_wires=1,
                                            num_zero_ctrl=0,
                                        ),
                                        1,
                                    ),
                                    (
                                        qre.ChangeOpBasis.resource_rep(
                                            cmpr_compute_op=qre.Adjoint.resource_rep(
                                                qre.Prod.resource_rep(
                                                    (
                                                        (qre.QFT.resource_rep(7 + 1), 1),
                                                        (qre.PhaseShift.resource_rep(), 7 + 1),
                                                    ),
                                                    num_wires=7 + 1,
                                                )
                                            ),
                                            cmpr_target_op=qre.Controlled.resource_rep(
                                                qre.X.resource_rep(), 1, 1
                                            ),
                                            num_wires=7 + 2,
                                        ),
                                        1,
                                    ),
                                ),
                                num_wires=7 + 2,
                            ),
                            num_wires=7 + 2,
                        )
                    ),
                    qre.Deallocate(2),
                ],
            ),
            (
                qre.Adder(num_x_wires=5, mod=32),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(5),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 5),),
                                5,
                            ),
                            num_wires=5,
                        )
                    ),
                ],
            ),
            (
                qre.Adder(num_x_wires=5, mod=None),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(5),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 5),),
                                5,
                            ),
                            num_wires=5,
                        )
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert_decomp_equal(op.resource_decomp(**op.resource_params), expected_decomp)

    @pytest.mark.parametrize("z", (1, 2, 3, 5))
    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_pow_resources(self, num_x_wires, mod, z):
        """Test the pow_resource_decomp works as expected."""
        op = qre.Adder(num_x_wires, mod)
        cmpr_op = op.resource_rep_from_op()
        assert op.pow_resource_decomp(z, op.resource_params) == [GateCount(cmpr_op)]


class TestResourceOutAdder:
    """Test the OutAdder resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, num_y_wires, num_output_wires, mod, wires",
        (
            (3, 4, 5, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            (2, 2, 5, 2, ["a", "b", "c", 1, 2, 3, "d", "e", "f"]),
            (5, 4, 6, 16, None),
            (2, 2, 5, 2**5, None),
            (1, 3, 5, None, None),
        ),
    )
    def test_init(self, num_x_wires, num_y_wires, num_output_wires, mod, wires):
        """Test the init method of the class."""
        adder = qre.OutAdder(num_x_wires, num_y_wires, num_output_wires, mod, wires)
        resource_params = adder.resource_params

        if mod is None:
            mod = 2**num_output_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert resource_params["num_y_wires"] == num_y_wires
        assert resource_params["num_output_wires"] == num_output_wires
        assert adder.wires == (Wires(wires) if wires is not None else None)
        assert adder.num_wires == num_x_wires + num_y_wires + num_output_wires

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.OutAdder(2, 2, 3, mod=mod)

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 10 wires,"):
            qre.OutAdder(num_x_wires=3, num_y_wires=3, num_output_wires=4, mod=3, wires=[0, 1])

    @pytest.mark.parametrize(
        "num_x_wires, num_y_wires, num_output_wires, mod",
        (
            (3, 4, 5, 1),
            (2, 2, 5, 2),
            (5, 4, 6, 16),
            (2, 2, 5, 2**5),
            (1, 3, 5, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, num_y_wires, num_output_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.OutAdder(num_x_wires, num_y_wires, num_output_wires, mod)

        if mod is None:
            mod = 2**num_output_wires
        expected = qre.CompressedResourceOp(
            qre.OutAdder,
            num_x_wires + num_y_wires + num_output_wires,
            params={
                "num_x_wires": num_x_wires,
                "num_y_wires": num_y_wires,
                "num_output_wires": num_output_wires,
                "mod": mod,
            },
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.OutAdder(num_x_wires=2, num_y_wires=2, num_output_wires=5, mod=3),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(6),
                            cmpr_target_op=qre.Prod.resource_rep(
                                (
                                    (
                                        qre.Controlled.resource_rep(
                                            qre.PhaseAdder.resource_rep(5, 3),
                                            1,
                                            0,
                                        ),
                                        4,
                                    ),
                                ),
                                num_wires=9,
                            ),
                        )
                    ),
                ],
            ),
            (
                qre.OutAdder(num_x_wires=3, num_y_wires=4, num_output_wires=5, mod=31),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(6),
                            cmpr_target_op=qre.Prod.resource_rep(
                                (
                                    (
                                        qre.Controlled.resource_rep(
                                            qre.PhaseAdder.resource_rep(5, 31),
                                            1,
                                            0,
                                        ),
                                        7,
                                    ),
                                ),
                                num_wires=12,
                            ),
                        )
                    ),
                ],
            ),
            (
                qre.OutAdder(num_x_wires=5, num_y_wires=5, num_output_wires=5, mod=32),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(5),
                            cmpr_target_op=qre.Prod.resource_rep(
                                (
                                    (
                                        qre.Controlled.resource_rep(
                                            qre.PhaseAdder.resource_rep(5, 32),
                                            1,
                                            0,
                                        ),
                                        10,
                                    ),
                                ),
                                num_wires=15,
                            ),
                        )
                    ),
                ],
            ),
            (
                qre.OutAdder(num_x_wires=5, num_y_wires=5, num_output_wires=5, mod=None),
                [
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QFT.resource_rep(5),
                            cmpr_target_op=qre.Prod.resource_rep(
                                (
                                    (
                                        qre.Controlled.resource_rep(
                                            qre.PhaseAdder.resource_rep(5, 32),
                                            1,
                                            0,
                                        ),
                                        10,
                                    ),
                                ),
                                num_wires=15,
                            ),
                        )
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert op.resource_decomp(**op.resource_params) == expected_decomp


class TestResourceClassicalOutMultiplier:
    """Test the ClassicalOutMultiplier resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, mod, wires",
        (
            (3, 5, 1, [1, 2, 3, 4, 5, 6, 7, 8]),
            (2, 5, 2, ["a", "b", "c", 1, 2, 3, "d"]),
            (5, 6, 16, None),
            (2, 5, 2**5, None),
            (3, 5, None, None),
        ),
    )
    def test_init(self, num_x_wires, num_output_wires, mod, wires):
        """Test the init method of the class."""
        mult = qre.ClassicalOutMultiplier(num_x_wires, num_output_wires, mod, wires)
        resource_params = mult.resource_params

        if mod is None:
            mod = 2**num_output_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert resource_params["num_output_wires"] == num_output_wires
        assert mult.wires == (Wires(wires) if wires is not None else None)
        assert mult.num_wires == num_x_wires + num_output_wires

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.ClassicalOutMultiplier(2, 3, mod=mod)

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 8 wires,"):
            qre.ClassicalOutMultiplier(num_x_wires=3, num_output_wires=5, mod=3, wires=[0, 1])

    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, mod",
        (
            (3, 5, 1),
            (2, 5, 2),
            (5, 6, 16),
            (2, 5, 2**5),
            (3, 5, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, num_output_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.ClassicalOutMultiplier(num_x_wires, num_output_wires, mod)

        if mod is None:
            mod = 2**num_output_wires
        expected = qre.CompressedResourceOp(
            qre.ClassicalOutMultiplier,
            num_x_wires + num_output_wires,
            params={
                "num_x_wires": num_x_wires,
                "num_output_wires": num_output_wires,
                "mod": mod,
            },
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.ClassicalOutMultiplier(num_x_wires=2, num_output_wires=5, mod=3),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Adder.resource_rep(5, 3),
                            1,
                            0,
                        ),
                        2,
                    ),
                ],
            ),
            (
                qre.ClassicalOutMultiplier(num_x_wires=3, num_output_wires=5, mod=31),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Adder.resource_rep(5, 31),
                            1,
                            0,
                        ),
                        3,
                    ),
                ],
            ),
            (
                qre.ClassicalOutMultiplier(num_x_wires=5, num_output_wires=5, mod=32),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Adder.resource_rep(5, 32),
                            1,
                            0,
                        ),
                        5,
                    ),
                ],
            ),
            (
                qre.ClassicalOutMultiplier(num_x_wires=5, num_output_wires=5, mod=None),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Adder.resource_rep(5, 32),
                            1,
                            0,
                        ),
                        5,
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert op.resource_decomp(**op.resource_params) == expected_decomp

    @pytest.mark.parametrize("z", (1, 2, 3, 5))
    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, mod",
        (
            (3, 5, 1),
            (2, 5, 2),
            (5, 6, 16),
            (2, 5, 2**5),
            (3, 5, None),
        ),
    )
    def test_pow_resources(self, num_x_wires, num_output_wires, mod, z):
        """Test the pow_resource_decomp works as expected."""
        op = qre.ClassicalOutMultiplier(num_x_wires, num_output_wires, mod)
        cmpr_op = op.resource_rep_from_op()
        assert op.pow_resource_decomp(z, op.resource_params) == [GateCount(cmpr_op)]


class TestResourceMultiplier:
    """Test the Multiplier resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, mod, wires",
        (
            (5, 1, [1, 2, 3, 4, 5]),
            (5, 2, ["a", "b", "c", 1, 2]),
            (5, 5, None),
            (5, 2**5, None),
            (5, None, None),
        ),
    )
    def test_init(self, num_x_wires, mod, wires):
        """Test the init method of the class."""
        adder = qre.Multiplier(num_x_wires, mod, wires)
        resource_params = adder.resource_params

        if mod is None:
            mod = 2**num_x_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert adder.wires == (Wires(wires) if wires is not None else None)
        assert adder.num_wires == num_x_wires

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.Multiplier(num_x_wires=3, mod=mod)

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 3 wires,"):
            qre.Multiplier(num_x_wires=3, mod=3, wires=[0, 1])

    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.Multiplier(num_x_wires, mod)

        if mod is None:
            mod = 2**num_x_wires
        expected = qre.CompressedResourceOp(
            qre.Multiplier, num_x_wires, params={"num_x_wires": num_x_wires, "mod": mod}
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.Multiplier(num_x_wires=5, mod=2),
                [
                    qre.Allocate(5),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.ClassicalOutMultiplier.resource_rep(
                                5,
                                5,
                                2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=((qre.SWAP.resource_rep(), 5),),
                                num_wires=10,
                            ),
                            num_wires=10,
                        )
                    ),
                    qre.Deallocate(5),
                ],
            ),
            (
                qre.Multiplier(num_x_wires=7, mod=31),
                [
                    qre.Allocate(7),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.ClassicalOutMultiplier.resource_rep(
                                7,
                                7,
                                31,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=((qre.SWAP.resource_rep(), 7),),
                                num_wires=14,
                            ),
                            num_wires=14,
                        )
                    ),
                    qre.Deallocate(7),
                ],
            ),
            (
                qre.Multiplier(num_x_wires=5, mod=32),
                [
                    qre.Allocate(5),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.ClassicalOutMultiplier.resource_rep(
                                5,
                                5,
                                32,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=((qre.SWAP.resource_rep(), 5),),
                                num_wires=10,
                            ),
                            num_wires=10,
                        )
                    ),
                    qre.Deallocate(5),
                ],
            ),
            (
                qre.Multiplier(num_x_wires=5, mod=None),
                [
                    qre.Allocate(5),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.ClassicalOutMultiplier.resource_rep(
                                5,
                                5,
                                32,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                cmpr_factors_and_counts=((qre.SWAP.resource_rep(), 5),),
                                num_wires=10,
                            ),
                            num_wires=10,
                        )
                    ),
                    qre.Deallocate(5),
                ],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert_decomp_equal(op.resource_decomp(**op.resource_params), expected_decomp)

    @pytest.mark.parametrize("z", (1, 2, 3, 5))
    @pytest.mark.parametrize(
        "num_x_wires, mod",
        (
            (2, 1),
            (7, 2),
            (8, 5),
            (5, 2**5),
            (3, None),
        ),
    )
    def test_pow_resources(self, num_x_wires, mod, z):
        """Test the pow_resource_decomp works as expected."""
        op = qre.Multiplier(num_x_wires, mod)
        cmpr_op = op.resource_rep_from_op()
        assert op.pow_resource_decomp(z, op.resource_params) == [GateCount(cmpr_op)]


class TestResourceModExp:
    """Test the ModExp resource operator."""

    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, mod, wires",
        (
            (3, 5, 1, [1, 2, 3, 4, 5, 6, 7, 8]),
            (2, 5, 2, ["a", "b", "c", 1, 2, 3, "d"]),
            (5, 6, 16, None),
            (2, 5, 2**5, None),
            (3, 5, None, None),
        ),
    )
    def test_init(self, num_x_wires, num_output_wires, mod, wires):
        """Test the init method of the class."""
        exp = qre.ModExp(num_x_wires, num_output_wires, mod, wires)
        resource_params = exp.resource_params

        if mod is None:
            mod = 2**num_output_wires

        assert resource_params["mod"] == mod
        assert resource_params["num_x_wires"] == num_x_wires
        assert resource_params["num_output_wires"] == num_output_wires
        assert exp.wires == (Wires(wires) if wires is not None else None)
        assert exp.num_wires == num_x_wires + num_output_wires

    @pytest.mark.parametrize("mod", (0, -3, 9, 100))
    def test_init_raises_error(self, mod):
        """Test that an error is raised if number of wires is
        not compatible with the mod value."""
        with pytest.raises(
            ValueError, match=re.escape(f"mod must take values inbetween (1, 8), got {mod}")
        ):
            qre.ModExp(2, 3, mod=mod)

    def test_wire_error(self):
        """Test that an error is raised if the wires provided does not match
        the number of wires expected."""
        with pytest.raises(ValueError, match="Expected 8 wires,"):
            qre.ModExp(num_x_wires=3, num_output_wires=5, mod=3, wires=[0, 1])

    @pytest.mark.parametrize(
        "num_x_wires, num_output_wires, mod",
        (
            (3, 5, 1),
            (2, 5, 2),
            (5, 6, 16),
            (2, 5, 2**5),
            (3, 5, None),
        ),
    )
    def test_resource_rep(self, num_x_wires, num_output_wires, mod):
        """Test the resource_rep method of the class."""
        op = qre.ModExp(num_x_wires, num_output_wires, mod)

        if mod is None:
            mod = 2**num_output_wires
        expected = qre.CompressedResourceOp(
            qre.ModExp,
            num_x_wires + num_output_wires,
            params={
                "num_x_wires": num_x_wires,
                "num_output_wires": num_output_wires,
                "mod": mod,
            },
        )

        assert op.resource_rep_from_op() == expected

    @pytest.mark.parametrize(
        "op, expected_decomp",
        (
            (
                qre.ModExp(num_x_wires=2, num_output_wires=5, mod=3),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Multiplier.resource_rep(5, 3),
                            1,
                            0,
                        ),
                        2,
                    ),
                ],
            ),
            (
                qre.ModExp(num_x_wires=3, num_output_wires=5, mod=31),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Multiplier.resource_rep(5, 31),
                            1,
                            0,
                        ),
                        3,
                    ),
                ],
            ),
            (
                qre.ModExp(num_x_wires=5, num_output_wires=5, mod=32),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Multiplier.resource_rep(5, 32),
                            1,
                            0,
                        ),
                        5,
                    ),
                ],
            ),
            (
                qre.ModExp(num_x_wires=5, num_output_wires=5, mod=None),
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Multiplier.resource_rep(5, 32),
                            1,
                            0,
                        ),
                        5,
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, op, expected_decomp):
        """Test that the resource_decomp produces the expected resources."""
        assert op.resource_decomp(**op.resource_params) == expected_decomp
