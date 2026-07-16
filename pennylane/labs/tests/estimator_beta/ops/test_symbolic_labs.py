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
"""Tests for symbolic resource operators in the estimator_beta module."""

from collections import defaultdict
from functools import partial

import pytest

import pennylane as qp
import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta import CompressedResourceOp, GateCount
from pennylane.labs.estimator_beta.ops.op_math.symbolic import (
    ResourceQfunc,
    _generate_name,
    mark_subroutine,
)

# pylint: disable=no-self-use, too-few-public-methods


class TestGenerateName:
    """Tests for the ``_generate_name`` helper function."""

    @pytest.mark.parametrize(
        "args, kwargs",
        (
            ((10, True, "b"), {}),
            ((10, True), {"kwarg1": "b"}),
            ((10,), {"arg2": True, "kwarg1": "b"}),
            (tuple(), {"arg1": 10, "arg2": True, "kwarg1": "b"}),
        ),
    )
    def test_no_include_params(self, args, kwargs):
        """Test that the bare function name is returned when no params are included."""

        def my_func(arg1, arg2, kwarg1="a"):  # pylint: disable=unused-argument
            return

        assert _generate_name(my_func, None, *args, **kwargs) == "my_func"

    @pytest.mark.parametrize(
        "args, kwargs",
        (
            ((10, True, "b"), {}),
            ((10, True), {"kwarg1": "b"}),
            ((10,), {"arg2": True, "kwarg1": "b"}),
            (tuple(), {"arg1": 10, "arg2": True, "kwarg1": "b"}),
        ),
    )
    def test_include_params_positional_and_keyword(self, args, kwargs):
        """Test that selected positional and keyword params are formatted into the name."""

        def my_func(arg1, arg2, kwarg1="a"):  # pylint: disable=unused-argument
            return

        name = _generate_name(
            my_func,
            ["arg1", "kwarg1"],
            *args,
            **kwargs,
        )
        assert name == "my_func(arg1=10, kwarg1=b)"


class TestMarkSubroutine:
    """Tests for the ``mark_subroutine`` decorator."""

    def test_returns_resource_qfunc(self):
        """Test that the decorated function returns a ResourceQfunc instance."""

        @mark_subroutine
        def SubroutineA(num_iter):
            for _ in range(num_iter):
                qre.Z()

        op = SubroutineA(3)
        assert isinstance(op, ResourceQfunc)
        assert op.name == "SubroutineA"

    def test_set_num_wires(self):
        """Test that the decorated function returns a ResourceQfunc instance with set num_wires."""

        @partial(mark_subroutine, num_wires=11)
        def SubroutineA(num_iter):
            for _ in range(num_iter):
                qre.Z()

        op = SubroutineA(3)
        assert isinstance(op, ResourceQfunc)
        assert op.name == "SubroutineA"
        assert op.num_wires == 11

    def test_include_params_naming(self):
        """Test that the include_params keyword configures the tracking name."""

        @partial(mark_subroutine, include_params=["num_iter"])
        def SubroutineA(num_iter, op_type="Z"):  # pylint: disable=unused-argument
            for _ in range(num_iter):
                qre.Z()

        op = SubroutineA(5)
        assert isinstance(op, ResourceQfunc)
        assert op.name == "SubroutineA(num_iter=5)"

    def test_estimate_tracks_subroutine(self):
        """Test that the counts of the subroutine are tracked at the abstraction level."""

        @mark_subroutine
        def SubroutineA(num_iter):
            for _ in range(num_iter):
                qre.Z()

        def circuit():
            SubroutineA(5)
            qre.CNOT()
            SubroutineA(3)

        gate_set = {"CNOT", "SubroutineA"}
        res = qre.estimate(circuit, gate_set)()

        expected = qre.Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=2,
            gate_types=defaultdict(
                int,
                {
                    ResourceQfunc.resource_rep(
                        "SubroutineA", 1, tuple(qre.Z.resource_rep() for _ in range(5))
                    ): 1,
                    ResourceQfunc.resource_rep(
                        "SubroutineA", 1, tuple(qre.Z.resource_rep() for _ in range(3))
                    ): 1,
                    qre.CNOT.resource_rep(): 1,
                },
            ),
        )
        assert res == expected


class TestResourceQfunc:
    """Tests for the ResourceQfunc resource operator."""

    def test_init_raises_error_no_ops(self):
        """Test that an error is raised if there were no operators queued in the qfunc"""

        def qfunc():
            return

        with pytest.raises(ValueError, match="No operators were found"):
            _ = ResourceQfunc("EmptyFunc", qfunc)

    def test_init(self):
        """Test that the operator is instantiated correctly."""

        def qfunc(num_iter):
            for _ in range(num_iter):
                qre.Z()

        op = ResourceQfunc("SubA", qfunc, 3)
        assert op.name == "SubA"
        assert op.num_wires == 1
        assert op.wires is None
        assert op.cmpr_ops == tuple(qre.Z.resource_rep() for _ in range(3))

    def test_init_infers_wires_from_factors(self):
        """Test that the wires are inferred from the queued factors when provided."""

        def qfunc():
            qre.CNOT(wires=[0, 1])
            qre.X(wires=[2])

        op = ResourceQfunc("SubB", qfunc)
        assert op.name == "SubB"
        assert op.num_wires == 3
        assert op.wires == qp.wires.Wires([0, 1, 2])

    def test_init_set_wires_from_factors(self):
        """Test that the num_wires can be provided."""

        def qfunc():
            qre.CNOT(wires=[0, 1])
            qre.X(wires=[2])

        op = ResourceQfunc("SubB", qfunc, num_wires_=11)
        assert op.name == "SubB"
        assert op.num_wires == 11
        assert op.wires is None

    def test_init_maps_plain_operator(self):
        """Test that plain PennyLane operators are mapped to resource operators."""

        def qfunc():
            qp.Hadamard(0)
            qre.X(wires=1)

        op = ResourceQfunc("qfunc", qfunc)
        assert op.cmpr_ops == (qre.Hadamard.resource_rep(), qre.X.resource_rep())

    def test_marking_qubits_raises(self):
        """Test that queuing a MarkQubits instance raises a TypeError."""

        def qfunc():
            qre.X(wires=0)
            qre.MarkClean(wires=0)

        with pytest.raises(TypeError, match="Marking qubits is currently not supported"):
            ResourceQfunc("E", qfunc)

    def test_resource_params(self):
        """Test that the resource_params are returned correctly."""

        def qfunc():
            qre.Z()
            qre.Z()

        op = ResourceQfunc("SubA", qfunc)
        assert op.resource_params == {
            "name": "SubA",
            "num_wires": 1,
            "cmpr_ops": (qre.Z.resource_rep(), qre.Z.resource_rep()),
        }

    def test_resource_rep(self):
        """Test that the compressed representation is built correctly."""
        cmpr_ops = (qre.Z.resource_rep(), qre.Z.resource_rep())
        expected = CompressedResourceOp(
            ResourceQfunc,
            1,
            {"name": "SubA", "num_wires": 1, "cmpr_ops": cmpr_ops},
            name="SubA",
        )
        assert ResourceQfunc.resource_rep("SubA", 1, cmpr_ops) == expected

    def test_resource_decomp(self):
        """Test that we can obtain the resources as expected."""

        def qfunc():
            qre.Z()
            qre.X()

        op = ResourceQfunc("SubA", qfunc)
        expected = [
            GateCount(qre.Z.resource_rep()),
            GateCount(qre.X.resource_rep()),
        ]
        assert op.resource_decomp(**op.resource_params) == expected

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        cmpr_ops = (qre.Z.resource_rep(),)
        assert ResourceQfunc.tracking_name("SubA", 1, cmpr_ops) == "SubA"
        assert ResourceQfunc.tracking_name("SubroutineB", 2, cmpr_ops) == "SubroutineB"

    def test_estimate_full_decomposition(self):
        """Test that estimate expands the subroutine into its factors by default."""

        def qfunc():
            qre.Z()
            qre.Z()
            qre.X()

        op = ResourceQfunc("SubA", qfunc)
        expected = qre.Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=1,
            gate_types=defaultdict(
                int,
                {
                    qre.Z.resource_rep(): 2,
                    qre.X.resource_rep(): 1,
                },
            ),
        )
        assert qre.estimate(op) == expected
