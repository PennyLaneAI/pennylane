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
"""Unit tests for the cond_measure function"""

from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.ftqc import (
    ParametricMidMeasureMP,
    XMidMeasureMP,
    YMidMeasureMP,
    cond_measure,
    diagonalize_mcms,
    measure_arbitrary_basis,
    measure_x,
    measure_y,
    measure_z,
)
from pennylane.measurements import MeasurementValue


class TestCondMeas:
    """Test the behaviour of cond_measure for expected inputs"""

    @pytest.mark.parametrize("wire", (2, "c"))
    @pytest.mark.parametrize("reset", (True, False))
    @pytest.mark.parametrize("postselect", (None, 0, 1))
    def test_cond_measure_with_measurements(self, wire, reset, postselect):
        """Test that passing a MeasurementValue and measurement functions
        to cond_measure creates the expected measurements and MeasurementValue"""

        with qml.queuing.AnnotatedQueue() as q:
            m = qml.measure(0)
            m2 = cond_measure(m, measure_x, measure_y)(
                wires=wire, reset=reset, postselect=postselect
            )

        ops = qml.tape.QuantumScript.from_queue(q).operations

        assert len(ops) == 3
        conditional_mps = ops[1:]

        # the new measurements match the expected properties
        for meas in conditional_mps:
            assert isinstance(meas, qml.ops.Conditional)
            assert meas.wires == qml.wires.Wires([wire])
            assert meas.base.reset == reset
            assert meas.base.postselect == postselect

        # bases are correct
        assert isinstance(conditional_mps[0].base, XMidMeasureMP)
        assert isinstance(conditional_mps[1].base, YMidMeasureMP)

        # they have opposite conditions
        fn_x, fn_y = (m.meas_val.processing_fn for m in conditional_mps)
        assert fn_x(0) == fn_y(1)
        assert fn_x(1) == fn_y(0)

        # m2 is a MeasurementValue with correct measurements and processing_fn
        assert isinstance(m2, MeasurementValue)
        assert m2.measurements == [op.base for op in conditional_mps]
        fn = m2.processing_fn
        assert bool(fn(1, 1)) is True
        assert bool(fn(0, 0)) is False
        assert fn(1, 1) == fn(1, 0) == fn(0, 1)

    def test_cond_measure_with_partial(self):
        """Test that passing a MeasurementValue and partials of measurement functions
        executes successfully and creates the expected operator types"""

        with qml.queuing.AnnotatedQueue() as q:
            m = qml.measure(0)
            cond_measure(
                m,
                partial(measure_arbitrary_basis, angle=1.2, plane="ZX"),
                partial(measure_arbitrary_basis, angle=2.4, plane="XY"),
            )(2)

        ops = qml.tape.QuantumScript.from_queue(q).operations

        assert len(ops) == 3

        # expected measurements were created
        for meas in ops[1:]:
            assert isinstance(meas, qml.ops.Conditional)
            assert isinstance(meas.base, ParametricMidMeasureMP)
        assert ops[1].base.angle == 1.2
        assert ops[1].base.plane == "ZX"
        assert ops[2].base.angle == 2.4
        assert ops[2].base.plane == "XY"

    @pytest.mark.parametrize("val, meas_type", [(1, XMidMeasureMP), (0, YMidMeasureMP)])
    def test_condition_is_not_mcm(self, val, meas_type):
        """Test that passing a boolean rather than a MeasurementValue
        simplifies to applying the appropriate measurement"""

        with qml.queuing.AnnotatedQueue() as q:
            m = cond_measure(val, measure_x, measure_y)(0)

        ops = qml.tape.QuantumScript.from_queue(q).operations

        assert len(ops) == 1
        assert isinstance(ops[0], meas_type)

        assert len(m.measurements) == 1
        assert m.measurements[0] == ops[0]


class TestValidation:
    """Test the errors raised by validation in cond_measure"""

    @pytest.mark.parametrize("inp", [1, "string", qml.PauliZ(0)])
    def test_non_callable_raises_error(self, inp):
        """Test that an error is raised when the input is not a callable."""

        with pytest.raises(ValueError, match="Only measurement functions can be applied"):
            m = qml.measure(0)
            cond_measure(m, inp, measure_x)(0)

        with pytest.raises(ValueError, match="Only measurement functions can be applied"):
            m = qml.measure(0)
            cond_measure(m, measure_x, inp)(0)

    @pytest.mark.parametrize("inp", [qml.X, XMidMeasureMP])
    def test_incorrect_callable_raises_error(self, inp):
        """Test that an error is raised when the callable does not return a MeasurementValue"""

        with pytest.raises(
            ValueError,
            match="Only measurement functions that return a measurement value can be used",
        ):
            m = qml.measure(0)
            cond_measure(m, inp, measure_x)(0)

        with pytest.raises(
            ValueError,
            match="Only measurement functions that return a measurement value can be used",
        ):
            m = qml.measure(0)
            cond_measure(m, measure_x, inp)(0)

    @pytest.mark.parametrize("attribute, inp", [("reset", (True, False)), ("postselect", (0, 1))])
    def test_mismatched_settings_raises_error(self, attribute, inp):
        """Test that a mismatch between the measurement settings `reset` and `postselect`
        on the two measurements raises an error"""

        input1 = {attribute: inp[0]}
        input2 = {attribute: inp[1]}

        with pytest.raises(
            ValueError,
            match="behaviour must be consistent for both branches",
        ):
            m = qml.measure(0)
            cond_measure(m, partial(measure_y, **input1), partial(measure_x, **input2))(0)

    def test_mismatched_wires_raises_error(self):

        with pytest.raises(
            ValueError,
            match="behaviour must be consistent for both branches",
        ):
            m = qml.measure(0)
            cond_measure(m, partial(measure_y, wires=0), partial(measure_x, wires=1))()

    @pytest.mark.capture
    def test_program_capture(self):
        """Test that program capture works as expected with cond_measure"""
        import jax

        def func():
            m = qml.measure(0)
            cond_measure(m, measure_x, measure_y)(0)

        plxpr = jax.make_jaxpr(func)()

        cond_eq = plxpr.eqns[1]
        assert "cond" in str(cond_eq)
        cond_branches = cond_eq.params["jaxpr_branches"]
        assert len(cond_branches) == 2
        for branch, angle in zip(cond_branches, [0, 1.57]):
            branch_str = str(branch)
            assert "measure_in_basis" in branch_str
            assert "plane=XY" in branch_str
            assert str(angle) in branch_str


class TestWorkflows:

    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_execution_in_cond(self, mcm_method, shots):
        """Test that we can execute a QNode with a ParametricMidMeasureMP applied in a conditional,
        and produce an accurate result"""

        dev = qml.device("default.qubit")

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            qml.RX(np.pi, 0)
            m = qml.measure(0)  # always 1

            qml.RX(2.345, 1)
            cond_measure(m == 0, measure_x, measure_y)(1)  # always measure_y
            return qml.expval(qml.Z(1))

        if shots:
            # the result is on the order of 1 (-0.7), and an uncertainty ~1.5-2 orders of magnitude
            # smaller than the result is sufficiently accurate for a shots-based measurement
            assert np.isclose(diagonalize_mcms(circ)(), -np.sin(2.345), atol=0.03)
        else:
            assert np.isclose(diagonalize_mcms(circ)(), -np.sin(2.345))

        # without the transform, the mid-circuit measurements are all treated as computational
        # basis measurements, and they are inside Conditional, which doesn't execute correctly,
        # so we return incorrect results, even with a high atol (± ~20-30% of expected outcome)
        assert not np.isclose(circ(), -np.sin(2.345), atol=0.2)

    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_cascading_conditional_measurements(self, mcm_method, shots):
        """Test a workflow that feeds measurement values from conditional measurements forward
        into subsequent measurements and operations applied in `cond_measure` and `cond`"""

        dev = qml.device("default.qubit")

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ(x_rot, y_rot):
            qml.RX(np.pi, 0)
            m = qml.measure(0)  # always 1

            qml.RX(np.pi / 2, 1)
            m2 = cond_measure(m == 0, measure_x, measure_y)(1)  # always measure_y, always 1

            qml.RY(y_rot, 2)
            qml.RX(x_rot, 2)
            cond_measure(m2, measure_z, measure_y)(2)

            qml.cond(m2, qml.X)(3)

            return qml.expval(qml.Z(2)), qml.expval(qml.Z(3))

        (x, y) = 1.23, 3.45

        if shots:
            # the result is on the order of 1 (-0.7), and an uncertainty ~1.5-2 orders of magnitude
            # smaller than the result is sufficiently accurate for a shots-based measurement
            assert np.allclose(diagonalize_mcms(circ)(x, y), [np.cos(x) * np.cos(y), -1], atol=0.03)
        else:
            assert np.allclose(diagonalize_mcms(circ)(x, y), [np.cos(x) * np.cos(y), -1])

        # this can't be executed without diagonalize_mcms, because without the transform, it
        # tries to get concrete values for measurements that weren't executed when it hits
        # the conditional that depends on m2, and can't find it in the measurements dictionary
        with pytest.raises(KeyError):
            circ(x, y)
