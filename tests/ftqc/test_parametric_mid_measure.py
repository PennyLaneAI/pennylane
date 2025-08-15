# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for parametric mid-circuit measurements"""

from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit import measure as apply_qubit_measurement
from pennylane.exceptions import QuantumFunctionError
from pennylane.ftqc import (
    ParametricMidMeasureMP,
    XMidMeasureMP,
    YMidMeasureMP,
    cond_measure,
    diagonalize_mcms,
    measure_arbitrary_basis,
    measure_x,
    measure_y,
)
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.wires import Wires


class TestParametricMidMeasure:
    """Tests for the parametric mid-circuit measurement class in an arbitrary basis"""

    def test_hash(self):
        """Test that the hash for `ParametricMidMeasureMP` is defined correctly."""
        m1 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")
        m2 = ParametricMidMeasureMP(Wires(1), angle=1.23, id="m1", plane="XY")
        m3 = ParametricMidMeasureMP(Wires(0), angle=2.45, id="m1", plane="XY")
        m4 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m2", plane="XY")
        m5 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="YZ")
        m6 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="ZX")
        m7 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash != m4.hash
        assert m1.hash != m5.hash
        assert m1.hash != m6.hash
        assert m1.hash == m7.hash

    def test_flatten_unflatten(self):
        """Test that we can flatten and unflatten the ParametricMidMeasureMP"""

        op = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")
        data, metadata = op._flatten()  # pylint: disable = protected-access

        assert hash(metadata)  # metadata must be hashable

        unflattened_op = ParametricMidMeasureMP._unflatten(  # pylint: disable = protected-access
            data, metadata
        )
        assert op.hash == unflattened_op.hash

    @pytest.mark.jax
    def test_flatten_unflatten_jax(self):
        """Test that jax.tree_util can flatten and unflatten the ParametricMidMeasureMP"""

        import jax

        op = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")

        leaves, struct = jax.tree_util.tree_flatten(op)
        unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)

        assert op.hash == unflattened_op.hash

    @pytest.mark.parametrize(
        "plane, angle, wire, expected",
        [
            ("XY", 1.2, 0, "measure_xy(wires=[0], angle=1.2)"),
            ("ZX", 1.2, 0, "measure_zx(wires=[0], angle=1.2)"),
            ("YZ", 1.2, 0, "measure_yz(wires=[0], angle=1.2)"),
            ("XY", np.pi, 0, "measure_xy(wires=[0], angle=3.141592653589793)"),
            ("XY", 2.345, 1, "measure_xy(wires=[1], angle=2.345)"),
        ],
    )
    def test_repr(self, plane, angle, wire, expected):
        """Test the repr for ParametricMidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(wires=Wires([wire]), angle=angle, plane=plane)
        assert repr(mp) == expected

    @pytest.mark.parametrize(
        "plane, postselect, reset, expected",
        [
            ("XY", None, False, "┤↗ˣʸ├"),
            ("XY", None, True, "┤↗ˣʸ│  │0⟩"),
            ("XY", 0, False, "┤↗ˣʸ₀├"),
            ("XY", 0, True, "┤↗ˣʸ₀│  │0⟩"),
            ("XY", 1, False, "┤↗ˣʸ₁├"),
            ("XY", 1, True, "┤↗ˣʸ₁│  │0⟩"),
            ("ZX", None, False, "┤↗ᶻˣ├"),
            ("ZX", None, True, "┤↗ᶻˣ│  │0⟩"),
            ("ZX", 0, False, "┤↗ᶻˣ₀├"),
            ("ZX", 0, True, "┤↗ᶻˣ₀│  │0⟩"),
            ("ZX", 1, False, "┤↗ᶻˣ₁├"),
            ("ZX", 1, True, "┤↗ᶻˣ₁│  │0⟩"),
            ("YZ", None, False, "┤↗ʸᶻ├"),
            ("YZ", None, True, "┤↗ʸᶻ│  │0⟩"),
            ("YZ", 0, False, "┤↗ʸᶻ₀├"),
            ("YZ", 0, True, "┤↗ʸᶻ₀│  │0⟩"),
            ("YZ", 1, False, "┤↗ʸᶻ₁├"),
            ("YZ", 1, True, "┤↗ʸᶻ₁│  │0⟩"),
        ],
    )
    def test_label_no_decimals(self, plane, postselect, reset, expected):
        """Test that the label for a ParametricMidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(
            Wires([0]), angle=1.23, postselect=postselect, reset=reset, plane=plane
        )

        label = mp.label()
        assert label == expected

    @pytest.mark.parametrize(
        "decimals, postselect, expected",
        [
            (2, None, "┤↗ˣʸ(0.79)├"),
            (2, 0, "┤↗ˣʸ(0.79)₀├"),
            (4, None, "┤↗ˣʸ(0.7854)├"),
            (4, 0, "┤↗ˣʸ(0.7854)₀├"),
        ],
    )
    def test_label_with_decimals(self, decimals, postselect, expected):
        """Test that the label for a ParametricMidMeasureMP is correct when
        decimals are specified in the label function"""
        mp = ParametricMidMeasureMP(Wires([0]), angle=np.pi / 4, postselect=postselect, plane="XY")

        label = mp.label(decimals=decimals)
        assert label == expected

    @pytest.mark.parametrize(
        "plane, measurement_angle, corresponding_obs",
        [
            # XY measurements
            ("XY", 0, qml.X(0)),
            ("XY", np.pi / 4, qml.X(0) + qml.Y(0)),
            ("XY", np.pi / 2, qml.Y(0)),
            ("XY", 3 * np.pi / 4, qml.Y(0) - qml.X(0)),
            ("XY", np.pi, -qml.X(0)),
            ("XY", 5 * np.pi / 4, -qml.X(0) - qml.Y(0)),
            ("XY", -3 * np.pi / 4, -qml.X(0) - qml.Y(0)),
            # ZX measurements
            ("ZX", 0, qml.Z(0)),
            ("ZX", np.pi / 4, qml.X(0) + qml.Z(0)),
            ("ZX", np.pi / 2, qml.X(0)),
            ("ZX", 3 * np.pi / 4, qml.X(0) - qml.Z(0)),
            ("ZX", np.pi, -qml.Z(0)),
            ("ZX", 5 * np.pi / 4, -qml.X(0) - qml.Z(0)),
            ("ZX", -3 * np.pi / 4, -qml.X(0) - qml.Z(0)),
            # YZ measurements
            ("YZ", 0, qml.Z(0)),
            ("YZ", np.pi / 4, -qml.Y(0) + qml.Z(0)),
            ("YZ", np.pi / 2, -qml.Y(0)),
            ("YZ", 3 * np.pi / 4, -qml.Y(0) - qml.Z(0)),
            ("YZ", np.pi, -qml.Z(0)),
            ("YZ", 5 * np.pi / 4, qml.Y(0) - qml.Z(0)),
            ("YZ", -3 * np.pi / 4, qml.Y(0) - qml.Z(0)),
        ],
    )
    def test_diagonalizing_gates(self, plane, measurement_angle, corresponding_obs):
        """Test that diagonalizing a parametrized mid-circuit measurement and measuring
        in the computational basis corresponds to the expected observable"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ(state, angle):
            qml.StatePrep(state, wires=0)
            mp = ParametricMidMeasureMP([0], angle=angle, plane=plane)
            assert mp.has_diagonalizing_gates
            return qml.expval(qml.Z(0))

        input_state = np.random.random(2) + 1j * np.random.random(2)
        input_state = input_state / np.linalg.norm(input_state)

        res = circ(input_state, measurement_angle)

        expected_res = apply_qubit_measurement(qml.expval(corresponding_obs), input_state)
        if isinstance(corresponding_obs, qml.ops.Sum):
            expected_res = expected_res / np.sqrt(2)

        assert np.allclose(res, expected_res)

    def test_invalid_plane_raises_error(self):
        """Test that an error is raised at diagonalization if a plane other than
        XY, YZ, or ZX are passed."""

        mp = ParametricMidMeasureMP([0], angle=1.23, plane="AB")

        with pytest.raises(NotImplementedError, match="plane not implemented"):
            mp.diagonalizing_gates()


class TestMidMeasureXAndY:
    """Tests for the mid-circuit measurement class in the X and Y basis"""

    @pytest.mark.parametrize("mp_class, angle", [(XMidMeasureMP, 0), (YMidMeasureMP, np.pi / 2)])
    def test_attributes(self, mp_class, angle):
        """Test that the XMidMeasure and YMidMeasure have the expected attributes for a ParametricMidMeasureMP"""

        mp = mp_class(Wires(0))

        assert isinstance(mp, ParametricMidMeasureMP)
        assert mp.angle == angle
        assert mp.plane == "XY"

    @pytest.mark.parametrize("mp_class, angle", [(XMidMeasureMP, 0), (YMidMeasureMP, np.pi / 2)])
    def test_hash(self, mp_class, angle):
        """Test that the hash for XMidMeasureMP and YMidMeasureMP are defined correctly."""
        m1 = mp_class(Wires(0), id="m1")
        m2 = mp_class(Wires(2), id="m1")
        m3 = mp_class(Wires(0), id="m2")
        m4 = ParametricMidMeasureMP(Wires(0), angle=angle, id="m1", plane="XY")
        m5 = mp_class(Wires(0), id="m1")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash != m4.hash
        assert m1.hash == m5.hash

    @pytest.mark.parametrize("mp_class", [XMidMeasureMP, YMidMeasureMP])
    def test_flatten_unflatten(self, mp_class):
        """Test that we can flatten and unflatten the ParametricMidMeasureMP"""

        op = mp_class(Wires(0), id="m1")
        data, metadata = op._flatten()  # pylint: disable = protected-access

        assert hash(metadata)  # metadata must be hashable

        unflattened_op = mp_class._unflatten(data, metadata)  # pylint: disable = protected-access
        assert op.hash == unflattened_op.hash

    @pytest.mark.jax
    @pytest.mark.parametrize("mp_class", [XMidMeasureMP, YMidMeasureMP])
    def test_flatten_unflatten_jax(self, mp_class):
        """Test that jax.tree_util can flatten and unflatten the ParametricMidMeasureMP"""

        import jax

        op = mp_class(Wires(0), id="m1")

        leaves, struct = jax.tree_util.tree_flatten(op)
        unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)

        assert op.hash == unflattened_op.hash

    @pytest.mark.parametrize(
        "wire, expected",
        [(0, "measure_x(wires=[0])"), (1, "measure_x(wires=[1])"), ("a", "measure_x(wires=['a'])")],
    )
    def test_repr_x(self, wire, expected):
        """Test the repr for XMidMeasureMP is correct"""
        mp = XMidMeasureMP(wires=Wires([wire]))
        assert repr(mp) == expected

    @pytest.mark.parametrize(
        "wire, expected",
        [(0, "measure_y(wires=[0])"), (1, "measure_y(wires=[1])"), ("a", "measure_y(wires=['a'])")],
    )
    def test_repr_y(self, wire, expected):
        """Test the repr for YMidMeasureMP is correct"""
        mp = YMidMeasureMP(wires=Wires([wire]))
        assert repr(mp) == expected

    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, "┤↗ˣ├"),
            (None, True, "┤↗ˣ│  │0⟩"),
            (0, False, "┤↗ˣ₀├"),
            (0, True, "┤↗ˣ₀│  │0⟩"),
            (1, False, "┤↗ˣ₁├"),
            (1, True, "┤↗ˣ₁│  │0⟩"),
        ],
    )
    def test_label_x(self, postselect, reset, expected):
        """Test that the label for a XMidMeasureMP is correct"""
        mp = XMidMeasureMP(Wires([0]), postselect=postselect, reset=reset)
        label = mp.label()
        assert label == expected

    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, "┤↗ʸ├"),
            (None, True, "┤↗ʸ│  │0⟩"),
            (0, False, "┤↗ʸ₀├"),
            (0, True, "┤↗ʸ₀│  │0⟩"),
            (1, False, "┤↗ʸ₁├"),
            (1, True, "┤↗ʸ₁│  │0⟩"),
        ],
    )
    def test_label_y(self, postselect, reset, expected):
        """Test that the label for a YMidMeasureMP is correct"""
        mp = YMidMeasureMP(Wires([0]), postselect=postselect, reset=reset)
        label = mp.label()
        assert label == expected

    def test_diagonalizing_gates_x(self):
        """Test that diagonalizing a XMidMeasureMP and measuring in the computational
        basis corresponds to the expected observable"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ(state):
            qml.StatePrep(state, wires=0)
            mp = XMidMeasureMP([0])
            assert mp.has_diagonalizing_gates
            return qml.expval(qml.Z(0))

        rng = np.random.default_rng(seed=111)
        input_state = rng.random(2) + 1j * rng.random(2)
        input_state = input_state / np.linalg.norm(input_state)

        res = circ(input_state)

        expected_res = apply_qubit_measurement(qml.expval(qml.X(0)), input_state)

        assert np.allclose(res, expected_res)
        assert XMidMeasureMP([0]).diagonalizing_gates() == [qml.H(0)]

    def test_diagonalizing_gates_y(self):
        """Test that diagonalizing a YMidMeasureMP and measuring in the computational
        basis corresponds to the expected observable"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ(state):
            qml.StatePrep(state, wires=0)
            mp = YMidMeasureMP([0])
            assert mp.has_diagonalizing_gates
            return qml.expval(qml.Z(0))

        rng = np.random.default_rng(seed=111)
        input_state = rng.random(2) + 1j * rng.random(2)
        input_state = input_state / np.linalg.norm(input_state)

        res = circ(input_state)

        expected_res = apply_qubit_measurement(qml.expval(qml.Y(0)), input_state)

        assert np.allclose(res, expected_res)
        assert YMidMeasureMP([0]).diagonalizing_gates() == [qml.adjoint(qml.S(0)), qml.H(0)]


class TestMeasureFunctions:
    """Test that the measure functions (measure_arbitrary_basis, measure_x, and
    measure_y) behave as expected"""

    @pytest.mark.parametrize("wire", [0, 2])
    @pytest.mark.parametrize("angle", [1.2, -0.4])
    @pytest.mark.parametrize("plane", ["XY", "YZ", "XZ"])
    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_measure_arbitrary_basis(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, wire, angle, plane, reset, postselect
    ):
        """Test that measure_arbitrary_basis queues the expected ParametricMidMeasureMP
        and returns a linked MeasurementValue"""

        with qml.queuing.AnnotatedQueue() as q:
            m = measure_arbitrary_basis(
                wire, angle=angle, plane=plane, reset=reset, postselect=postselect
            )

        # a single op is queued
        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 1
        op = tape.operations[0]

        # that op is stored as the measurement on the returned MeasurementValue
        assert isinstance(m, MeasurementValue)
        assert m.measurements == [op]

        # the op is the expected ParametricMidMeasureMP
        assert isinstance(op, ParametricMidMeasureMP)
        assert op.angle == angle
        assert op.wires == Wires([wire])
        assert op.plane == plane
        assert op.reset == reset
        assert op.postselect == postselect

    @pytest.mark.parametrize("wire", [0, 2])
    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_measure_x(self, wire, reset, postselect):
        """Test that measure_arbitrary_basis queues the expected XMidMeasureMP
        and returns a linked MeasurementValue"""

        with qml.queuing.AnnotatedQueue() as q:
            m = measure_x(wire, reset=reset, postselect=postselect)

        # a single op is queued
        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 1
        op = tape.operations[0]

        # that op is stored as the measurement on the returned MeasurementValue
        assert isinstance(m, MeasurementValue)
        assert m.measurements == [op]

        # the op is the expected ParametricMidMeasureMP
        assert isinstance(op, XMidMeasureMP)
        assert op.wires == Wires([wire])
        assert op.reset == reset
        assert op.postselect == postselect

    @pytest.mark.parametrize("wire", [0, 2])
    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_measure_y(self, wire, reset, postselect):
        """Test that measure_arbitrary_basis queues the expected YMidMeasureMP
        and returns a linked MeasurementValue"""

        with qml.queuing.AnnotatedQueue() as q:
            m = measure_y(wire, reset=reset, postselect=postselect)

        # a single op is queued
        tape = qml.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 1
        op = tape.operations[0]

        # that op is stored as the measurement on the returned MeasurementValue
        assert isinstance(m, MeasurementValue)
        assert m.measurements == [op]

        # the op is the expected ParametricMidMeasureMP
        assert isinstance(op, YMidMeasureMP)
        assert op.wires == Wires([wire])
        assert op.reset == reset
        assert op.postselect == postselect

    @pytest.mark.parametrize(
        "func", [partial(measure_arbitrary_basis, angle=-0.8, plane="XY"), measure_x, measure_y]
    )
    def test_error_is_raised_if_too_many_wires(self, func):
        """Test that a QuanutmFunctionError is raised if too many wires are passed"""

        with pytest.raises(
            QuantumFunctionError,
            match="Only a single qubit can be measured in the middle of the circuit",
        ):
            func([0, 1])

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "func", [partial(measure_arbitrary_basis, angle=-0.8, plane="XY"), measure_x, measure_y]
    )
    def test_error_is_raised_if_too_many_wires_capture(self, func):
        """Test that a QuanutmFunctionError is raised if too many wires are passed when using capture"""

        with pytest.raises(
            QuantumFunctionError,
            match="Only a single qubit can be measured in the middle of the circuit",
        ):
            func([0, 1])

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [0, 1, None])
    def test_measure_z_dispatches_to_measure(self, reset, postselect):
        """Test that the measurement in the computational basis calls the standard
        PennyLane MCM class, and passes through the arguments"""

        m = qml.ftqc.measure_z(0, reset=reset, postselect=postselect)

        assert isinstance(m, MeasurementValue)

        assert len(m.measurements) == 1

        mp = m.measurements[0]
        assert mp.reset == reset
        assert mp.postselect == postselect
        assert isinstance(mp, MidMeasureMP)

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    @pytest.mark.capture
    @pytest.mark.parametrize(
        "meas_func, angle, plane", [(measure_x, 0.0, "XY"), (measure_y, np.pi / 2, "XY")]
    )
    @pytest.mark.parametrize(
        "wire, reset, postselect", ((2, True, None), (3, False, 0), (0, True, 1))
    )
    def test_x_and_y_with_program_capture(self, meas_func, angle, plane, wire, reset, postselect):
        """Test that the measure_ functions are captured as expected"""
        import jax

        def circ():
            m = meas_func(wire, reset=reset, postselect=postselect)
            qml.cond(m, qml.X, qml.Y)(0)
            return qml.expval(qml.Z(2))

        plxpr = jax.make_jaxpr(circ)()
        captured_measurement = str(plxpr.eqns[0])

        # measurement is captured as epxected
        assert "measure_in_basis" in captured_measurement
        assert f"plane={plane}" in captured_measurement
        assert f"postselect={postselect}" in captured_measurement
        assert f"reset={reset}" in captured_measurement

        # parameters held in invars
        assert jax.numpy.isclose(angle, plxpr.eqns[0].invars[0].val)
        assert jax.numpy.isclose(wire, plxpr.eqns[0].invars[1].val)

        # measurement value is assigned and passed forward
        conditional = str(plxpr.eqns[1])
        assert "cond" in conditional
        assert captured_measurement[:8] == "a:bool[]"
        assert "lambda ; a:i64[]" in conditional

    @pytest.mark.capture
    @pytest.mark.parametrize("angle, plane", [(1.23, "XY"), (1.5707, "YZ"), (-0.34, "ZX")])
    @pytest.mark.parametrize(
        "wire, reset, postselect", ((2, True, None), (3, False, 0), (0, True, 1))
    )
    @pytest.mark.parametrize("angle_type", ["float", "numpy", "jax"])
    def test_arbitrary_basis_with_program_capture(
        self, angle, plane, wire, reset, postselect, angle_type
    ):
        """Test that the measure_ functions are captured as expected"""
        import jax
        import networkx as nx

        if angle_type == "numpy":
            angle = np.array(angle)
        elif angle_type == "jax":
            angle = jax.numpy.array(angle)

        def circ():
            m = measure_arbitrary_basis(
                wire, angle=angle, plane=plane, reset=reset, postselect=postselect
            )
            qml.cond(m, qml.X, qml.Y)(0)
            qml.ftqc.make_graph_state(nx.grid_graph((4,)), [0, 1, 2, 3])
            return qml.expval(qml.Z(2))

        plxpr = jax.make_jaxpr(circ)()
        captured_measurement = str(plxpr.eqns[0])

        # measurement is captured as expected
        assert "measure_in_basis" in captured_measurement
        assert f"plane={plane}" in captured_measurement
        assert f"postselect={postselect}" in captured_measurement
        assert f"reset={reset}" in captured_measurement

        # dynamic parameters held in invars for numpy, and consts for jax
        if "jax" in angle_type:
            assert jax.numpy.isclose(angle, plxpr.consts[0])
        else:
            assert jax.numpy.isclose(angle, plxpr.eqns[0].invars[0].val)

        # Wires captured as invars
        assert jax.numpy.allclose(wire, plxpr.eqns[0].invars[1].val)

        # measurement value is assigned and passed forward
        conditional = str(plxpr.eqns[1])
        assert "cond" in conditional
        assert captured_measurement[:8] == "a:bool[]"
        assert "lambda ; a:i64[]" in conditional

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "func, kwargs",
        [
            (measure_x, {"wires": 2}),
            (measure_y, {"wires": 2}),
            (measure_arbitrary_basis, {"wires": 2, "angle": 1.2, "plane": "XY"}),
        ],
    )
    def test_calling_functions_with_capture_enabled(self, func, kwargs):
        """Test that the functions can still be called and return a measurement value
        with capture enabled."""

        m = func(**kwargs)
        assert isinstance(m, MeasurementValue)


class TestDrawParametricMidMeasure:
    @pytest.mark.matplotlib
    @pytest.mark.parametrize(
        "mp_class, expected_label",
        [(ParametricMidMeasureMP, "XY"), (XMidMeasureMP, "X"), (YMidMeasureMP, "Y")],
    )
    def test_draw_mpl_label(self, mp_class, expected_label):
        """Test that the plane label is added to the MCM in a mpl drawing"""

        from matplotlib import pyplot as plt

        dev = qml.device("default.qubit", wires=2)

        if mp_class == ParametricMidMeasureMP:
            args = {"wires": Wires([0]), "angle": np.pi / 4, "plane": "XY"}
        else:
            args = {"wires": Wires([0])}

        @qml.qnode(dev)
        def circ():
            mp_class(**args)
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()
        assert len(ax.texts) == 2  # one wire label, 1 box label on the MCM
        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == expected_label

        plt.close()

    @pytest.mark.parametrize(
        "mp_class, expected_label",
        [(ParametricMidMeasureMP, "XY"), (XMidMeasureMP, "X"), (YMidMeasureMP, "Y")],
    )
    @pytest.mark.matplotlib
    def test_draw_mpl_reset(self, mp_class, expected_label):
        """Test that the reset is added after the MCM as expected in a mpl drawing"""

        from matplotlib import pyplot as plt

        dev = qml.device("default.qubit", wires=2)

        if mp_class == ParametricMidMeasureMP:
            args = {"wires": Wires([0]), "angle": np.pi / 4, "plane": "XY"}
        else:
            args = {"wires": Wires([0])}

        @qml.qnode(dev)
        def circ():
            mp_class(**args, reset=True)
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()
        assert len(ax.texts) == 3  # one wire label, 1 box label on the MCM, one reset box

        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == expected_label
        assert ax.texts[2].get_text() == "|0⟩"

        plt.close()

    @pytest.mark.parametrize(
        "mp_class, expected_string",
        [
            (ParametricMidMeasureMP, "0: ──┤↗ˣʸ(0.79)├─┤  <Z>"),
            (XMidMeasureMP, "0: ──┤↗ˣ├─┤  <Z>"),
            (YMidMeasureMP, "0: ──┤↗ʸ├─┤  <Z>"),
        ],
    )
    def test_text_drawer(self, mp_class, expected_string):
        """Test that the text drawer works as expected"""

        dev = qml.device("default.qubit", wires=2)

        if mp_class == ParametricMidMeasureMP:
            args = {"wires": Wires([0]), "angle": np.pi / 4, "plane": "XY"}
        else:
            args = {"wires": Wires([0])}

        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ():
            mp_class(**args)
            return qml.expval(qml.Z(0))

        assert qml.draw(circ)() == expected_string


class TestDiagonalizeMCMs:
    def test_diagonalize_mcm_with_no_parametrized_mcms(self):
        """Test that the diagonalize_mcms transform leaves standard operations
        and MidMeasureMP on the tape untouched"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            m = qml.measure(0)
            qml.cond(m, qml.RY, qml.RX)(1.2, 0)

        original_tape = qml.tape.QuantumScript.from_queue(q)
        (new_tape,), _ = diagonalize_mcms(qml.tape.QuantumScript.from_queue(q))
        assert qml.equal(original_tape, new_tape)

    def test_diagonalize_mcm_transform(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        containing ParametricMidMeasureMPs"""

        tape = qml.tape.QuantumScript(
            [qml.RY(np.pi / 4, 0), ParametricMidMeasureMP(Wires([0]), angle=np.pi, plane="XY")]
        )
        diagonalizing_gates = tape.operations[1].diagonalizing_gates()

        (new_tape,), _ = diagonalize_mcms(tape)
        assert len(new_tape.operations) == 4

        assert new_tape.operations[1] == diagonalizing_gates[0]
        assert new_tape.operations[2] == diagonalizing_gates[1]

        assert isinstance(new_tape.operations[3], MidMeasureMP)
        assert not isinstance(new_tape.operations[3], ParametricMidMeasureMP)
        assert new_tape.operations[3].wires == tape.operations[1].wires

    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_diagonalize_mcm_transform_preserves_postselect(self, postselect):
        """Test that the diagonalize_mcm transform preserves postselet on a diagonalized MCM"""

        op = ParametricMidMeasureMP(Wires([0]), angle=np.pi, plane="XY", postselect=postselect)
        tape = qml.tape.QuantumScript([op])
        (new_tape,), _ = diagonalize_mcms(tape)

        assert isinstance(new_tape.operations[-1], MidMeasureMP)
        assert new_tape.operations[-1].postselect == postselect

    @pytest.mark.parametrize("reset", [True, False])
    def test_diagonalize_mcm_transform_preserves_reset(self, reset):
        """Test that the diagonalize_mcm transform preserves reset on a diagonalized MCM"""

        op = ParametricMidMeasureMP(Wires([0]), angle=np.pi, plane="XY", reset=reset)
        tape = qml.tape.QuantumScript([op])
        (new_tape,), _ = diagonalize_mcms(tape)

        assert isinstance(new_tape.operations[-1], MidMeasureMP)
        assert new_tape.operations[-1].reset == reset

    def test_diagonalize_conditional_mcms(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        conditionally applying two ParametricMidMeasureMPs as the true and false
        condition respectively"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            m = qml.measure(0)
            cond_measure(
                m == 1,
                partial(measure_arbitrary_basis, angle=1.2),
                partial(measure_arbitrary_basis, angle=-1.2),
            )(wires=2, plane="XY")

        original_tape = qml.tape.QuantumScript.from_queue(q)
        diag_gates_true = ParametricMidMeasureMP(
            Wires([2]), angle=1.2, plane="XY"
        ).diagonalizing_gates()
        diag_gates_false = ParametricMidMeasureMP(
            Wires([2]), angle=-1.2, plane="XY"
        ).diagonalizing_gates()

        (new_tape,), _ = diagonalize_mcms(qml.tape.QuantumScript.from_queue(q))
        assert len(new_tape.operations) == 7
        measurement = new_tape.operations[6]

        # conditional diagonalizing gates for the true_cond measurement
        original_meas = original_tape.operations[2]
        for gate, expected_base in zip(new_tape.operations[2:4], diag_gates_true):
            assert isinstance(gate, qml.ops.Conditional)
            assert gate.base == expected_base
            assert gate.wires == original_meas.wires
            assert gate.meas_val.measurements == original_meas.meas_val.measurements
            assert gate.meas_val.processing_fn == original_meas.meas_val.processing_fn

        # conditional diagonalizing gates for the false_cond measurement
        original_meas = original_tape.operations[3]
        for gate, expected_base in zip(new_tape.operations[4:6], diag_gates_false):
            assert isinstance(gate, qml.ops.Conditional)
            assert gate.base == expected_base
            assert gate.wires == original_meas.wires
            assert gate.meas_val.measurements == original_meas.meas_val.measurements
            assert gate.meas_val.processing_fn == original_meas.meas_val.processing_fn

        # diagonalized ParametricMidMeasureMP
        assert isinstance(measurement, MidMeasureMP)
        assert not isinstance(measurement, ParametricMidMeasureMP)

    def test_diagonalizing_mcm_used_as_cond(self):
        """Test that the measurements in a ``MeasurementValue`` passed to
        qml.cond are updated when those measurements are replaced by the
        diagonalize_mcms transform."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            mp = ParametricMidMeasureMP(0, angle=1.2, plane="YZ")
            mv = MeasurementValue([mp], processing_fn=lambda v: v)
            # using qml.cond and conditionally applying a gate
            qml.cond(mv == 0, partial(qml.RX, 1.2), partial(qml.RX, 2.4))(wires=2)

        original_tape = qml.tape.QuantumScript.from_queue(q)
        old_mp = original_tape.operations[1]
        assert isinstance(old_mp, ParametricMidMeasureMP)

        # expected ops: RX, diagonalizing gate, new_mp, conditional RX, conditional RX
        (new_tape,), _ = diagonalize_mcms(original_tape)
        assert len(new_tape.operations) == 5

        new_mp = new_tape.operations[2]
        assert not isinstance(new_mp, ParametricMidMeasureMP)
        assert isinstance(new_mp, MidMeasureMP)

        # the conditionals' MeasurementValues are mapped to the new, diagonalized mp
        processing_fns = []
        for op in new_tape.operations[3:]:
            assert isinstance(op, qml.ops.Conditional)
            assert op.meas_val.measurements == [new_mp]
            processing_fns.append(op.meas_val.processing_fn)

        # true and false processing fns are preserved on the conditionals
        assert [fn(0) for fn in processing_fns] == [True, False]
        assert [fn(1) for fn in processing_fns] == [False, True]

    def test_diagonalizing_mcm_used_as_cond_and_op(self):
        """Test diagonalization behaviour when all arguments of cond_measure
        require diagonalization. This test confirms that
            1. the measurements in a ``MeasurementValue`` passed to cond_measure
            are updated when those measurements are replaced by the diagonalize_mcms transform.
            2. the applied MCMs in cond_measure are diagonalized as expected
        """

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            mp = ParametricMidMeasureMP(0, angle=1.2, plane="YZ")
            mv = MeasurementValue([mp], processing_fn=lambda v: v)
            # using cond_measure and conditionally applying measurements
            cond_measure(mv == 0, measure_x, measure_y)(2)

        original_tape = qml.tape.QuantumScript.from_queue(q)
        old_mp = original_tape.operations[1]
        assert isinstance(old_mp, ParametricMidMeasureMP)

        # ops are: RX, diag_gate, mp, conditional(H), conditional(H), conditional(adjoint(S)), mp
        (new_tape,), _ = diagonalize_mcms(original_tape)
        assert len(new_tape.operations) == 7

        new_mp = new_tape.operations[2]
        assert not isinstance(new_mp, ParametricMidMeasureMP)
        assert isinstance(new_mp, MidMeasureMP)

        # the conditionals' MeasurementValues are mapped to the new, diagonalized mp
        processing_fns = []
        for op in new_tape.operations[3:6]:
            assert isinstance(op, qml.ops.Conditional)
            assert op.meas_val.measurements == [new_mp]
            processing_fns.append(op.meas_val.processing_fn)

        # true and false processing fns are preserved on the conditionals
        assert [fn(0) for fn in processing_fns] == [True, False, False]
        assert [fn(1) for fn in processing_fns] == [False, True, True]


class TestWorkflows:
    @pytest.mark.parametrize(
        "rot_gate, measurement_fn",
        [
            (partial(qml.RX, 2.345), measure_y),
            (partial(qml.RY, -2.345), measure_x),
            (partial(qml.RX, 2.345), partial(measure_arbitrary_basis, angle=np.pi / 2, plane="XY")),
        ],
    )
    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_simple_execution(self, rot_gate, measurement_fn, mcm_method, shots):
        """Test that we can execute a QNode with a ParametricMidMeasureMP and produce
        an accurate result"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            rot_gate(0)
            measurement_fn(0)
            return qml.expval(qml.Z(0))

        if shots:
            # the result is on the order of 1 (-0.7), and an uncertainty ~1.5-2 orders of magnitude
            # smaller than the result is sufficiently accurate for a shots-based measurement
            assert np.isclose(circ(), -np.sin(2.345), atol=0.03)
        else:
            assert np.isclose(circ(), -np.sin(2.345))

    @pytest.mark.parametrize(
        "rot_gate, measurement_fn",
        [
            (partial(qml.RX, np.pi / 2), measure_y),
            (partial(qml.RY, -np.pi / 2), measure_x),
            (
                partial(qml.RX, np.pi / 2),
                partial(measure_arbitrary_basis, angle=np.pi / 2, plane="XY"),
            ),
        ],
    )
    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_condition_of_cond(self, rot_gate, measurement_fn, mcm_method, shots):
        """Test that we can execute a QNode with a ParametricMidMeasureMP as the condition of a conditional,
        and produce an accurate result"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            rot_gate(0)
            m = measurement_fn(0)  # always 1

            qml.cond(m, qml.RX)(2.345, 1)
            return qml.expval(qml.Z(1)), qml.expval(qml.Z(0))

        if shots:
            # both results are on the order of 1 (-0.7, -1), and an uncertainty ~1.5-2 orders
            # of magnitude smaller than the result is sufficiently accurate for a shots-based measurement
            assert np.allclose(circ(), [np.cos(2.345), -1], atol=0.03)
        else:
            assert np.allclose(circ(), [np.cos(2.345), -1])

    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    @pytest.mark.parametrize("angle", [0.1234, np.array([-0.4321])])
    @pytest.mark.parametrize("angle_type", ["float", "numpy", "jax"])
    @pytest.mark.parametrize("use_jit", [False, True])
    def test_diagonalize_mcms_returns_parametrized_mcms(
        self, mcm_method, shots, angle, angle_type, use_jit
    ):  # pylint: disable=too-many-arguments
        """Test that when diagonalizing, parametrized mid-circuit measurements can be returned
        by the QNode"""

        if "jax" in angle_type or use_jit:
            jax = pytest.importorskip("jax")
            array_fn = jax.numpy.array
        else:
            array_fn = np.array

        if mcm_method == "tree-traversal" and use_jit:
            # https://docs.pennylane.ai/en/stable/introduction/dynamic_quantum_circuits.html#tree-traversal-algorithm
            pytest.skip("TT & jax.jit are incompatible")

        dev = qml.device("default.qubit")

        if angle_type == "numpy":
            angle = array_fn(angle)
        elif angle_type == "jax":
            angle = array_fn(angle)

        def jit_wrapper(func):
            if use_jit:
                import jax

                return jax.jit(func)
            return func

        @jit_wrapper
        @diagonalize_mcms
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ(angle):
            m0 = measure_x(0)
            m1 = measure_y(1)
            m2 = measure_arbitrary_basis(2, angle=angle, plane="XY")

            return qml.expval(m0), qml.expval(m1), qml.expval(m2)

        circ(angle)

    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_diagonalize_mcms_returns_cond_measure_result(self, mcm_method, shots):
        """Test that when diagonalizing, the MeasurementValue output by cond_measure can be returned
        by the QNode"""

        if mcm_method == "one-shot":
            pytest.xfail(reason="not implemented yet")  # sc-90607

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            qml.H(0)
            m0 = measure_x(0)
            m1 = cond_measure(m0, measure_x, measure_y)(1)
            return qml.expval(m1)

        circ()
