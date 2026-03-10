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

import pennylane as qml
from pennylane.boolean_fn import And, Not, Or, Xor
from pennylane.noise.conditionals import _get_ops, _get_wires


class TestNoiseConditionals:
    """Test for the Conditional classes"""

    def test_noise_conditional_lambda(self):
        """Test for BooleanFn builds correct objects"""

        func = qml.BooleanFn(lambda x: x < 10, "less_than_ten")

        assert isinstance(func, qml.BooleanFn)
        assert func(2) and not func(42)
        assert repr(func) == "BooleanFn(less_than_ten)"

    def test_noise_conditional_def(self):
        """Test for BooleanFn builds correct objects"""

        def greater_than_five(x):
            return x > 5

        func = qml.BooleanFn(greater_than_five)

        assert isinstance(func, qml.BooleanFn)
        assert not func(3) and func(7)
        assert repr(func) == "BooleanFn(greater_than_five)"

    def test_and_conditionals(self):
        """Test for BooleanFn supports bitwise AND"""

        @qml.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qml.BooleanFn
        def has_bit_length_3(x):
            return x.bit_length() == 3

        func = is_int & has_bit_length_3

        assert isinstance(func, And)
        assert func(4) and not func(2.3)
        assert repr(func) == "And(is_int, has_bit_length_3)"
        assert str(func) == "is_int & has_bit_length_3"
        assert func.bitwise

    def test_or_conditionals(self):
        """Test for BooleanFn supports bitwise OR"""

        @qml.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qml.BooleanFn
        def less_than_five(x):
            return x < 5

        func = is_int | less_than_five

        assert isinstance(func, Or)
        assert func(4) and not func(7.5)
        assert repr(func) == "Or(is_int, less_than_five)"
        assert str(func) == "is_int | less_than_five"
        assert func.bitwise

    def test_xor_conditionals(self):
        """Test for BooleanFn supports bitwise XOR"""

        @qml.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qml.BooleanFn
        def has_bit_length_3(x):
            return x.bit_length() == 3

        func = is_int ^ has_bit_length_3

        assert isinstance(func, Xor)
        assert not func(4) and func(11)
        assert repr(func) == "Xor(is_int, has_bit_length_3)"
        assert str(func) == "is_int ^ has_bit_length_3"
        assert func.bitwise

    def test_not_conditionals(self):
        """Test for BooleanFn supports bitwise NOT"""

        def is_int(x):
            return isinstance(x, int)

        func = ~qml.BooleanFn(is_int)

        assert isinstance(func, Not)
        assert not func(4) and func(7.5)
        assert repr(func) == "Not(is_int)"
        assert str(func) == "~is_int"
        assert func.bitwise


class TestNoiseFunctions:
    """Test for the Conditional methods"""

    @pytest.mark.parametrize(
        ("obj", "wires", "result"),
        [
            (0, 0, True),
            ([1, 2, 3], 0, False),
            (qml.wires.Wires(["aurora", "borealis"]), "borealis", True),
            (qml.wires.Wires(1), [0], False),
            (qml.Y(2), 2, True),
            (qml.CNOT(["a", "c"]), "b", False),
            (qml.CZ(["a", "c"]), "c", True),
            (qml.DoubleExcitation(1.2, ["alpha", "beta", "gamma", "delta"]), "alpha", True),
            (qml.TrotterProduct(qml.Z(0) + qml.Z(1), -3j), 2, False),
            (qml.TrotterProduct(qml.Z("a") + qml.Z("b"), -3j), "b", True),
            (qml.expval(qml.CNOT([0, 1])), qml.Z(0), True),
            ([qml.counts(qml.Z("a")), qml.sample(qml.Y("b"))], qml.Y(0), False),
            (qml.mutual_info([0, 1], ["a", "b"]), qml.CNOT([0, "b"]), True),
        ],
    )
    def test_wires_in(self, obj, wires, result):
        """Test for checking WiresIn work as expected for checking if a wire exist in a set of specified wires"""

        func = qml.noise.wires_in(obj)

        assert isinstance(func, qml.BooleanFn)
        assert str(func) == f"WiresIn({list(_get_wires(obj))})"
        assert func(wires) == result

    @pytest.mark.parametrize(
        ("obj", "wires", "result"),
        [
            (0, 0, True),
            ([1, 2], [1, 2], True),
            (qml.wires.Wires(["street", "fighter"]), "fighter", False),
            (qml.wires.Wires(1), [1], True),
            (qml.Y(2), 2, True),
            (qml.CNOT(["a", "c"]), "b", False),
            (qml.CNOT(["c", "d"]), ["c", "d"], True),
            (qml.TrotterProduct(qml.Z(0) + qml.Z(1), -3j), 2, False),
            (qml.TrotterProduct(qml.Z("b") + qml.Z("a"), -3j), ["b", "a"], True),
            (qml.counts(qml.Z("a")), qml.Y("a"), True),
            (qml.shadow_expval(qml.X(0) + qml.Y(1) @ qml.Z(2)), qml.CSWAP([0, 1, 2]), True),
            (qml.measure(0), qml.RX(1.23, wires=1), False),
        ],
    )
    def test_wires_eq(self, obj, wires, result):
        """Test for checking WiresEq work as expected for checking if a set of wires is equal to specified wires"""

        func = qml.noise.wires_eq(obj)
        _wires = list(_get_wires(obj))
        assert isinstance(func, qml.BooleanFn)
        assert str(func) == f"WiresEq({_wires if len(_wires) > 1 else _wires[0]})"
        assert func(wires) == result

    def test_get_wires_error(self):
        """Test for checking _get_wires method raise correct error"""

        with pytest.raises(ValueError, match="Wires cannot be computed for"):
            _get_wires(qml.RX)

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            ([qml.RX, qml.RY], qml.RY(1.0, 1), True),
            ([qml.RX, qml.RY], qml.RX, True),
            (qml.RX(0, 1), qml.RY(1.0, 1), False),
            ("RX", qml.RX(0, 1), True),
            (["CZ", "RY", "CNOT"], qml.CNOT([0, 1]), True),
            (qml.Y(1), qml.RY(1.0, 1), False),
            (qml.CNOT(["a", "c"]), qml.CNOT([0, 1]), True),
            ([qml.S("a"), qml.adjoint(qml.T)("b")], qml.adjoint(qml.T)([0]), True),
            ([qml.CZ(["a", "c"]), qml.Y(1)], qml.CZ([0, 1]), True),
            ([qml.RZ(1.9, 0), qml.Z(0) @ qml.Z(1)], qml.Z("b") @ qml.Z("a"), True),
            ([qml.Z(0) + qml.Z(1), qml.Z(2)], qml.Z("b") + qml.Z("a"), True),
            ([qml.Z(0), qml.Z(0) + 1.2 * qml.Z(1)], qml.Y("b") + qml.Y("a"), False),
            ([qml.expval(qml.Z(0)), qml.var(qml.Y("a"))], qml.Z("b"), True),
            (
                [qml.counts(qml.Z(0) @ qml.X(1)), qml.sample(qml.Y("a") @ qml.Z("b"))],
                qml.Z("b") @ qml.X(2),
                True,
            ),
            (
                [qml.counts(qml.Z(0) @ qml.X(1)), qml.sample(qml.Y("a") @ qml.Z("b"))],
                qml.X("b") @ qml.Y(2),
                False,
            ),
            (
                [qml.shadow_expval(qml.X(0) + qml.Y(1)), qml.purity(0)],
                qml.X("a") + qml.Y("b"),
                True,
            ),
        ],
    )
    def test_op_in(self, obj, op, result):
        """Test for checking OpIn work as expected for checking if an operation exist in a set of specified operation"""

        func = qml.noise.op_in(obj)

        assert isinstance(func, qml.BooleanFn)

        op_repr = list(getattr(op, "__name__") for op in _get_ops(obj))
        assert str(func) == f"OpIn({op_repr})"
        assert func(op) == result

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            ("I", qml.I(wires=[0, 1]), True),
            (qml.Y(1), qml.RY(1.0, 1), False),
            (qml.CNOT(["a", "c"]), qml.CNOT([0, 1]), True),
            (qml.RX(0, 1), qml.RY(1.0, 1), False),
            ("RX", qml.RX(0, 1), True),
            ([qml.RX, qml.RY], qml.RX, False),
            (qml.measure(1, reset=True), qml.measure(2, reset=True), True),
            (qml.measure(1, reset=True), qml.measure(1, reset=False), False),
            ([qml.RX, qml.RY], [qml.RX(1.0, 1), qml.RY(2.0, 2)], True),
            (["CZ", "RY"], [qml.CZ([0, 1]), qml.RY(1.0, [1])], True),
            (qml.Z(0) @ qml.Z(1), qml.Z("b") @ qml.Z("a"), True),
            (qml.Z(0) + qml.Z(1), qml.Z("b") + qml.Z("a"), True),
            (qml.Z(0) + 1.2 * qml.Z(1), 2.4 * qml.Z("b") + qml.Z("a"), False),
            (qml.Z(0) + 1.2 * qml.Z(1), qml.Z("b") + qml.Z("a"), False),
            (qml.exp(qml.RX(1.2, 0), 1.2), qml.exp(qml.RX(2.3, "a"), 1.2), True),
            (qml.exp(qml.Z(0) + qml.Z(1), 1.2), qml.exp(qml.Z("b") + qml.Z("a"), 1.2), True),
            (qml.exp(qml.Z(0) @ qml.Z(1), 2j), qml.exp(qml.Z("b") @ qml.Z("a"), 1j), False),
            (qml.expval(qml.Z(0) @ qml.Y(1)), qml.Z("b"), False),
            (qml.sample(qml.Y("a") @ qml.Z("b")), qml.Y("b") @ qml.Z(2), True),
            (qml.shadow_expval(qml.X(0) + qml.Y(1)), qml.X("a") + qml.Y("b"), True),
        ],
    )
    def test_op_eq(self, obj, op, result):
        """Test for checking OpEq work as expected for checking if an operation is equal to specified operation"""

        func = qml.noise.op_eq(obj)

        assert isinstance(func, qml.BooleanFn)

        op_repr = [getattr(op, "__name__", op) for op in _get_ops(obj)]
        assert str(func) == f"OpEq({op_repr if len(op_repr) > 1 else op_repr[0]})"
        assert func(op) == result

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            (qml.expval(qml.X(0)), qml.expval(qml.Z(0)), True),
            (qml.expval(qml.X(0)), qml.sample(qml.X(0)), False),
            (qml.purity(wires=[0, 1]), qml.purity(wires=["a"]), True),
            (qml.mutual_info([0], [1]), qml.mutual_info(["a", "b"], ["c"]), True),
            (qml.vn_entropy([0, 1], log_base=3), qml.vn_entropy(["a"], log_base=10), True),
            (qml.shadow_expval(qml.X(0) + qml.Y(1)), qml.shadow_expval(qml.X(0) @ qml.Y(1)), True),
            (qml.measure(1, reset=True), qml.measure(1, reset=False), True),
            (qml.counts(wires=[0, 1]), qml.state(), False),
            (qml.density_matrix(wires=[0, 1]), qml.measurements.StateMP(wires=[0, 1]), False),
            (qml.expval(qml.X(0)), qml.Z(0), False),
            (qml.sample(qml.X(0)), qml.Y, False),
            (qml.var(qml.X(0)), qml.adjoint, False),
            (qml.expval(qml.X(0)), [qml.sample(qml.X(0)), qml.expval(qml.Z(0))], False),
        ],
    )
    def test_meas_eq(self, obj, op, result):
        """Test for checking MeasEq work as expected for checking if an measurement process is equal to specified measurement process"""

        func = qml.noise.meas_eq(obj)

        assert isinstance(func, qml.BooleanFn)

        op_mps = list(getattr(op, "__name__", op.__class__.__name__) for op in func.condition)
        op_repr = [
            repr(op) if not isinstance(op, property) else repr(func.condition[idx].__name__)
            for idx, op in enumerate(op_mps)
        ]
        assert str(func) == f"MeasEq({op_repr if len(op_repr) > 1 else op_repr[0]})"
        assert func(op) == result

    def test_meas_eq_error(self):
        """Test for checking MeasEq raise correct error when used with something that is not a measurement process"""

        with pytest.raises(
            ValueError, match="MeasEq should be initialized with a MeasurementProcess"
        ):
            qml.noise.meas_eq(qml.RX)

        with pytest.raises(
            ValueError, match="MeasEq should be initialized with a MeasurementProcess"
        ):
            qml.noise.meas_eq(qml.adjoint)

    def test_conditional_bitwise(self):
        """Test that conditionals can be operated with bitwise operations"""

        cond1 = qml.noise.wires_eq(0)
        cond2 = qml.noise.op_eq(qml.X)
        conds = (cond1.condition, cond2.condition)

        and_cond = cond1 & cond2
        assert and_cond(qml.X(0)) and not and_cond(qml.X(1))
        assert and_cond.condition == conds

        or_cond = cond1 | cond2
        assert or_cond(qml.X(1)) and not or_cond(qml.Y(1))
        assert or_cond.condition == conds

        xor_cond = cond1 ^ cond2
        assert xor_cond(qml.X(1)) and not xor_cond(qml.Y(1))
        assert xor_cond.condition == conds

        not_cond = ~cond1
        assert not_cond(qml.X(1))
        assert not_cond.condition == conds[:1]

    def test_partial_wires(self):
        """Test for checking partial_wires work as expected for building callables to make op with correct wires"""

        op = qml.noise.partial_wires(qml.RX(1.2, [12]))(qml.RY(1.0, ["wires"]))
        qml.assert_equal(op, qml.RX(1.2, wires=["wires"]))

        op = qml.noise.partial_wires(qml.RX, 3.2, [20])(qml.RY(1.0, [0]))
        qml.assert_equal(op, qml.RX(3.2, wires=[0]))

        op = qml.noise.partial_wires(qml.RX, phi=1.2)(qml.RY(1.0, [2]))
        qml.assert_equal(op, qml.RX(1.2, wires=[2]))

        op = qml.noise.partial_wires(qml.PauliRot(1.2, "XY", wires=(0, 1)))(qml.CNOT([2, 1]))
        qml.assert_equal(op, qml.PauliRot(1.2, "XY", wires=(2, 1)))

        op = qml.noise.partial_wires(qml.RX(1.2, [12]), phi=2.3)(qml.RY(1.0, ["light"]))
        qml.assert_equal(op, qml.RX(2.3, wires=["light"]))

        op = qml.noise.partial_wires(qml.adjoint(qml.X(2)))(3)
        qml.assert_equal(op, qml.adjoint(qml.X(3)))

        op = qml.noise.partial_wires(qml.adjoint)(op=qml.X(7) @ qml.Y(8))
        qml.assert_equal(op, qml.adjoint(qml.X(7) @ qml.Y(8)))

        op = qml.noise.partial_wires(qml.ctrl, op=qml.Hadamard(0), control=[1, 2])("a")
        qml.assert_equal(op, qml.ctrl(qml.Hadamard("a"), control=[1, 2]))

        op = qml.noise.partial_wires(qml.PrepSelPrep(qml.X(1) + qml.Z(2), control=3))([2, 3, 4])
        qml.assert_equal(op, qml.PrepSelPrep(qml.X(3) + qml.Z(4), control=2))

        mp = qml.noise.partial_wires(qml.expval, op=qml.Z(9))("photon")
        qml.assert_equal(mp, qml.expval(qml.Z("photon")))

        mp = qml.noise.partial_wires(qml.density_matrix, wires=2)(1)
        qml.assert_equal(mp, qml.density_matrix([1]))

        mp = qml.noise.partial_wires(qml.mutual_info(1, 2))([2, 3])
        qml.assert_equal(mp, qml.mutual_info(2, 3))

        mp = qml.noise.partial_wires(qml.shadow_expval, H=qml.X(0) @ qml.Y(1))(["bra", "ket"])
        qml.assert_equal(mp, qml.shadow_expval(H=qml.X("bra") @ qml.Y("ket")))

        mp = qml.noise.partial_wires(qml.probs)(op=qml.X(7))
        qml.assert_equal(mp, qml.probs(op=qml.X(7)))

        mp = qml.noise.partial_wires(qml.counts)(qml.X("light"))
        qml.assert_equal(mp, qml.counts(wires=["light"]))

    def test_partial_wires_queuing(self):
        """Test for checking partial_wires correctly queue operations"""

        op1 = qml.X(2)
        qs1 = qml.tape.make_qscript(qml.noise.partial_wires(qml.DepolarizingChannel, 0.01))
        qs2 = qml.tape.make_qscript(qml.noise.partial_wires(qml.DepolarizingChannel(0.01, [0])))
        assert qs1(op1).operations == qs2(op1).operations and len(qs1(op1).operations) == 1

        op1 = qml.CNOT(["a", "b"])
        d1, d2 = [-0.9486833] * 4, [-0.31622777] * 2
        krs = [qml.math.diag(d1), qml.math.diag(d2, k=2) + qml.math.diag(d2, k=-2)]
        qs1 = qml.tape.make_qscript(qml.noise.partial_wires(qml.QubitChannel, krs))
        qs2 = qml.tape.make_qscript(qml.noise.partial_wires(qml.QubitChannel(krs, [0, 1])))
        assert qs1(op1).operations == qs2(op1).operations and len(qs1(op1).operations) == 1

    def test_partial_wires_error(self):
        """Test for checking partial_wires raise correct error when args are given"""

        with pytest.raises(
            ValueError, match="Args cannot be provided when operation is an instance"
        ):
            qml.noise.partial_wires(qml.RX(1.2, [12]), 1.2)(qml.RY(1.0, ["wires"]))
