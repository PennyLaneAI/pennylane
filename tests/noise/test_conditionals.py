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

import pennylane as qp
from pennylane.boolean_fn import And, Not, Or, Xor
from pennylane.noise.conditionals import _get_ops, _get_wires


class TestNoiseConditionals:
    """Test for the Conditional classes"""

    def test_noise_conditional_lambda(self):
        """Test for BooleanFn builds correct objects"""

        func = qp.BooleanFn(lambda x: x < 10, "less_than_ten")

        assert isinstance(func, qp.BooleanFn)
        assert func(2) and not func(42)
        assert repr(func) == "BooleanFn(less_than_ten)"

    def test_noise_conditional_def(self):
        """Test for BooleanFn builds correct objects"""

        def greater_than_five(x):
            return x > 5

        func = qp.BooleanFn(greater_than_five)

        assert isinstance(func, qp.BooleanFn)
        assert not func(3) and func(7)
        assert repr(func) == "BooleanFn(greater_than_five)"

    def test_and_conditionals(self):
        """Test for BooleanFn supports bitwise AND"""

        @qp.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qp.BooleanFn
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

        @qp.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qp.BooleanFn
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

        @qp.BooleanFn
        def is_int(x):
            return isinstance(x, int)

        @qp.BooleanFn
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

        func = ~qp.BooleanFn(is_int)

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
            (qp.wires.Wires(["aurora", "borealis"]), "borealis", True),
            (qp.wires.Wires(1), [0], False),
            (qp.Y(2), 2, True),
            (qp.CNOT(["a", "c"]), "b", False),
            (qp.CZ(["a", "c"]), "c", True),
            (qp.DoubleExcitation(1.2, ["alpha", "beta", "gamma", "delta"]), "alpha", True),
            (qp.TrotterProduct(qp.Z(0) + qp.Z(1), -3j), 2, False),
            (qp.TrotterProduct(qp.Z("a") + qp.Z("b"), -3j), "b", True),
            (qp.expval(qp.CNOT([0, 1])), qp.Z(0), True),
            ([qp.counts(qp.Z("a")), qp.sample(qp.Y("b"))], qp.Y(0), False),
            (qp.mutual_info([0, 1], ["a", "b"]), qp.CNOT([0, "b"]), True),
        ],
    )
    def test_wires_in(self, obj, wires, result):
        """Test for checking WiresIn work as expected for checking if a wire exist in a set of specified wires"""

        func = qp.noise.wires_in(obj)

        assert isinstance(func, qp.BooleanFn)
        assert str(func) == f"WiresIn({list(_get_wires(obj))})"
        assert func(wires) == result

    @pytest.mark.parametrize(
        ("obj", "wires", "result"),
        [
            (0, 0, True),
            ([1, 2], [1, 2], True),
            (qp.wires.Wires(["street", "fighter"]), "fighter", False),
            (qp.wires.Wires(1), [1], True),
            (qp.Y(2), 2, True),
            (qp.CNOT(["a", "c"]), "b", False),
            (qp.CNOT(["c", "d"]), ["c", "d"], True),
            (qp.TrotterProduct(qp.Z(0) + qp.Z(1), -3j), 2, False),
            (qp.TrotterProduct(qp.Z("b") + qp.Z("a"), -3j), ["b", "a"], True),
            (qp.counts(qp.Z("a")), qp.Y("a"), True),
            (qp.shadow_expval(qp.X(0) + qp.Y(1) @ qp.Z(2)), qp.CSWAP([0, 1, 2]), True),
            (qp.measure(0), qp.RX(1.23, wires=1), False),
        ],
    )
    def test_wires_eq(self, obj, wires, result):
        """Test for checking WiresEq work as expected for checking if a set of wires is equal to specified wires"""

        func = qp.noise.wires_eq(obj)
        _wires = list(_get_wires(obj))
        assert isinstance(func, qp.BooleanFn)
        assert str(func) == f"WiresEq({_wires if len(_wires) > 1 else _wires[0]})"
        assert func(wires) == result

    def test_get_wires_error(self):
        """Test for checking _get_wires method raise correct error"""

        with pytest.raises(ValueError, match="Wires cannot be computed for"):
            _get_wires(qp.RX)

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            ([qp.RX, qp.RY], qp.RY(1.0, 1), True),
            ([qp.RX, qp.RY], qp.RX, True),
            (qp.RX(0, 1), qp.RY(1.0, 1), False),
            ("RX", qp.RX(0, 1), True),
            (["CZ", "RY", "CNOT"], qp.CNOT([0, 1]), True),
            (qp.Y(1), qp.RY(1.0, 1), False),
            (qp.CNOT(["a", "c"]), qp.CNOT([0, 1]), True),
            ([qp.S("a"), qp.adjoint(qp.T)("b")], qp.adjoint(qp.T)([0]), True),
            ([qp.CZ(["a", "c"]), qp.Y(1)], qp.CZ([0, 1]), True),
            ([qp.RZ(1.9, 0), qp.Z(0) @ qp.Z(1)], qp.Z("b") @ qp.Z("a"), True),
            ([qp.Z(0) + qp.Z(1), qp.Z(2)], qp.Z("b") + qp.Z("a"), True),
            ([qp.Z(0), qp.Z(0) + 1.2 * qp.Z(1)], qp.Y("b") + qp.Y("a"), False),
            ([qp.expval(qp.Z(0)), qp.var(qp.Y("a"))], qp.Z("b"), True),
            (
                [qp.counts(qp.Z(0) @ qp.X(1)), qp.sample(qp.Y("a") @ qp.Z("b"))],
                qp.Z("b") @ qp.X(2),
                True,
            ),
            (
                [qp.counts(qp.Z(0) @ qp.X(1)), qp.sample(qp.Y("a") @ qp.Z("b"))],
                qp.X("b") @ qp.Y(2),
                False,
            ),
            (
                [qp.shadow_expval(qp.X(0) + qp.Y(1)), qp.purity(0)],
                qp.X("a") + qp.Y("b"),
                True,
            ),
        ],
    )
    def test_op_in(self, obj, op, result):
        """Test for checking OpIn work as expected for checking if an operation exist in a set of specified operation"""

        func = qp.noise.op_in(obj)

        assert isinstance(func, qp.BooleanFn)

        op_repr = list(getattr(op, "__name__") for op in _get_ops(obj))
        assert str(func) == f"OpIn({op_repr})"
        assert func(op) == result

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            ("I", qp.I(wires=[0, 1]), True),
            (qp.Y(1), qp.RY(1.0, 1), False),
            (qp.CNOT(["a", "c"]), qp.CNOT([0, 1]), True),
            (qp.RX(0, 1), qp.RY(1.0, 1), False),
            ("RX", qp.RX(0, 1), True),
            ([qp.RX, qp.RY], qp.RX, False),
            (qp.measure(1, reset=True), qp.measure(2, reset=True), True),
            (qp.measure(1, reset=True), qp.measure(1, reset=False), False),
            ([qp.RX, qp.RY], [qp.RX(1.0, 1), qp.RY(2.0, 2)], True),
            (["CZ", "RY"], [qp.CZ([0, 1]), qp.RY(1.0, [1])], True),
            (qp.Z(0) @ qp.Z(1), qp.Z("b") @ qp.Z("a"), True),
            (qp.Z(0) + qp.Z(1), qp.Z("b") + qp.Z("a"), True),
            (qp.Z(0) + 1.2 * qp.Z(1), 2.4 * qp.Z("b") + qp.Z("a"), False),
            (qp.Z(0) + 1.2 * qp.Z(1), qp.Z("b") + qp.Z("a"), False),
            (qp.exp(qp.RX(1.2, 0), 1.2), qp.exp(qp.RX(2.3, "a"), 1.2), True),
            (qp.exp(qp.Z(0) + qp.Z(1), 1.2), qp.exp(qp.Z("b") + qp.Z("a"), 1.2), True),
            (qp.exp(qp.Z(0) @ qp.Z(1), 2j), qp.exp(qp.Z("b") @ qp.Z("a"), 1j), False),
            (qp.expval(qp.Z(0) @ qp.Y(1)), qp.Z("b"), False),
            (qp.sample(qp.Y("a") @ qp.Z("b")), qp.Y("b") @ qp.Z(2), True),
            (qp.shadow_expval(qp.X(0) + qp.Y(1)), qp.X("a") + qp.Y("b"), True),
        ],
    )
    def test_op_eq(self, obj, op, result):
        """Test for checking OpEq work as expected for checking if an operation is equal to specified operation"""

        func = qp.noise.op_eq(obj)

        assert isinstance(func, qp.BooleanFn)

        op_repr = [getattr(op, "__name__", op) for op in _get_ops(obj)]
        assert str(func) == f"OpEq({op_repr if len(op_repr) > 1 else op_repr[0]})"
        assert func(op) == result

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            (qp.expval(qp.X(0)), qp.expval(qp.Z(0)), True),
            (qp.expval(qp.X(0)), qp.sample(qp.X(0)), False),
            (qp.purity(wires=[0, 1]), qp.purity(wires=["a"]), True),
            (qp.mutual_info([0], [1]), qp.mutual_info(["a", "b"], ["c"]), True),
            (qp.vn_entropy([0, 1], log_base=3), qp.vn_entropy(["a"], log_base=10), True),
            (qp.shadow_expval(qp.X(0) + qp.Y(1)), qp.shadow_expval(qp.X(0) @ qp.Y(1)), True),
            (qp.measure(1, reset=True), qp.measure(1, reset=False), True),
            (qp.counts(wires=[0, 1]), qp.state(), False),
            (qp.density_matrix(wires=[0, 1]), qp.measurements.StateMP(wires=[0, 1]), False),
            (qp.expval(qp.X(0)), qp.Z(0), False),
            (qp.sample(qp.X(0)), qp.Y, False),
            (qp.var(qp.X(0)), qp.adjoint, False),
            (qp.expval(qp.X(0)), [qp.sample(qp.X(0)), qp.expval(qp.Z(0))], False),
        ],
    )
    def test_meas_eq(self, obj, op, result):
        """Test for checking MeasEq work as expected for checking if an measurement process is equal to specified measurement process"""

        func = qp.noise.meas_eq(obj)

        assert isinstance(func, qp.BooleanFn)

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
            qp.noise.meas_eq(qp.RX)

        with pytest.raises(
            ValueError, match="MeasEq should be initialized with a MeasurementProcess"
        ):
            qp.noise.meas_eq(qp.adjoint)

    def test_conditional_bitwise(self):
        """Test that conditionals can be operated with bitwise operations"""

        cond1 = qp.noise.wires_eq(0)
        cond2 = qp.noise.op_eq(qp.X)
        conds = (cond1.condition, cond2.condition)

        and_cond = cond1 & cond2
        assert and_cond(qp.X(0)) and not and_cond(qp.X(1))
        assert and_cond.condition == conds

        or_cond = cond1 | cond2
        assert or_cond(qp.X(1)) and not or_cond(qp.Y(1))
        assert or_cond.condition == conds

        xor_cond = cond1 ^ cond2
        assert xor_cond(qp.X(1)) and not xor_cond(qp.Y(1))
        assert xor_cond.condition == conds

        not_cond = ~cond1
        assert not_cond(qp.X(1))
        assert not_cond.condition == conds[:1]

    def test_partial_wires(self):
        """Test for checking partial_wires work as expected for building callables to make op with correct wires"""

        op = qp.noise.partial_wires(qp.RX(1.2, [12]))(qp.RY(1.0, ["wires"]))
        qp.assert_equal(op, qp.RX(1.2, wires=["wires"]))

        op = qp.noise.partial_wires(qp.RX, 3.2, [20])(qp.RY(1.0, [0]))
        qp.assert_equal(op, qp.RX(3.2, wires=[0]))

        op = qp.noise.partial_wires(qp.RX, phi=1.2)(qp.RY(1.0, [2]))
        qp.assert_equal(op, qp.RX(1.2, wires=[2]))

        op = qp.noise.partial_wires(qp.PauliRot(1.2, "XY", wires=(0, 1)))(qp.CNOT([2, 1]))
        qp.assert_equal(op, qp.PauliRot(1.2, "XY", wires=(2, 1)))

        op = qp.noise.partial_wires(qp.RX(1.2, [12]), phi=2.3)(qp.RY(1.0, ["light"]))
        qp.assert_equal(op, qp.RX(2.3, wires=["light"]))

        op = qp.noise.partial_wires(qp.adjoint(qp.X(2)))(3)
        qp.assert_equal(op, qp.adjoint(qp.X(3)))

        op = qp.noise.partial_wires(qp.adjoint)(op=qp.X(7) @ qp.Y(8))
        qp.assert_equal(op, qp.adjoint(qp.X(7) @ qp.Y(8)))

        op = qp.noise.partial_wires(qp.ctrl, op=qp.Hadamard(0), control=[1, 2])("a")
        qp.assert_equal(op, qp.ctrl(qp.Hadamard("a"), control=[1, 2]))

        op = qp.noise.partial_wires(qp.PrepSelPrep(qp.X(1) + qp.Z(2), control=3))([2, 3, 4])
        qp.assert_equal(op, qp.PrepSelPrep(qp.X(3) + qp.Z(4), control=2))

        mp = qp.noise.partial_wires(qp.expval, op=qp.Z(9))("photon")
        qp.assert_equal(mp, qp.expval(qp.Z("photon")))

        mp = qp.noise.partial_wires(qp.density_matrix, wires=2)(1)
        qp.assert_equal(mp, qp.density_matrix([1]))

        mp = qp.noise.partial_wires(qp.mutual_info(1, 2))([2, 3])
        qp.assert_equal(mp, qp.mutual_info(2, 3))

        mp = qp.noise.partial_wires(qp.shadow_expval, H=qp.X(0) @ qp.Y(1))(["bra", "ket"])
        qp.assert_equal(mp, qp.shadow_expval(H=qp.X("bra") @ qp.Y("ket")))

        mp = qp.noise.partial_wires(qp.probs)(op=qp.X(7))
        qp.assert_equal(mp, qp.probs(op=qp.X(7)))

        mp = qp.noise.partial_wires(qp.counts)(qp.X("light"))
        qp.assert_equal(mp, qp.counts(wires=["light"]))

    def test_partial_wires_queuing(self):
        """Test for checking partial_wires correctly queue operations"""

        op1 = qp.X(2)
        qs1 = qp.tape.make_qscript(qp.noise.partial_wires(qp.DepolarizingChannel, 0.01))
        qs2 = qp.tape.make_qscript(qp.noise.partial_wires(qp.DepolarizingChannel(0.01, [0])))
        assert qs1(op1).operations == qs2(op1).operations and len(qs1(op1).operations) == 1

        op1 = qp.CNOT(["a", "b"])
        d1, d2 = [-0.9486833] * 4, [-0.31622777] * 2
        krs = [qp.math.diag(d1), qp.math.diag(d2, k=2) + qp.math.diag(d2, k=-2)]
        qs1 = qp.tape.make_qscript(qp.noise.partial_wires(qp.QubitChannel, krs))
        qs2 = qp.tape.make_qscript(qp.noise.partial_wires(qp.QubitChannel(krs, [0, 1])))
        assert qs1(op1).operations == qs2(op1).operations and len(qs1(op1).operations) == 1

    def test_partial_wires_error(self):
        """Test for checking partial_wires raise correct error when args are given"""

        with pytest.raises(
            ValueError, match="Args cannot be provided when operation is an instance"
        ):
            qp.noise.partial_wires(qp.RX(1.2, [12]), 1.2)(qp.RY(1.0, ["wires"]))
