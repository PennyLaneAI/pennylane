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
"""
Tests for the QROM template.
"""

import math

import numpy
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.decomposition.decomposition_rule import DecompositionRule
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.templates.subroutines.qrom import (
    _calculate_n_select_work_wires,
    _count_tempAND_in_measurement_qrom,
    _qrom_decomposition,
    _qrom_measurement_condition,
    _qrom_measurement_decomposition,
    _qrom_measurement_resources,
)
from pennylane.templates.subroutines.select import _select_decomp_unary

clifford_t_measure = {
    qp.H,
    qp.T,
    qp.S,
    qp.X,
    qp.Y,
    qp.Z,
    qp.CNOT,
    qp.CZ,
    qp.Hadamard,
    PauliMeasure,
}

has_jax = True
try:
    from jax import numpy as jnp
except ImportError:
    has_jax = False


@pytest.mark.jax
def test_assert_valid_qrom():
    """Run standard validity tests."""
    data = (
        (0, 0, 0),
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (0, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )

    op = qp.QROM(data, control_wires=[0, 1, 2], target_wires=[3, 4, 5], work_wires=[6, 7, 8])
    qp.ops.functions.assert_valid(op, skip_differentiation=True)


@pytest.mark.jax
def test_falsy_zero_as_work_wire():
    """Test that work wire is not treated as a falsy zero."""
    op = qp.QROM(
        ((1,), (0,), (0,), (1,)),
        control_wires=[1, 2],
        target_wires=[3],
        work_wires=0,
    )
    qp.ops.functions.assert_valid(op, skip_differentiation=True)


class TestQROM:
    """Test the qp.QROM template."""

    @pytest.mark.jax
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        ("data", "target_wires", "control_wires", "work_wires", "clean"),
        [
            (
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
                None,
                False,
            ),
            (
                np.array(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                    ]
                ),
                np.array([0, 1, 2]),
                np.array([3, 4]),
                None,
                True,
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5],
                True,
            ),
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5, 6, 7, 8, 9],
                True,
            ),
            (
                np.array([[0, 1], [0, 1], [0, 0], [0, 0]]),
                np.array(["a", "b"]),
                np.array([2, 3]),
                np.array([4, 5, 6]),
                False,
            ),
            (
                np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0],
                    ]
                ),
                np.array([0, 1, "b"]),
                np.array([2, 3]),
                None,
                False,
            ),
            (
                np.array(
                    [
                        [1, 1, 1, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                    ]
                ),
                np.array([0, 1, "b", "d"]),
                np.array([2, 3]),
                np.array(["a", 5, 6, 7]),
                True,
            ),
        ],
    )
    def test_operation_result(
        self, data, target_wires, control_wires, work_wires, clean
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the QROM template output."""
        dev = qp.device("default.qubit")

        if has_jax and not isinstance(data, numpy.ndarray):
            data, control_wires, target_wires, work_wires = (
                jnp.array(data),
                jnp.array(control_wires),
                jnp.array(target_wires),
                jnp.array(work_wires),
            )

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(j):
            qp.BasisEmbedding(j, wires=control_wires)
            qp.QROM(data, control_wires, target_wires, work_wires, clean)
            return qp.sample(wires=target_wires)

        for j in range(2 ** len(control_wires)):
            assert np.allclose(circuit(j), [int(bit) for bit in data[j]])

    @pytest.mark.parametrize(
        ("data", "target_wires", "control_wires", "work_wires"),
        [
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [0, 1],
                [2, 3],
                [4, 5],
            ),
            (
                [[0, 1], [0, 1], [0, 0], [0, 0]],
                ["a", "b"],
                [2, 3],
                [4, 5, 6],
            ),
            (
                [
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0],
                ],
                [0, 1, "b"],
                [2, 3],
                ["a", 5, 6],
            ),
            (
                [
                    [1, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                ],
                [0, 1, "b", "d"],
                [2, 3],
                ["a", 5, 6, 7],
            ),
        ],
    )
    def test_work_wires_output(self, data, target_wires, control_wires, work_wires):
        """Tests that the ``clean = True`` version don't modify the initial state in work_wires."""
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            # Initialize the work wires to a non-zero state
            for ind, wire in enumerate(work_wires):
                qp.RX(ind, wires=wire)

            for wire in control_wires:
                qp.Hadamard(wires=wire)

            qp.QROM(data, control_wires, target_wires, work_wires)

            for ind, wire in enumerate(work_wires):
                qp.RX(-ind, wires=wire)

            return qp.probs(wires=work_wires)

        assert np.isclose(circuit()[0], 1.0)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        qrom_decomposition = qp.QROM(
            [[1], [0], [0], [1]],
            control_wires=[0, 1],
            target_wires=[2],
            work_wires=[3],
            clean=True,
        ).decomposition()

        expected_gates = [
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Select(
                ops=(
                    qp.BasisEmbedding(1, wires=[2]) @ qp.BasisEmbedding(0, wires=[3]),
                    qp.BasisEmbedding(0, wires=[2]) @ qp.BasisEmbedding(1, wires=[3]),
                ),
                control=[0],
            ),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Hadamard(wires=[2]),
            qp.CSWAP(wires=[1, 2, 3]),
            qp.Select(
                ops=(
                    qp.BasisEmbedding(1, wires=[2]) @ qp.BasisEmbedding(0, wires=[3]),
                    qp.BasisEmbedding(0, wires=[2]) @ qp.BasisEmbedding(1, wires=[3]),
                ),
                control=0,
            ),
            qp.CSWAP(wires=[1, 2, 3]),
        ]

        for op1, op2 in zip(qrom_decomposition, expected_gates):
            qp.assert_equal(op1, op2)

    @pytest.mark.parametrize(
        ("data", "control_wires", "target_wires", "work_wires", "clean"),
        [
            (
                [[1, 1], [0, 1], [0, 0], [1, 0]],
                [2, 3],
                [0, 1],
                [4, 5, 6, 7, 8, 9],
                True,
            ),
            ([[1], [0], [0], [1]], [0, 1], [2], [3], True),
            ([[1]], [], [0], [1], True),
            (
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                ],
                [0, 1, 2],
                [3, 4],
                [5],
                False,
            ),
            (
                [
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                ],
                [0, 1, 2],
                [3, 4],
                [5],
                True,
            ),
            ([[1], [0], [0], [1]], [0, 1], [2], [], False),
            ([[1], [0], [0], [1]], [0, 1], [2], [3, 4], False),
        ],  # pylint: disable=too-many-arguments
    )
    def test_decomposition_new(
        self, data, control_wires, target_wires, work_wires, clean
    ):  # pylint: disable=too-many-arguments
        """Tests the decomposition rule implemented with the new system."""
        op = qp.QROM(
            data,
            control_wires=control_wires,
            target_wires=target_wires,
            work_wires=work_wires,
            clean=clean,
        )
        for rule in qp.list_decomps(qp.QROM):
            _test_decomposition_rule(op, rule)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_select_decomposition_unary(self):
        """Tests that _select_decomp_unary is actually invoked within QROM decomposition."""

        bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]
        control_wires = [0, 1, 2]
        target_wires = [3, 4]

        class SpyRule(DecompositionRule):
            """Wraps a DecompositionRule, tracking __call__ invocations."""

            def __init__(self, original):  # pylint: disable=super-init-not-called
                self._original = original
                self.call_count = 0

            def __call__(self, *args, **kwargs):
                self.call_count += 1
                return self._original(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._original, name)

        spy = SpyRule(_select_decomp_unary)

        @qp.transforms.decompose(
            gate_set={"TemporaryAND", "Adjoint(TemporaryAND)", *qp.ops.__all__},
            fixed_decomps={qp.QROM: _qrom_decomposition, qp.Select: spy},
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.QROM(bitstrings, control_wires, target_wires, work_wires=[5, 6])
            return qp.state()

        circuit()
        assert spy.call_count > 0, "_select_decomp_unary was never called"

    def test_zero_control_wires(self):
        """Test that the edge case of zero control wires works"""

        dev = qp.device("default.qubit", wires=2)
        qs = qp.tape.QuantumScript(
            qp.QROM.compute_decomposition(
                ((1, 0),),
                target_wires=[0, 1],
                work_wires=None,
                control_wires=[],
                clean=False,
            ),
            [qp.probs(wires=[0, 1])],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert len(tape[0][0].operations) == 1
        assert qp.equal(tape[0][0][0], qp.BasisEmbedding([1, 0], wires=[0, 1]))
        assert qp.math.allclose(output, [0, 0, 1, 0])

    @pytest.mark.jax
    def test_traced_wires(self):
        """Test that QROM construction and decomposition do not raise TracerBoolConversionError
        when wires are JAX tracers."""

        import jax

        jax.config.update("jax_enable_x64", True)

        def build_and_decompose(data, control_wires, target_wires, work_wires):
            op = qp.QROM(data, control_wires, target_wires, work_wires)
            op.decomposition()
            return True

        n, m, w = 2, 2, 1
        data = jnp.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        control_wires = jnp.arange(0, n)
        target_wires = jnp.arange(n, n + m)
        work_wires = jnp.arange(n + m, n + m + w)

        jax.make_jaxpr(build_and_decompose)(data, control_wires, target_wires, work_wires)


@pytest.mark.parametrize(
    ("control_wires", "target_wires", "work_wires", "msg_match"),
    [
        (
            [0, 1, 2],
            [0, 3],
            [4, 5],
            "Target wires should be different from control wires.",
        ),
        (
            [0, 1, 2],
            [4],
            [2, 5],
            "Control wires should be different from work wires.",
        ),
        (
            [0, 1, 2],
            [4],
            [4],
            "Target wires should be different from work wires.",
        ),
    ],
)
def test_wires_error(control_wires, target_wires, work_wires, msg_match):
    """Test an error is raised when a control wire is in one of the ops"""
    with pytest.raises(ValueError, match=msg_match):
        qp.QROM([[1]] * 8, control_wires, target_wires, work_wires)


def test_repr():
    """Test that the __repr__ method works as expected."""

    op = qp.QROM(
        [[1], [0], [0], [1]],
        control_wires=[0, 1],
        target_wires=[2],
        work_wires=[3],
        clean=True,
    )
    res = repr(op)
    expected = "QROM(control_wires=[0, 1], target_wires=[2],  work_wires=[3], clean=True)"
    assert res == expected


@pytest.mark.parametrize(
    ("data", "control_wires", "target_wires", "msg_match"),
    [
        (
            [[1], [0], [0], [1]],
            [0],
            [2],
            r"Not enough control wires \(1\) for the desired number of data \(4\). At least 2 control wires are required.",
        ),
        (
            [[1], [0], [0], [1]],
            [0, 1],
            [2, 3],
            r"Bitstring length must match the number of target wires.",
        ),
    ],
)
def test_wrong_wires_error(data, control_wires, target_wires, msg_match):
    """Test that error is raised if more ops are requested than can fit in control wires"""
    with pytest.raises(ValueError, match=msg_match):
        qp.QROM(data, control_wires, target_wires, work_wires=None)


def test_none_work_wires_case():
    """Test that clean version is not applied if work wires are not used"""

    gates_clean = qp.QROM.compute_decomposition(
        np.array([[1], [0], [0], [1]]), [0, 1], [2], [], clean=True
    )
    expected_gates = qp.QROM.compute_decomposition(
        np.array([[1], [0], [0], [1]]), [0, 1], [2], [], clean=False
    )

    assert gates_clean == expected_gates


def test_too_many_work_wires_case():
    """Test that QROM works when more work wires are given than necessary"""

    gates_clean = qp.QROM.compute_decomposition(
        np.array([[1], [0], [0], [1]]), [0, 1], [2], [3, 4, 5], clean=False
    )
    expected_gates = qp.QROM.compute_decomposition(
        np.array([[1], [0], [0], [1]]),
        [0, 1],
        [2],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        clean=False,
    )

    assert gates_clean == expected_gates


@pytest.mark.parametrize(
    ("terms", "n_ctrl", "n_target", "n_work", "expected"),
    [
        (16, 4, 1, 3, 2),
        (16, 4, 10, 5, 5),
        (7, 3, 2, 2, 2),
        (14, 4, 2, 10, 4),
        (256, 8, 2, 10, 8),
        (4, 2, 1, 1, 0),
    ],
)
def test_calculate_n_select_work_wires(terms, n_ctrl, n_target, n_work, expected):
    """Test the allocation logic for Select vs Swap work wires."""

    result = _calculate_n_select_work_wires(
        terms=terms, num_control_wires=n_ctrl, num_target_wires=n_target, num_work_wires=n_work
    )

    assert result == expected


class TestMeasurementQROM:
    """Test the correctness of the measurement-based QROM decomposition."""

    @pytest.mark.parametrize(
        ("L", "expected_ands"),
        [
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 1),
            (5, 3),
            (6, 4),
            (7, 4),
            (8, 5),
            (9, 7),
            (10, 8),
            (11, 9),
            (12, 10),
            (13, 10),
            (14, 11),
            (15, 12),
            (16, 13),
        ],
    )
    def test_count_TemporaryAnd(self, L, expected_ands):
        """Test that TemporaryAND count matches expected values."""
        assert _count_tempAND_in_measurement_qrom(L) == expected_ands

    def test_resources_small_cases(self):
        """Test resource estimates for the L <= 1 and L == 2 edge cases."""
        res_one = _qrom_measurement_resources(num_bitstrings=1, num_target_wires=3)
        assert res_one[qp.resource_rep(qp.BasisState, num_wires=3)] == 1

        res_two = _qrom_measurement_resources(num_bitstrings=2, num_target_wires=3)
        assert res_two[qp.resource_rep(qp.BasisState, num_wires=3)] == 1
        assert res_two[qp.CNOT] == 3

    def test_resources_general_case(self):
        """Test that the general resource estimate contains the expected gate types."""
        res = _qrom_measurement_resources(num_bitstrings=8, num_target_wires=3)
        assert res[PauliMeasure] > 0
        assert res[qp.CZ] > 0

    def test_resources_from_base_params(self):
        """Test that resources are extracted from ``base_params`` (Adjoint path)."""
        base_params = {"num_bitstrings": 8, "num_target_wires": 3}
        res_direct = _qrom_measurement_resources(num_bitstrings=8, num_target_wires=3)
        res_base = _qrom_measurement_resources(base_params=base_params)
        assert res_base == res_direct

    def test_condition_without_compiler(self):
        """Test that the measurement decomposition is disabled without an active compiler."""
        assert (
            _qrom_measurement_condition(num_bitstrings=8, num_work_wires=2, num_control_wires=3)
            is False
        )

    def test_condition_with_compiler(self, mocker):
        """Test the condition logic when a compiler is active."""
        mocker.patch("pennylane.templates.subroutines.qrom.compiler.active", return_value=True)

        # Small tables (<= 2 bitstrings) are always applicable.
        assert (
            _qrom_measurement_condition(num_bitstrings=2, num_work_wires=0, num_control_wires=1)
            is True
        )
        # Enough work wires: applicable.
        assert (
            _qrom_measurement_condition(num_bitstrings=8, num_work_wires=2, num_control_wires=3)
            is True
        )
        # Too few work wires: not applicable.
        assert (
            _qrom_measurement_condition(num_bitstrings=8, num_work_wires=0, num_control_wires=3)
            is False
        )
        # Parameters extracted from ``base_params`` (Adjoint path).
        assert (
            _qrom_measurement_condition(
                base_params={
                    "num_bitstrings": 8,
                    "num_work_wires": 2,
                    "num_control_wires": 3,
                }
            )
            is True
        )

    def test_decomposition_single_bitstring(self):
        """Test the L == 1 branch of the measurement decomposition."""
        with qp.queuing.AnnotatedQueue() as q:
            _qrom_measurement_decomposition(
                data=np.array([[1, 0, 1]]),
                control_wires=[0],
                target_wires=[1, 2, 3],
                work_wires=[],
            )
        ops = q.queue
        assert len(ops) == 1
        assert isinstance(ops[0], qp.BasisState)
        assert ops[0].wires == qp.wires.Wires([1, 2, 3])
        assert np.array_equal(ops[0].data[0], np.array([1, 0, 1]))

    def test_decomposition_two_bitstrings(self):
        """Test the L == 2 branch of the measurement decomposition."""
        with qp.queuing.AnnotatedQueue() as q:
            _qrom_measurement_decomposition(
                data=np.array([[1, 0, 1], [0, 0, 1]]),
                control_wires=[0],
                target_wires=[1, 2, 3],
                work_wires=[],
            )
        ops = q.queue
        assert isinstance(ops[0], qp.BasisState)
        # Only the differing bit (index 0) produces a controlled load.
        assert all(isinstance(op, qp.CNOT) for op in ops[1:])
        assert len(ops[1:]) == 1

    def test_decomposition_from_base_operator(self):
        """Test that the decomposition extracts arguments from ``base`` (Adjoint path)."""
        op = qp.QROM(
            [[1], [0], [0], [1]],
            control_wires=[0, 1],
            target_wires=[2],
            work_wires=[3],
        )
        with qp.queuing.AnnotatedQueue() as q_base:
            _qrom_measurement_decomposition(base=op)
        with qp.queuing.AnnotatedQueue() as q_direct:
            _qrom_measurement_decomposition(
                data=op.data[0],
                control_wires=op.control_wires,
                target_wires=op.target_wires,
                work_wires=op.work_wires,
            )
        assert len(q_base.queue) == len(q_direct.queue)
        for op_base, op_direct in zip(q_base.queue, q_direct.queue):
            assert type(op_base) is type(op_direct)
            assert op_base.wires == op_direct.wires

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "L",
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16],
    )
    def test_correctness(self, L, seed):
        """Test correctness of measurement-based QROM for various sizes."""
        rng = np.random.default_rng(seed)
        n_target = 3
        n_input = math.ceil(math.log2(L))
        n_work = n_input - 1

        bitstrings = rng.choice(2, size=(L, n_target))

        total_wires = n_input + n_work + n_target
        dev = qp.device("lightning.qubit", wires=total_wires)

        wires = qp.registers(
            {"control_wires": n_input, "work_wires": n_work, "target_wires": n_target}
        )

        shots = 10

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=clifford_t_measure)
        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit(j):
            qp.BasisState(j, wires=wires["control_wires"])
            _qrom_measurement_decomposition(data=bitstrings, **wires, clean=True)
            return qp.sample(wires=wires["target_wires"]), qp.sample(wires=wires["work_wires"])

        for j in range(L):
            target_samples, work_samples = circuit(j)

            assert target_samples.shape == (shots, n_target)
            assert np.all(
                target_samples == bitstrings[j]
            ), f"L={L}, j={j}: got {target_samples}, expected {bitstrings[j]} (x{shots})"
            assert np.allclose(work_samples, 0), f"j={j}: work wires not clean, got {work_samples}"

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "L", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    )
    def test_no_phases_error(self, L):
        """Test that QROM introduces no relative phases on the control register."""
        rng = np.random.default_rng(42 + L)
        n_target = 3

        n_input = max(1, math.ceil(math.log2(L)))
        n_work = max(1, n_input - 1)

        bitstrings = rng.integers(0, 2, size=(L, n_target)).tolist()

        total_wires = n_input + n_work + n_target
        dev = qp.device("lightning.qubit", wires=total_wires)

        control_wires = list(range(n_input))
        work_wires = list(range(n_input, n_input + n_work))
        target_wires = list(range(n_input + n_work, total_wires))

        x_state = rng.random(L) + 1j * rng.random(L)
        x_state /= np.linalg.norm(x_state)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=clifford_t_measure)
        @qp.qnode(dev)
        def circuit():
            qp.StatePrep(x_state, wires=control_wires, pad_with=0.0)

            # Put target wires in |+> state, so that QROM should act like the identity overall.
            for wire in target_wires:
                qp.Hadamard(wire)

            _qrom_measurement_decomposition(
                data=bitstrings,
                control_wires=control_wires,
                target_wires=target_wires,
                work_wires=work_wires,
                clean=True,
            )

            qp.adjoint(qp.StatePrep(x_state, wires=control_wires, pad_with=0.0))
            return qp.probs(wires=control_wires), qp.probs(wires=work_wires)

        assert np.isclose(circuit()[0][0], 1.0)
        assert np.isclose(circuit()[1][0], 1.0)

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "L",
        [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    )
    def test_out_of_range_maps_to_identity(self, L, seed):
        """Non-partial QROM: indices j in [L, 2**c) must map to the identity.

        For a non-power-of-two number of bitstrings, the unary iterator fires no
        operator on the out-of-range control states. Those states must therefore
        leave the target register untouched (it stays in |0>) and the work wires
        clean. This is exactly the property that distinguishes the non-partial
        decomposition from the partial one.
        """
        rng = np.random.default_rng(seed)
        n_target = 3
        n_input = math.ceil(math.log2(L))
        n_work = max(1, n_input - 1)

        # Sanity: there must actually be out-of-range states to test.
        assert L < 2**n_input, f"L={L} is a power of two; no out-of-range states"

        bitstrings = rng.choice(2, size=(L, n_target))

        total_wires = n_input + n_work + n_target
        dev = qp.device("lightning.qubit", wires=total_wires)

        wires = qp.registers(
            {"control_wires": n_input, "work_wires": n_work, "target_wires": n_target}
        )

        shots = 10

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=clifford_t_measure)
        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit(j):
            qp.BasisState(j, wires=wires["control_wires"])
            _qrom_measurement_decomposition(data=bitstrings, **wires, clean=True)
            return qp.sample(wires=wires["target_wires"]), qp.sample(wires=wires["work_wires"])

        for j in range(L, 2**n_input):
            target_samples, work_samples = circuit(j)

            assert target_samples.shape == (shots, n_target)
            assert np.allclose(
                target_samples, 0
            ), f"L={L}, out-of-range j={j}: target not identity, got {target_samples}"
            assert np.allclose(
                work_samples, 0
            ), f"L={L}, out-of-range j={j}: work wires not clean, got {work_samples}"
