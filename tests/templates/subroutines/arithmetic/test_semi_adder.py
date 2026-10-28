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
Tests for the SemiAdder template.
"""

from collections import Counter
from functools import partial

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.core.operator import abstractify
from pennylane.ops import CNOT, adjoint
from pennylane.templates.subroutines.arithmetic.semi_adder import (
    _controlled_semi_adder,
    _controlled_semi_adder_ppm,
    _controlled_semi_adder_ppm_condition,
    _controlled_semi_adder_ppm_work_wires,
    _effective_skip_input_pos,
    _semi_adder,
    _semi_adder_ppm,
    _semi_adder_ppm_condition,
    _semi_adder_ppm_work_wires,
    _semi_adder_resources,
)
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.testing import assert_decomposable
from pennylane.typing import Wire
from pennylane.wires import Wires


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_standard_validity_SemiAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1, 2]
    y_wires = [3, 4, 5]
    work_wires = [6, 7]
    op = qp.SemiAdder(x_wires, y_wires, work_wires)
    # PPM rules contain nested measurement branches, tested with Catalyst below.
    with (
        qp.exclude_decomps(qp.SemiAdder, "_semi_adder_ppm"),
        qp.exclude_decomps("C(SemiAdder)", "flip_zero_ctrl_values(_controlled_semi_adder_ppm)"),
    ):
        qp.testing.assert_valid(op)


class TestSemiAdder:
    """Test the qp.SemiAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "x", "y"),
        [
            ([0], [1], None, 1, 0),
            ([0], [1], None, 1, 1),
            ([0, 1], [2], None, 0, 1),
            ([0, 1], [2], None, 1, 1),
            ([0, 1], [2], None, 1, 0),
            ([0, 1], [2, 3], [4], 2, 0),
            ([0, 1], [2, 3], [4], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 1, 2),
            ([0, 1, 2], [3, 4, 5], [6, 7], 5, 6),
            ([0, 1], [2, 3, 4], [5, 6], 3, 2),
            ([0, 1], [2, 3, 4, 5], [6, 7, 8], 3, 10),
            ([0, 1, 2], [3, 4, 5], [6, 7], 7, 7),
            ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9], 6, 5),
            ([0], [3, 4, 5, 6], [7, 8, 9], 1, 5),
            ([0, 1, 2, 3, 4], [5, 6], [7], 11, 2),
            # work_wires are optional: the missing ones are allocated dynamically
            ([0, 1], [2, 3, 4], [5], 3, 2),
            ([0, 1, 2], [3, 4, 5], [6], 5, 6),
            ([0, 1], [2, 3], None, 2, 0),
            ([0, 1], [2, 3, 4], None, 3, 2),
            ([0, 1, 2], [3, 4, 5], None, 5, 6),
            ([0, 1, 2, 3, 4], [5, 6], None, 11, 2),
            (["a", "b", "d"], ["e", "h", "p"], None, 4, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z"], 4, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z", "u", "q"], 4, 2),
            (["a", "b", "d"], ["e", "h", "p"], ["f", "z", "u", "q", "v"], 4, 2),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, work_wires, x, y
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the SemiAdder template output."""
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(x, y):
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            y_bin = qp.math.int_to_binary(y, len(y_wires))
            qp.BasisEmbedding(x_bin, wires=x_wires)
            qp.BasisEmbedding(y_bin, wires=y_wires)
            qp.SemiAdder(x_wires, y_wires, work_wires)
            return qp.sample(wires=y_wires), qp.probs(wires=work_wires)

        output = circuit(x, y)

        sample = output[0]

        #  check that the output sample is the binary representation of x + y mod 2^len(y_wires)
        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(sample[0, :]))),
            (x + y) % 2 ** len(y_wires),
        )

        if work_wires:
            # check work_wires are in state |0>
            assert np.isclose(output[1][0], 1.0)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [1, 6],
                "x_wires and work_wires must not overlap",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [3, 6],
                "y_wires and work_wires must not overlap",
            ),
            (
                [1],
                [0],
                [0],
                "y_wires and work_wires must not overlap",
            ),
            (
                [0, 1, 2],
                [2, 3, 4, 5],
                [6, 7, 8],
                "x_wires and y_wires must not overlap",
            ),
        ],
    )
    def test_wires_error(
        self, x_wires, y_wires, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            qp.SemiAdder(x_wires, y_wires, work_wires)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""
        x_wires, y_wires, work_wires = (
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13],
        )

        adder_decomposition = qp.SemiAdder(x_wires, y_wires, work_wires).compute_decomposition(
            x_wires, y_wires, work_wires
        )

        names = [op.name for op in adder_decomposition]

        # Example in Fig 1.  https://arxiv.org/pdf/1709.06648
        assert names.count("TemporaryAND") == 4
        assert names.count("Adjoint(TemporaryAND)") == 4
        assert names.count("CNOT") == 21

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("work_wires", [[9, 10, 11], None])
    @pytest.mark.parametrize(("x_wires"), [[0, 1, 2], [0, 1], [0, 1, 2, 3]])
    def test_decomposition_rule(self, x_wires, work_wires):
        """Tests that SemiAdder is decomposed properly."""

        # The generic tape validator does not flatten nested measurement branches.
        with qp.exclude_decomps(qp.SemiAdder, "_semi_adder_ppm"):
            assert_decomposable(qp.SemiAdder(x_wires, [5, 6, 7, 8], work_wires))

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "wire_lens",
        [
            {"x_wires": 3, "y_wires": 3, "work_wires": 2},  # all work wires provided
            {"x_wires": 3, "y_wires": 4, "work_wires": 1},  # some work wires allocated
            {"x_wires": 3, "y_wires": 3, "work_wires": 0},  # all work wires allocated
            {"x_wires": 2, "y_wires": 1, "work_wires": 0},  # single y wire
        ],
    )
    def test_decomposition_rule_capture_dynamic_wires(self, wire_lens):
        """Test that the decomposition rules of SemiAdder can be captured with dynamic wires."""
        import jax

        registers = qp.registers(wire_lens)
        kwargs = {name: qp.math.array(wires, like="jax") for name, wires in registers.items()}
        for rule in qp.list_decomps(qp.SemiAdder):
            if not rule.is_applicable(**kwargs):
                continue
            # pylint: disable-next=protected-access
            jax.make_jaxpr(qp.capture.subroutine(rule._impl))(**kwargs)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        x_wires = [0, 1, 4]
        y_wires = [2, 3, 5]
        work_wires = [7, 8]
        dev = qp.device("default.qubit")

        @jax.jit
        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            y_bin = qp.math.int_to_binary(y, len(y_wires))
            qp.BasisEmbedding(x_bin, wires=x_wires)
            qp.BasisEmbedding(y_bin, wires=y_wires)
            qp.SemiAdder(x_wires, y_wires, work_wires)
            return qp.sample(wires=y_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))),
            (x + y) % 2 ** len(y_wires),
        )

    @pytest.mark.parametrize(
        (
            "x_wires",
            "y_wires",
            "work_wires",
            "control_wires",
            "x_value",
            "y_value",
            "control_values",
        ),
        [
            ([1, 2], [3, 4], [5, 6, 7], [0], 3, 0, [0]),
            ([3], [0, 1, 2], [6, 7], [8], 1, 0, [1]),
            ([3], [0, 1, 2], [6, 7, 9], [8], 1, 0, [1]),
            ([0, 1, 2], [3], [6], [8], 1, 0, [0]),
            ([0, 1, 2], [3], [6], [8], 4, 1, [1]),
            ([0, 1, 2], [3, 4], [6], [8], 1, 1, [1]),
            ([0, 1, 2], [3, 4, 5], [6, 7], [8], 2, 2, [1]),
            ([0, 1, 2], [3, 4, 5], [6, 7], [8], 3, 7, [1]),
            ([0, 1, 2, 3], [4, 5, 6], [7, 8], [10], 2, 2, [1]),
            ([0, 1, 2, 3], [4, 5, 6], [7, 8], [10], 2, 2, [0]),
            ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11], 3, 4, [1]),
            ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11], 3, 4, [0]),
            ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 12, 13], [11], 3, 4, [0]),
            ([0], [1], None, [2], 1, 1, [1]),
            ([0], [1], None, [2], 1, 1, [0]),
            # work_wires are optional: when not provided, they are allocated dynamically
            ([0, 1], [2, 3], None, [4], 3, 1, [1]),
            ([0, 1], [2, 3], None, [4], 3, 1, [0]),
            ([0, 1, 2], [3, 4, 5], None, [6], 5, 6, [1]),
            ([0, 1, 2], [3, 4, 5], None, [6, 7], 5, 6, [1, 0]),
            ([1, 2], [3, 4], [5, 6, 7], [0, 8], 3, 0, [0, 1]),
            ([3], [0, 1, 2], [6, 7], [8, 4], 1, 0, [1, 1]),
            ([3], [0, 1, 2], [6, 7, 9, 10], [8, 4, 5], 1, 0, [1, 0, 1]),
            ([0, 1, 2], [3], [6, 9, 10], [8, 4, 5], 1, 0, [0, 0, 0]),
            ([0, 1, 2], [3], [6], [8, 9], 1, 1, [1, 1]),
            ([0, 1, 2], [3, 4], [6], [8, 9], 7, 1, [1, 0]),
            ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9], 2, 2, [1, 0]),
            ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9], 3, 7, [1, 1]),
            ([0], [1], None, [2, 3], 1, 1, [1, 0]),
            ([0], [1], None, [2, 3], 1, 1, [0, 1]),
        ],
    )
    @pytest.mark.parametrize("split_work_wires", [True, False])
    def test_controlled_decomposition(
        self,
        x_wires,
        y_wires,
        work_wires,
        control_wires,
        x_value,
        y_value,
        control_values,
        split_work_wires,
    ):  # pylint: disable=too-many-arguments
        """Test correctness of C(SemiAdder) decomposition"""

        dev = qp.device("default.qubit")
        m = len(y_wires)
        k = len(control_wires)

        if split_work_wires and work_wires is not None:
            work_wires_adder = work_wires[: m - 1]
            work_wires_ctrl = work_wires[m - 1 :]
        else:
            work_wires_adder = work_wires
            work_wires_ctrl = None

        op = qp.SemiAdder(x_wires, y_wires, work_wires_adder)

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(c_value):
            x_value_bin = qp.math.int_to_binary(x_value, len(x_wires))
            y_value_bin = qp.math.int_to_binary(y_value, len(y_wires))
            c_value_bin = qp.math.int_to_binary(c_value, len(control_wires))
            qp.BasisState(x_value_bin, x_wires)
            qp.BasisState(y_value_bin, y_wires)
            qp.BasisState(c_value_bin, control_wires)
            _controlled_semi_adder(
                op, control_wires, control_values, work_wires_ctrl, work_wire_type="zeroed"
            )
            return (
                qp.counts(wires=x_wires),
                qp.counts(wires=y_wires),
                qp.counts(wires=control_wires),
                qp.counts(wires=work_wires),
            )

        activating_c_value = 2 ** np.arange(k - 1, -1, -1) @ control_values
        for c_value in range(2**k):
            output = circuit(c_value)
            output = tuple(int(list(out.keys())[0], 2) for out in output)
            expected_y_out = (
                (x_value + y_value) % 2 ** len(y_wires)
                if c_value == activating_c_value
                else y_value
            )
            expected_output = (x_value, expected_y_out, c_value, 0)
            # If work_wires=None, counts(work_wires) is not what we want it to be
            if work_wires is None:
                assert output[:-1] == expected_output[:-1]
            else:
                assert output == expected_output

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "wire_lens",
        [
            {"control": 1, "x_wires": 3, "y_wires": 3, "work_wires": 2},  # all work wires provided
            {
                "control": 2,
                "x_wires": 3,
                "y_wires": 4,
                "work_wires": 1,
            },  # some work wires allocated
            {"control": 3, "x_wires": 3, "y_wires": 3, "work_wires": 0},  # all work wires allocated
            {"control": 2, "x_wires": 2, "y_wires": 1, "work_wires": 0},  # single y wire
        ],
    )
    def test_ctrl_decomposition_rule_capture_dynamic_wires(self, wire_lens):
        """Test that the decomposition rules of C(SemiAdder) can be captured with dynamic wires."""
        import jax
        from jax import numpy as jnp

        registers = qp.registers(wire_lens)
        control_wires = qp.math.array(registers.pop("control"), like="jax")
        base = qp.SemiAdder(**registers)
        decomp_args = {
            "base": base,
            "control_wires": control_wires,
            "control_values": jnp.zeros(len(control_wires)),
            "work_wires": jnp.array([]),
        }
        for rule in qp.list_decomps("C(SemiAdder)"):
            if not rule.is_applicable(**decomp_args):
                continue
            # pylint: disable-next=protected-access
            subroutine = qp.capture.subroutine(partial(rule._impl, work_wire_type="borrowed"))
            jax.make_jaxpr(subroutine)(**decomp_args)


class TestSemiAdderSkipInputPos:
    """Independent tests for the ``skip_input_pos`` feature of ``_semi_adder``."""

    @staticmethod
    def _sparse_addend(x, num_x_wires, num_y_wires, skip_input_pos):
        """Integer added to ``y`` when ``skip_input_pos`` zeros some input bit slots."""
        skip = _effective_skip_input_pos(num_x_wires, num_y_wires, skip_input_pos)
        used_x = 0
        addend = 0
        for i in range(num_y_wires):
            if i in skip:
                continue
            addend |= ((x >> used_x) & 1) << i
            used_x += 1
        return addend

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "work_wires", "x", "y", "skip_input_pos"),
        [
            ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9], 5, 3, [1]),
            ([0, 1, 2], [3, 4, 5], [6, 7], 7, 1, [1]),
            ([0, 1], [2, 3, 4, 5], [6, 7, 8], 3, 2, [1, 2]),
            ([0, 1], [2, 3, 4], [5, 6], 3, 4, [2]),
            ([0, 1], [2, 3, 4, 5], [6, 7, 8], 3, 5, [1, 3]),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, work_wires, x, y, skip_input_pos
    ):  # pylint: disable=too-many-arguments
        """Test that skipped input positions inject classical zeros into the addend."""
        n = len(x_wires)
        m = len(y_wires)
        expected = (y + self._sparse_addend(x, n, m, skip_input_pos)) % (2**m)
        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit():
            qp.BasisEmbedding(qp.math.int_to_binary(x, n), wires=x_wires)
            qp.BasisEmbedding(qp.math.int_to_binary(y, m), wires=y_wires)
            _semi_adder(x_wires, y_wires, work_wires, skip_input_pos=skip_input_pos)
            return qp.sample(wires=y_wires), qp.probs(wires=work_wires)

        sample, work_probs = circuit()
        int_sample = 2 ** np.arange(len(sample[0, :])) @ sample[0, ::-1]
        assert np.allclose(int_sample, expected)
        assert np.isclose(work_probs[0], 1.0)

    def test_resources_match_decomposition(self):
        """Test that ``_semi_adder_resources`` matches the queued ``skip_input_pos``
        decomposition."""
        x_wires, y_wires, work_wires = [0, 1, 2], [3, 4, 5, 6], [7, 8, 9]
        skip_input_pos = [1, 2]

        with qp.queuing.AnnotatedQueue() as q:
            _semi_adder(x_wires, y_wires, work_wires, skip_input_pos=skip_input_pos)

        names = [op.name for op in qp.tape.QuantumScript.from_queue(q).operations]
        resources = _semi_adder_resources(
            x_wires, y_wires, work_wires, skip_input_pos=skip_input_pos
        )

        assert names.count("TemporaryAND") == resources[TemporaryAND]
        assert names.count("Adjoint(TemporaryAND)") == resources[adjoint(TemporaryAND(Wire[3]))]
        assert names.count("CNOT") == resources[CNOT]


class TestControlledSemiAdderPPM:
    """Test the Pauli-product decomposition of a coherently controlled SemiAdder."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("compiler_active", [False, True])
    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    def test_condition(self, compiler_active, num_controls, mocker):
        """The specialized PPM rule needs a compiler and at least one control."""
        mocker.patch(
            "pennylane.templates.subroutines.arithmetic.semi_adder.compiler.active",
            return_value=compiler_active,
        )
        base = qp.SemiAdder(Wire[2], Wire[3], Wire[5])
        assert _controlled_semi_adder_ppm_condition(base, Wire[num_controls]) is (
            (compiler_active or qp.capture.enabled()) and num_controls >= 1
        )

    @pytest.mark.parametrize(
        "case",
        [
            (1, 1, 0, 0, "borrowed", 3),
            (1, 3, 0, 0, "zeroed", 5),
            (1, 3, 2, 1, "zeroed", 2),
            (1, 3, 2, 3, "zeroed", 0),
            (1, 3, 2, 3, "borrowed", 3),
            (1, 3, 7, 3, "borrowed", 0),
            (2, 1, 0, 0, "zeroed", 4),
            (2, 3, 2, 3, "zeroed", 1),
            (2, 3, 2, 3, "borrowed", 4),
            (3, 3, 2, 5, "zeroed", 0),
            (3, 3, 7, 3, "borrowed", 0),
        ],
    )
    def test_work_wires(self, case):
        """Borrowed control work wires cannot replace the clean PPM work wires."""
        num_controls, num_y, num_base, num_extra, work_wire_type, expected = case
        base = qp.SemiAdder(Wire[2], Wire[num_y], Wire[num_base])
        spec = _controlled_semi_adder_ppm_work_wires(
            base=base,
            control_wires=Wire[num_controls],
            work_wires=Wire[num_extra],
            work_wire_type=work_wire_type,
        )
        assert spec == {"zeroed": expected}

    @pytest.mark.parametrize("num_y", [1, 2, 5])
    @pytest.mark.parametrize("num_controls", [1, 2, 3])
    def test_resources(self, num_y, num_controls):
        """Count the carry and controlled-sum rotations and worst-case measurements."""
        rule = _controlled_semi_adder_ppm
        base = qp.SemiAdder(Wire[2], Wire[num_y], Wire[num_y + num_controls + 1])
        resources = rule.compute_resources(
            base,
            Wire[num_controls],
            control_values=[True] * num_controls,
            work_wires=Wire[0],
            work_wire_type="borrowed",
        )
        assert rule.exact_resources is False
        assert (
            sum(count for op, count in resources.gate_counts.items() if isinstance(op, qp.PPR_8))
            == 8 * num_y + 4 * num_controls - 8
        )
        assert (
            sum(
                count
                for op, count in resources.gate_counts.items()
                if isinstance(op, qp.ops.PauliMeasure)
            )
            == 14 * num_y + 4 * num_controls - 10
        )
        assert resources.gate_counts[qp.X(Wire[1])] == 5 * num_y + num_controls - 3

    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    @pytest.mark.parametrize("num_y", [1, 2, 4])
    def test_resources_bound_all_corrections(self, num_controls, num_y, mocker):
        """The resource bound includes every correction, including nested branches."""

        def record_measurement(pauli_word, wires):
            qp.ops.PauliMeasure(pauli_word, wires)
            return 1

        module = "pennylane.templates.subroutines.arithmetic.semi_adder"
        mocker.patch(f"{module}.pauli_measure", side_effect=record_measurement)
        # Exercise every correction independently to check the declared upper bound.
        mocker.patch(f"{module}.cond", side_effect=lambda _, correction, *args: correction)
        x_wires = Wires(range(num_controls, num_controls + num_y))
        y_wires = Wires(range(num_controls + num_y, num_controls + 2 * num_y))
        work_wires = Wires(range(num_controls + 2 * num_y, 2 * num_controls + 3 * num_y + 1))
        base = qp.SemiAdder(x_wires, y_wires, work_wires)
        rule = _controlled_semi_adder_ppm if num_controls else _semi_adder_ppm
        arguments = (
            {"base": base, "control_wires": Wires(range(num_controls))}
            if num_controls
            else {"x_wires": x_wires, "y_wires": y_wires, "work_wires": work_wires}
        )

        with qp.queuing.AnnotatedQueue() as queue:
            rule(**arguments)

        counts = Counter(abstractify(op) for op in queue.queue)
        assert counts == rule.compute_resources(**arguments).gate_counts

    @pytest.mark.capture
    @pytest.mark.parametrize(
        "case",
        [
            (1, 0, 0, "borrowed", 4),
            (1, 1, 2, "zeroed", 1),
            (1, 1, 2, "borrowed", 3),
            (1, 4, 0, "zeroed", 0),
            (2, 0, 0, "zeroed", 5),
            (2, 1, 2, "borrowed", 4),
            (3, 1, 2, "zeroed", 3),
            (3, 6, 0, "zeroed", 0),
        ],
    )
    def test_allocation(self, case):
        """Capture allocates only the missing clean wires and marks them restored."""
        import jax

        num_controls, num_base, num_extra, work_wire_type, expected = case
        base = qp.SemiAdder(
            range(num_controls, num_controls + 2),
            range(num_controls + 2, num_controls + 4),
            list(range(num_controls + 4, num_controls + 4 + num_base)),
        )
        extra = list(range(num_controls + 4 + num_base, num_controls + 4 + num_base + num_extra))
        rule = partial(_controlled_semi_adder_ppm, work_wire_type=work_wire_type)
        jaxpr = jax.make_jaxpr(qp.capture.subroutine(rule))(
            base=base, control_wires=Wires(range(num_controls)), work_wires=extra
        )
        eqns = jaxpr.eqns[0].params["jaxpr"].eqns
        allocations = [eqn for eqn in eqns if eqn.primitive.name == "allocate"]
        deallocations = [eqn for eqn in eqns if eqn.primitive.name == "deallocate"]
        if expected:
            assert len(allocations) == len(deallocations) == 1
            assert allocations[0].params["num_wires"] == expected
            assert allocations[0].params["restored"] is True
        else:
            assert not allocations
            assert not deallocations

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        ("num_x", "num_y", "control_values", "work_layout"),
        [
            (1, 1, [True], "base"),
            (1, 3, [False], "base"),
            (3, 2, [True], "base"),
            (2, 3, [True], "base"),
            (3, 3, [False], "base"),
            (2, 2, [False], "split"),
            (2, 2, [True], "control"),
            (2, 2, [True], "borrowed"),
            (1, 1, [True, True], "base"),
            (2, 2, [True, False], "split"),
            (2, 3, [False, False], "control"),
            (2, 2, [False, True, False], "borrowed"),
            (2, 1, [True, True, True], "base"),
        ],
    )
    def test_coherent_control(self, num_x, num_y, control_values, work_layout):
        """Preserve all relative phases and restore all ancillas on entangled inputs."""
        control_wires = list(range(len(control_values)))
        x_wires = list(range(len(control_wires), len(control_wires) + num_x))
        y_wires = list(range(len(control_wires) + num_x, len(control_wires) + num_x + num_y))
        data_wires = control_wires + x_wires + y_wires
        work_wires = list(range(len(data_wires), len(data_wires) + num_y + len(control_wires) + 1))
        base_work, extra_work = work_wires, []
        work_wire_type = "zeroed"
        num_wires = len(data_wires) + len(work_wires)
        if work_layout == "split":
            base_work, extra_work = work_wires[:1], work_wires[1:]
        elif work_layout == "control":
            base_work, extra_work = [], work_wires
        elif work_layout == "borrowed":
            extra_work = [num_wires]
            num_wires += 1
            work_wire_type = "borrowed"

        def prepare():
            for index, wire in enumerate(data_wires):
                qp.RY(0.31 + 0.17 * index, wires=wire)
                qp.RZ(0.23 + 0.19 * index, wires=wire)
            for control, target in zip(data_wires, data_wires[1:]):
                qp.CNOT([control, target])
            if work_layout == "borrowed":
                qp.H(extra_work[0])
                qp.CNOT([extra_work[0], 0])

        @qp.qnode(qp.device("default.qubit", wires=num_wires))
        def reference():
            prepare()
            qp.ctrl(
                qp.SemiAdder(x_wires, y_wires, work_wires[: num_y - 1]),
                control=control_wires,
                control_values=control_values,
            )
            return qp.state()

        rule = qp.list_decomps("C(SemiAdder)")["flip_zero_ctrl_values(_controlled_semi_adder_ppm)"]
        base = qp.SemiAdder(x_wires, y_wires, base_work)

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            prepare()
            rule(
                base,
                control_wires=Wires(control_wires),
                control_values=control_values,
                work_wires=extra_work,
                work_wire_type=work_wire_type,
            )
            return qp.state()

        expected = reference()
        for _ in range(3):
            assert qp.math.allclose(circuit(), expected)

    @pytest.mark.catalyst
    @pytest.mark.parametrize("num_provided", [0, 2])
    @pytest.mark.parametrize("num_controls", [1, 2, 3])
    def test_compiled_allocation(self, num_provided, num_controls):
        """Execute the allocation and cleanup of missing clean work wires."""
        num_wires = num_controls + 3 + num_provided
        base = qp.SemiAdder(
            [num_controls],
            [num_controls + 1, num_controls + 2],
            list(range(num_controls + 3, num_wires)),
        )

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            for wire in range(num_controls + 1):
                qp.X(wire)
            qp.X(num_controls + 2)
            _controlled_semi_adder_ppm(base, Wires(range(num_controls)))
            return qp.probs(wires=range(num_wires))

        # All controls |1>, addend |1>, sum |10> and clean provided work wires.
        expected = qp.math.eye(1, 2**num_wires, (2 ** (num_controls + 3) - 2) * 2**num_provided)[0]
        assert qp.math.allclose(circuit(), expected)

    @pytest.mark.parametrize("control_values", [[True], [True, True], [False, True, False]])
    def test_graph_selects_ppm_rule(self, control_values, mocker):
        """A PPM target gate set selects the registered specialized controlled rule."""
        mocker.patch(
            "pennylane.templates.subroutines.arithmetic.semi_adder.compiler.active",
            return_value=True,
        )
        num_controls = len(control_values)
        op = qp.ctrl(
            qp.SemiAdder(
                range(num_controls, num_controls + 2),
                range(num_controls + 2, num_controls + 4),
                range(num_controls + 4, 2 * num_controls + 7),
            ),
            control=range(num_controls),
            control_values=control_values,
        )
        assert op.base.wires == Wires(range(num_controls, 2 * num_controls + 7))
        graph = qp.decomposition.DecompositionGraph(
            [op],
            gate_set={
                "PPR_8": 2,
                "PauliMeasure": 1,
                "PauliX": 0,
                "PauliZ": 0,
                "GlobalPhase": 0,
            },
        )
        solution = graph.solve()
        assert (
            solution.decomposition(op).name == "flip_zero_ctrl_values(_controlled_semi_adder_ppm)"
        )

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        ("control_values", "initial_controls", "expected_sum"),
        [
            ([True], [1], 1),
            ([False], [0], 1),
            ([False], [1], 0),
            ([True, True], [1, 1], 1),
            ([True, False], [1, 0], 1),
            ([True, False], [1, 1], 0),
            ([False, True, False], [0, 1, 0], 1),
            ([False, True, False], [0, 0, 0], 0),
        ],
    )
    def test_compiled_graph_decomposition(self, control_values, initial_controls, expected_sum):
        """Compile the public controlled operator through the registered PPM rule."""
        num_controls = len(control_values)
        num_wires = 2 * num_controls + 7
        gate_set = {
            "PPR_8": 2,
            "PauliMeasure": 1,
            "PauliX": 0,
            "PauliZ": 0,
            "GlobalPhase": 0,
        }

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            for wire, value in enumerate(initial_controls):
                if value:
                    qp.X(wire)
            qp.X(num_controls + 1)
            qp.ctrl(
                qp.SemiAdder(
                    range(num_controls, num_controls + 2),
                    range(num_controls + 2, num_controls + 4),
                    range(num_controls + 4, num_wires),
                ),
                control=range(num_controls),
                control_values=control_values,
            )
            return qp.probs(wires=range(num_wires))

        # Include every control, the addend, the sum and all clean work wires.
        bits = initial_controls + [0, 1, 0, expected_sum] + [0] * (num_controls + 3)
        expected = sum(bit * 2**index for index, bit in enumerate(reversed(bits)))
        assert qp.math.allclose(circuit(), qp.math.eye(1, 2**num_wires, expected)[0])


class TestSemiAdderPPM:
    """Test the measurement-based (Pauli product) decomposition of qp.SemiAdder."""

    def test_condition_without_compiler(self):
        """Test that the PPM decomposition is not applicable without an active compiler."""
        assert _semi_adder_ppm_condition(Wire[3], Wire[3], work_wires=Wire[3]) is False

    @pytest.mark.parametrize(
        ("num_y_wires", "num_provided", "expected"),
        [(4, 0, 5), (4, 3, 2), (4, 5, 0), (4, 7, 0), (1, 0, 2)],
    )
    def test_work_wires(self, num_y_wires, num_provided, expected):
        """The ladder needs the carries, the shared auxiliary qubit and one wire held at |0>."""
        spec = _semi_adder_ppm_work_wires(y_wires=Wire[num_y_wires], work_wires=Wire[num_provided])
        assert spec == {"zeroed": expected}

    @pytest.mark.capture
    @pytest.mark.parametrize("num_provided", [0, 2])
    def test_ppm_allocates_missing_work_wires(self, num_provided):
        """Test that the work wires that were not provided are allocated and given back."""
        import jax

        jaxpr = jax.make_jaxpr(qp.capture.subroutine(_semi_adder_ppm))(
            Wires([0, 1]), Wires([2, 3]), list(range(4, 4 + num_provided))
        )
        eqns = jaxpr.eqns[0].params["jaxpr"].eqns
        allocations = [eqn for eqn in eqns if str(eqn.primitive) == "allocate"]
        assert len(allocations) == 1
        # the ladder needs len(y_wires) + 1 = 3 work wires in total
        assert allocations[0].params["num_wires"] == 3 - num_provided
        assert allocations[0].params["restored"] is True
        assert len([eqn for eqn in eqns if str(eqn.primitive) == "deallocate"]) == 1

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        ("num_x_wires", "num_y_wires"),
        [(1, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3), (2, 4), (4, 4)],
    )
    def test_ppm_correctness(self, num_x_wires, num_y_wires, seed):
        """Test that the PPM decomposition equals SemiAdder amplitude by amplitude, global phase included."""

        x_wires = list(range(num_x_wires))
        y_wires = list(range(num_x_wires, num_x_wires + num_y_wires))
        work_wires = list(range(num_x_wires + num_y_wires, num_x_wires + 2 * num_y_wires + 1))
        num_wires = num_x_wires + 2 * num_y_wires + 1

        def prepare():
            rng = np.random.default_rng(seed)
            for wire in x_wires + y_wires:
                qp.RY(float(rng.uniform(0.4, 2.4)), wires=wire)
                qp.RZ(float(rng.uniform(0.4, 2.4)), wires=wire)
            for control, target in zip(x_wires + y_wires, (x_wires + y_wires)[1:]):
                qp.CNOT(wires=[control, target])

        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def reference():
            prepare()
            _semi_adder(x_wires, y_wires, work_wires[: num_y_wires - 1])
            return qp.state()

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def ppm():
            prepare()
            _semi_adder_ppm(Wires(x_wires), Wires(y_wires), work_wires)
            return qp.state()

        expected = np.asarray(reference())
        for _ in range(2):
            assert qp.math.allclose(ppm(), expected)

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        ("num_x_wires", "num_y_wires", "x", "y"),
        [
            (1, 1, 1, 1),
            (2, 2, 1, 0),
            (2, 2, 3, 3),
            (1, 3, 1, 6),
            (3, 2, 5, 2),
            (4, 4, 11, 9),
        ],
    )
    def test_ppm_arithmetic_and_clean_work_wires(self, num_x_wires, num_y_wires, x, y):
        """Test that the PPM decomposition adds correctly and returns every work wire to |0>."""
        x_wires = list(range(num_x_wires))
        y_wires = list(range(num_x_wires, num_x_wires + num_y_wires))
        work_wires = list(range(num_x_wires + num_y_wires, num_x_wires + 2 * num_y_wires + 1))
        num_wires = num_x_wires + 2 * num_y_wires + 1

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            for i, wire in enumerate(reversed(x_wires)):
                if (x >> i) & 1:
                    qp.X(wire)
            for i, wire in enumerate(reversed(y_wires)):
                if (y >> i) & 1:
                    qp.X(wire)
            _semi_adder_ppm(Wires(x_wires), Wires(y_wires), work_wires)
            return qp.probs(wires=y_wires), qp.probs(wires=work_wires)

        y_probs, work_probs = circuit()
        # x is truncated to the size of y_wires, which does not change the sum modulo 2**len(y)
        assert np.isclose(y_probs[(x + y) % 2**num_y_wires], 1.0)
        assert np.isclose(work_probs[0], 1.0)
