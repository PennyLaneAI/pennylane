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
Tests for the PhaseAdder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.arithmetic.phase_adder import _add_k_fourier


@pytest.mark.jax
def test_standard_validity_Phase_Adder():
    """Check the operation using the assert_valid function."""
    k = 6
    mod = 11
    x_wires = [0, 1, 2, 3]
    work_wire = [4]
    op = qml.PhaseAdder(k, x_wires=x_wires, mod=mod, work_wire=work_wire)
    qml.ops.functions.assert_valid(op)


def test_falsy_zero_as_work_wire():
    """Test that work wire is not treated as a falsy zero."""
    k = 6
    mod = 11
    x_wires = [1, 2, 3, 4]
    work_wire = 0
    op = qml.PhaseAdder(k, x_wires=x_wires, mod=mod, work_wire=work_wire)
    qml.ops.functions.assert_valid(op)


def test_add_k_fourier():
    """Test the private _add_k_fourier function."""

    ops = _add_k_fourier(2, wires=range(2))
    assert len(ops) == 2
    assert ops[0].name == "PhaseShift"
    assert ops[1].name == "PhaseShift"
    assert np.isclose(ops[0].parameters[0], 2 * np.pi)
    assert np.isclose(ops[1].parameters[0], np.pi)


class TestPhaseAdder:
    """Test the PhaseAdder template."""

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wire", "x"),
        [
            (
                6,
                [0, 1, 2],
                7,
                [4],
                2,
            ),
            (
                6,
                ["a", "b", "c"],
                7,
                ["d"],
                3,
            ),
            (
                0,
                [0, 1, 2, 3, 5],
                9,
                [4],
                2,
            ),
            (
                2,
                [0, 1, 4],
                4,
                [3],
                1,
            ),
            (
                0,
                [0, 1, 4],
                4,
                [3],
                0,
            ),
            (
                -2,
                [0, 1, 4],
                4,
                [3],
                0,
            ),
            (
                10,
                [0, 1, 2, 5],
                9,
                [3],
                3,
            ),
            (
                1,
                [0, 1, 2, 4],
                7,
                [3],
                3,
            ),
            (
                6,
                [0, 1, 2, 3],
                None,
                [4],
                2,
            ),
        ],
    )
    def test_operation_result(
        self, k, x_wires, mod, work_wire, x
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the PhaseAdder template output."""
        dev = qml.device("default.qubit")

        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

        if mod is None:
            mod = 2 ** len(x_wires)

        # pylint: disable=bad-reversed-sequence
        assert np.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x)[0, :]))), (x + k) % mod
        )

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wire", "msg_match"),
        [
            (
                1,
                [0, 1, 2],
                9,
                [3],
                ("PhaseAdder must have enough x_wires to represent mod."),
            ),
            (
                1,
                [0, 1, 2],
                9,
                None,
                (r"If mod is not"),
            ),
            (
                3,
                [0, 1, 2, 3, 4],
                12,
                [4],
                "None of the wires in work_wire should be included in x_wires.",
            ),
        ],
    )
    def test_operation_and_wires_error(
        self, k, x_wires, mod, work_wire, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test errors are raised"""
        with pytest.raises(ValueError, match=msg_match):
            qml.PhaseAdder(k, x_wires, mod, work_wire)

    @pytest.mark.parametrize("work_wire", [None, [], [3, 4]])
    def test_validation_of_num_work_wires(self, work_wire):
        """Test that when mod is not 2**len(x_wires), validation confirms two
        work wires are present, while any work wires are accepted for mod=2**len(x_wires)"""

        # if mod=2**len(x_wires), anything goes
        qml.PhaseAdder(3, [0, 1, 2], mod=8, work_wire=work_wire)

        with pytest.raises(ValueError, match="one work wire should be provided"):
            qml.PhaseAdder(3, [0, 1, 2], mod=7, work_wire=work_wire)

    def test_valid_inputs_for_work_wires(self):
        """Test that both an integer and a list with a length of 1 are valid
        inputs for work_wires, and have the same result"""

        op1 = qml.PhaseAdder(3, [0, 1, 2], mod=8, work_wire=[3])
        op2 = qml.PhaseAdder(3, [0, 1, 2], mod=8, work_wire=3)

        assert op1.hyperparameters["work_wire"] == op2.hyperparameters["work_wire"]

    @pytest.mark.parametrize(
        ("k", "x_wires", "mod", "work_wire", "msg_match"),
        [
            (
                2.3,
                [0, 1, 2],
                9,
                [3],
                ("Both k and mod must be integers"),
            ),
            (
                2,
                [0, 1, 2],
                3.2,
                [3],
                ("Both k and mod must be integers"),
            ),
        ],
    )
    def test_types_error(
        self, k, x_wires, mod, work_wire, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test errors are raised"""
        with pytest.raises(ValueError, match=msg_match):
            qml.PhaseAdder(k, x_wires, mod, work_wire)

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        k = 4
        x_wires = [1, 2, 3]
        mod = 7
        work_wire = [0]

        phase_adder_decomposition = qml.PhaseAdder(
            k, x_wires, mod, work_wire
        ).compute_decomposition(k, x_wires, mod, work_wire)
        op_list = []

        if mod == 2 ** (len(x_wires)):
            op_list.extend(_add_k_fourier(k, x_wires))
        else:
            aux_k = x_wires[0]
            op_list.extend(_add_k_fourier(k, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=1))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(_add_k_fourier(k, x_wires))

        for op1, op2 in zip(phase_adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("mod", [7, 8])
    def test_decomposition_new(self, mod):
        """Tests the decomposition rule implemented with the new system."""

        k = 4
        x_wires = [1, 2, 3]
        work_wire = [0]
        op = qml.PhaseAdder(k, x_wires, mod, work_wire)
        for rule in qml.list_decomps(qml.PhaseAdder):
            _test_decomposition_rule(op, rule)

    def test_work_wires_added_correctly(self):
        """Test that no work wires are added if work_wire = None"""
        wires = qml.PhaseAdder(1, x_wires=[1, 2]).wires
        assert wires == qml.wires.Wires([1, 2])

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        x = 2
        k = 6
        mod = 7
        x_wires = [0, 1, 2]
        work_wire = [4]
        dev = qml.device("default.qubit")

        @jax.jit
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.QFT(wires=x_wires)
            qml.PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

        # pylint: disable=bad-reversed-sequence
        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()[0, :]))), (x + k) % mod
        )
