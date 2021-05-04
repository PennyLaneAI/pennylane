# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import pennylane as qml
import numpy as np


class TestQuantumMultiplier:
    """Tests for the QuantumMultiplier template."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("00", "00"),
            ("01", "01"),
            ("10", "01"),
            ("10", "10"),
            ("11", "11"),
            ("1", "110"),
            ("1", "111"),
            ("111", "111"),
        ],
    )
    def test_correct_product(self, a, b):
        """Tests whether QuantumMultiplier produces the correct sums"""
        a_input = np.array([int(item) for item in list(a)])
        b_input = np.array([int(item) for item in list(b)])
        multiplicand_wires = list(range(len(a_input)))
        multiplier_wires = list(range(len(a_input), len(b_input) + len(a_input)))
        accumulator_wires = list(
            range(multiplier_wires[-1] + 1, len(multiplicand_wires) + multiplier_wires[-1] + 1)
        )
        carry_wires = list(
            range(
                accumulator_wires[-1] + 1,
                len(multiplier_wires) + len(multiplicand_wires) + accumulator_wires[-1] + 1,
            )
        )
        work_wire = carry_wires[-1] + 1
        ancilla_wire = work_wire + 1
        n_wires = ancilla_wire + 1
        result_wires = carry_wires[::-1] + accumulator_wires
        result_wires = result_wires[-(len(multiplier_wires) + len(multiplicand_wires)) :]

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(
                np.append(a_input, b_input), wires=(multiplicand_wires + multiplier_wires)
            )
            qml.templates.QuantumMultiplier(
                multiplicand_wires=multiplicand_wires,
                multiplier_wires=multiplier_wires,
                carry_wires=carry_wires,
                accumulator_wires=accumulator_wires,
                work_wire=work_wire,
            )
            return qml.state()

        state = circuit()
        state_value = np.argmax(state)
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = format(state_value, binary_format_string)
        result_string = ""
        for i in result_wires:
            result_string = result_string + state_binary[i]

        assert int(result_string, 2) == int(a, 2) * int(b, 2)

    @pytest.mark.parametrize(
        "a_state,b_state",
        [
            ([1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
            (
                [1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4)],
                [1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4)],
            ),
            ([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], [1 / np.sqrt(3), np.sqrt(2 / 3), 0, 0]),
            (
                [1j / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                [1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(2), 0],
            ),
            ([-1j / np.sqrt(2), 1j / np.sqrt(2), 0, 0], [-1j / np.sqrt(3), np.sqrt(2 / 3), 0, 0]),
        ],
    )
    def test_superposition(self, a_state, b_state):
        """Tests whether QuantumMultiplier produces correct products of superposition states"""
        a_input = np.array(a_state)
        b_input = np.array(b_state)
        n_a_wires = int(np.log2(len(a_state)))
        n_b_wires = int(np.log2(len(b_state)))
        multiplicand_wires = list(range(n_a_wires))
        multiplier_wires = list(range(n_a_wires, n_a_wires + n_b_wires))
        accumulator_wires = list(
            range(multiplier_wires[-1] + 1, len(multiplicand_wires) + multiplier_wires[-1] + 1)
        )
        carry_wires = list(
            range(
                accumulator_wires[-1] + 1,
                len(multiplier_wires) + len(multiplicand_wires) + accumulator_wires[-1] + 1,
            )
        )
        work_wire = carry_wires[-1] + 1
        ancilla_wire = work_wire + 1
        n_wires = ancilla_wire + 1
        result_wires = carry_wires[::-1] + accumulator_wires
        result_wires = result_wires[-(len(multiplier_wires) + len(multiplicand_wires)) :]
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(
                np.kron(a_input, b_input), wires=multiplicand_wires + multiplier_wires
            )
            qml.templates.QuantumMultiplier(
                multiplicand_wires=multiplicand_wires,
                multiplier_wires=multiplier_wires,
                carry_wires=carry_wires,
                accumulator_wires=accumulator_wires,
                work_wire=work_wire,
            )
            return qml.state()

        state = circuit()
        # state post processing
        state_value = np.argwhere(state != 0)  # the decimal location of the nonzero states
        probs = [
            np.abs(state[int(i)]) ** 2 for i in state_value
        ]  # the probability of these nonzero states
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = [
            format(int(state_value), binary_format_string) for state_value in state_value
        ]  # the binary value of these nonzero states
        result_strings = [""] * len(probs)  # the binary value of the result
        for j in range(len(probs)):
            for bit in result_wires:
                result_strings[j] = result_strings[j] + state_binary[j][bit]
        result_value = [int(i, 2) for i in result_strings]

        expval = np.inner(np.array(result_value), np.array(probs))
        classexpval = np.inner(
            np.array([np.abs(i) ** 2 for i in a_state]), np.array(range(len(a_state)))
        ) * np.inner(np.array([np.abs(i) ** 2 for i in b_state]), np.array(range(len(b_state))))
        assert np.isclose(expval, classexpval)
