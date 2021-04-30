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
from scipy.stats import unitary_group


class TestQuantumAdder:
    """Tests for the QuantumAdder template."""

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
        ],
    )
    def test_correct_sum(self, a, b):
        """Tests whether QuantumAdder produces the correct sums"""
        a_input = np.array([int(item) for item in list(a)])
        b_input = np.array([int(item) for item in list(b)])
        a_wires = list(range(len(a_input))) np.kron(
        b_wires = list(range(len(a_input), len(b_input) + len(a_input)))
        carry_wires = list(
            range(
                len(a_input) + len(b_input),
                max(len(a_input), len(b_input) + 1 + len(a_input) + len(b_input)),
            )
        )
        dev = qml.device("default.qubit", wires=(len(a_wires) + len(b_wires) + len(carry_wires)))

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.append(a_input, b_input), wires=(a_wires + b_wires))
            qml.templates.QuantumAdder(a_wires=a_wires, b_wires=b_wires, carry_wires=carry_wires)
            return qml.state()

        state = circuit()
        state_value = np.argmax(state)
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = format(state_value, binary_format_string)
        result = [carry_wires[0]] + b_wires
        result_string = ""
        for i in result:
            result_string = result_string + state_binary[i]

        assert int(result_string, 2) == int(a, 2) + int(b, 2)
    @pytest.mark.parametrize(
        "a_state,b_state",
        [
            ([1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]),
            ([1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4)], [1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4)]),
            ([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], [1/np.sqrt(3), np.sqrt(2/3), 0, 0]),
            ([1j/np.sqrt(2), 1/np.sqrt(2), 0, 0], [1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(2), 0]),
            ([-1j/np.sqrt(2), 1j/np.sqrt(2), 0, 0], [-1j/np.sqrt(3), np.sqrt(2/3), 0, 0]),
        ],
    )
    def test_addr_superpositions(self, a_state, b_state):
        """Tests whether ControlledQuantumAdder produces the correct outputs for superposition inputs"""
        a_input = np.array(a_state)
        b_input = np.array(b_state)
        n_a_wires = int(np.log2(len(a_state)))
        n_b_wires = int(np.log2(len(b_state)))
        a_wires = list(range(n_a_wires))
        b_wires = list(range(n_a_wires, n_a_wires+n_b_wires))
        carry_wires = list(
            range(
                n_a_wires+n_b_wires,
                n_b_wires + 1 + n_a_wires + n_b_wires,
            )
        )
        result_wires = [carry_wires[0]] + b_wires
        dev = qml.device("default.qubit", wires=(len(a_wires) + len(b_wires) + len(carry_wires)))
        @qml.qnode(dev)
        def circuit():
            qml.templates.MottonenStatePreparation(np.kron(a_input,b_input), wires=a_wires + b_wires)
            qml.templates.QuantumAdder(a_wires=a_wires, b_wires=b_wires, carry_wires=carry_wires)
            return qml.state()

        state = circuit()
        #state post processing
        state_value = np.argwhere(state!=0) #the decimal location of the nonzero states
        probs = [np.abs(state[int(i)])**2 for i in state_value] #the probability of these nonzero states
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = [format(int(state_value), binary_format_string) for state_value in state_value] #the binary value of these nonzero states
        result_strings = [""]*len(probs) #the binary value of the result
        for j in range(len(probs)):
            for bit in result_wires:
                result_strings[j] = result_strings[j] + state_binary[j][bit]
        result_value = [int(i,2) for i in result_strings]


        expval = np.inner(np.array(result_value),np.array(probs))
        classexpval = np.inner(np.array([np.abs(i)**2 for i in a_state]),np.array(range(len(a_state))))+ np.inner(np.array([np.abs(i)**2 for i in b_state]),np.array(range(len(b_state))))
        assert np.isclose(expval, classexpval)

    def test_wrong_order(self):
        """Test if a ValueError is raised when a_wires has more wires than b_wires"""
        with pytest.raises(ValueError, match="The longer bit string must be in b_wires"):
            qml.templates.QuantumAdder([0, 1], [2], [3, 4, 5])

    def test_few_carry_wires(self):
        """Test if a ValueError is raised when there are not enough carry wires"""
        with pytest.raises(
            ValueError, match="The carry wires must have 1 more wire than the b wires"
        ):
            qml.templates.QuantumAdder([0, 1], [2, 3], [4, 5])

class TestControlledQuantumAdder:
    """Tests for the QuantumAdder template."""

    @pytest.mark.parametrize(
        "a,b,c",
        [
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("0", "1", "0"),
            ("0", "1", "1"),
            ("1", "0", "0"),
            ("1", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
            ("00", "00", "0"),
            ("00", "00", "1"),
            ("01", "01", "0"),
            ("01", "01", "1"),
            ("10", "01", "0"),
            ("10", "01", "1"),
            ("10", "10", "0"),
            ("10", "10", "1"),
            ("11", "11", "0"),
            ("11", "11", "1"),
            ("1", "110", "0"),
            ("1", "110", "1"),
            ("1", "111", "0"),
            ("1", "111", "1"),
        ],
    )
    def test_correct_sum(self, a, b, c):
        """Tests whether ControlledQuantumAdder produces the correct outputs"""
        a_input = np.array([int(item) for item in list(a)])
        b_input = np.array([int(item) for item in list(b)])
        a_wires = list(range(len(a_input)))
        b_wires = list(range(len(a_input), len(b_input) + len(a_input)))
        carry_wires = list(
            range(
                len(a_input) + len(b_input),
                len(b_input) + 1 + len(a_input) + len(b_input),
            )
        )
        control_wire = carry_wires[-1]+1
        work_wire = control_wire+1
        dev = qml.device("default.qubit", wires=(len(a_wires) + len(b_wires) + len(carry_wires) + 2))

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.concatenate((a_input, b_input,np.array([int(c)]))), wires=(a_wires + b_wires + [control_wire]))
            qml.templates.ControlledQuantumAdder(a_wires=a_wires, b_wires=b_wires, carry_wires=carry_wires, control_wire=control_wire, work_wire=work_wire)
            return qml.state()

        state = circuit()
        state_value = np.argmax(state)
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = format(state_value, binary_format_string)
        result = [carry_wires[0]] + b_wires
        result_string = ""
        for i in result:
            result_string = result_string + state_binary[i]

        assert int(result_string, 2) == int(c, 2) * int(a, 2) + int(b, 2)

    @pytest.mark.parametrize(
        "a_state,b_state,c_state",
        [
            ([1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)], [0, 1]),
            ([1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4)], [1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4), 1/np.sqrt(4)], [0, 1]),
            ([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], [1/np.sqrt(3), np.sqrt(2/3), 0, 0], [0, 1]),
            ([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], [1/np.sqrt(3), np.sqrt(2/3), 0, 0], [np.sqrt(1/3), np.sqrt(2/3)]),
            ([-1j/np.sqrt(2), 1j/np.sqrt(2), 0, 0], [-1j/np.sqrt(3), np.sqrt(2/3), 0, 0], [1j/np.sqrt(3), np.sqrt(2/3)]),
        ],
    )
    def test_ctrladdr_superpositions(self, a_state, b_state, c_state):
        """Tests whether ControlledQuantumAdder produces the correct outputs for superposition inputs"""
        a_input = np.array(a_state)
        b_input = np.array(b_state)
        c_input = np.array(c_state)
        n_a_wires = int(np.log2(len(a_state)))
        n_b_wires = int(np.log2(len(b_state)))
        a_wires = list(range(n_a_wires))
        b_wires = list(range(n_a_wires, n_a_wires+n_b_wires))
        carry_wires = list(
            range(
                n_a_wires+n_b_wires,
                n_b_wires + 1 + n_a_wires + n_b_wires,
            )
        )
        control_wire = carry_wires[-1]+1
        work_wire = control_wire+1
        result_wires = [carry_wires[0]] + b_wires
        dev = qml.device("default.qubit", wires=(len(a_wires) + len(b_wires) + len(carry_wires) + 2))
        @qml.qnode(dev)
        def circuit():
            qml.templates.MottonenStatePreparation(np.kron(a_input, np.kron(b_input, c_input)), wires=(a_wires + b_wires + [control_wire]))
            qml.templates.ControlledQuantumAdder(a_wires=a_wires, b_wires=b_wires, carry_wires=carry_wires, control_wire=control_wire, work_wire=work_wire)
            return qml.state()

        state = circuit()
        #state post processing
        state_value = np.argwhere(state!=0) #the decimal location of the nonzero states
        probs = [np.abs(state[int(i)])**2 for i in state_value] #the probability of these nonzero states
        binary_format_string = "0" + str(dev.num_wires) + "b"
        state_binary = [format(int(state_value), binary_format_string) for state_value in state_value] #the binary value of these nonzero states
        result_strings = [""]*len(probs) #the binary value of the result
        for j in range(len(probs)):
            for bit in result_wires:
                result_strings[j] = result_strings[j] + state_binary[j][bit]
        result_value = [int(i,2) for i in result_strings]


        expval = np.inner(np.array(result_value),np.array(probs))
        classexpval = np.abs(c_state[1])**2*np.inner(np.array([np.abs(i)**2 for i in a_state]),np.array(range(len(a_state))))+ np.inner(np.array([np.abs(i)**2 for i in b_state]),np.array(range(len(b_state))))
        assert np.isclose(expval, classexpval)

    def test_wrong_order(self):
        """Test if a ValueError is raised when a_wires has more wires than b_wires"""
        with pytest.raises(ValueError, match="The longer bit string must be in b_wires"):
            qml.templates.QuantumAdder([0, 1], [2], [3, 4, 5])

    def test_few_carry_wires(self):
        """Test if a ValueError is raised when there are not enough carry wires"""
        with pytest.raises(
            ValueError, match="The carry wires must have 1 more wire than the b wires"
        ):
            qml.templates.QuantumAdder([0, 1], [2, 3], [4, 5])
