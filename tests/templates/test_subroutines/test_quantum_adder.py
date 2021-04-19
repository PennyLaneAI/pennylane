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

    @pytest.mark.parametrize("a,b",
    [
        ('0', '0'),
        ('0', '1'),
        ('1', '0'),
        ('1', '1'),
        ('00', '00'),
        ('01', '01'),
        ('10', '01'),
        ('10', '10'),
        ('11', '11'),
        ('1', '110'),
        ('1', '111')
    ],
    )
    def test_correct_sum(self, a, b):
        """Tests whether qml.QuantumAdder() produces the correct sums"""
        a_input = np.array([int(item) for item in list(a)])
        b_input = np.array([int(item) for item in list(b)])
        a_wires=list(range(len(a_input)))
        b_wires=list(range(len(a_input),len(b_input)+len(a_input)))
        carry_wires=list(range(len(a_input)+len(b_input),max(len(a_input),len(b_input)+1+len(a_input)+len(b_input))))
        dev = qml.device('default.qubit',wires=(len(a_wires)+len(b_wires)+len(carry_wires)))
        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.append(a_input,b_input),wires=(a_wires+b_wires))
            qml.templates.QuantumAdder(a_wires=a_wires,b_wires=b_wires,carry_wires=carry_wires)
            return qml.state()

        state=circuit()
        state_value=np.argmax(state)
        binary_format_string = "0"+str(dev.num_wires)+"b"
        state_binary=format(state_value,binary_format_string)
        result = [carry_wires[0]]+b_wires
        result_string=""
        for i in result:
            result_string=result_string+state_binary[i]

        assert int(result_string,2) == int(a,2) + int(b,2)

    def test_wrong_order(self):
        """Test if a ValueError is raised when a_wires has more wires than b_wires"""
        with pytest.raises(ValueError, match="The longer bit string must be in b_wires"):
            qml.templates.QuantumAdder([0, 1], [2], [3, 4, 5])

    def test_few_carry_wires(self):
        """Test if a ValueError is raised when there are not enough carry wires"""
        with pytest.raises(ValueError, match="The carry wires must have one more wire than the a and b wires"):
            qml.templates.QuantumAdder([0, 1], [2, 3], [4, 5])
