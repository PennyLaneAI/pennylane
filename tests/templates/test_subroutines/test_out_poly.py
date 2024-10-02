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
Tests for the OutPoly template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def f_test(x, y, z):
    return x ** 2 + y * x * z ** 5 - z ** 3 + 3

def test_standard_validity_OutPoly():
    """Check the operation using the assert_valid function."""
    wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "aux": 2})

    op = qml.OutPoly(f_test,
                    wires["x"],
                    wires["y"],
                    wires["z"],
                    wires["output"],
                    mod = 5,
                    work_wires = wires["aux"]
                    )

    print(op.wires)

    op2 = op.map_wires({i:20 + i for i in range(14)})
    print(op2.wires)

    qml.ops.functions.assert_valid(op)


class TestOutPoly:
    """Test the qml.OutPoly template."""

    @pytest.mark.parametrize(
        ("f", "x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (lambda x, y: 3 * x ** 3 - 3 * y, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [10, 11]),
            (lambda x, y: 2 * x * y - 3 * x, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 7, [10, 11]),
            (lambda x, y: -2 * x - 3 * y + 1, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], None, None),
            (lambda x, y: 2 * x * y - 3 * x + 5, [0, 1, 2], [3, 4, 5], [6, 7, 8, 9], None, None),

        ],
    )
    def test_operation_result(
        self, f, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutPoly template output."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():

            qml.BasisEmbedding(2, wires=x_wires)
            qml.BasisEmbedding(1, wires=y_wires)
            qml.OutPoly(f, x_wires, y_wires, output_wires, mod = mod, work_wires = work_wires)
            return qml.probs(wires=output_wires)

        if mod == None:
            mod = int(2**len(output_wires))
        assert np.isclose(np.argmax(circuit()), f(2, 1) % mod)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6.1, [10, 11], "mod must be integer."),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [0, 11], "None of the wires in"),
            ([0, 1, 2], [3, 4, 5], [6, 7, 8, 9], 6, [10], "If mod is not"),
            ([0, 1, 2], None,  [6, 7, 8, 9], 6, [10,11], "The function takes"),
        ],
    )
    def test_operation_and_test_wires_error(
        self, x_wires, y_wires, output_wires, mod, work_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        with pytest.raises(ValueError, match=msg_match):
            if y_wires:
                qml.OutPoly(lambda x, y: 3 * x ** 3 - 3 * y, x_wires, y_wires, output_wires, mod=mod, work_wires=work_wires)
            else:
                qml.OutPoly(lambda x, y: 3 * x ** 3 - 3 * y, x_wires, output_wires, mod=mod, work_wires=work_wires)


    def test_non_polynomial_function_error(self):
        """Test that proper errors are raised for non polynomial functions"""

        def f(x, y):
            return x + y ** 0.5

        x_wires = [0, 1, 2]
        y_wires = [3, 4, 5]
        output_wires = [6, 7, 8]

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.OutPoly(f, x_wires, y_wires, output_wires)
            return qml.state()

        with pytest.raises(ValueError, match="The function must be polynomial"):
            circuit()

    def test_decomposition(self):
        """Test that compute_decomposition and decomposition work as expected."""

        def f(x, y):
            return x + y

        x_wires = [0, 1, 2]
        y_wires = [3, 4, 5]
        output_wires = [6,7,8]
        poly_decomposition = qml.OutPoly(
            f, x_wires, y_wires, output_wires
        ).decomposition()

        for op in poly_decomposition:
            if isinstance(op, qml.QFT):
                assert op.wires == qml.wires.Wires(output_wires)
            elif isinstance(op, qml.ops.op_math.controlled.Controlled):
                for wire in op.control_wires:
                    assert wire in qml.wires.Wires(x_wires) + qml.wires.Wires(y_wires)
                assert isinstance(op.base, qml.PhaseAdder)
            else:
                return False

    '''
    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)
        x = 2
        k = 6
        mod = 7
        x_wires = [0, 1, 2]
        work_wires = [3, 4]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

        # pylint: disable=bad-reversed-sequence
        result = sum(bit * (2**i) for i, bit in enumerate(reversed(circuit())))
        assert jax.numpy.allclose(result, (x + k) % mod)
    '''


