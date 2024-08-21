"""
Contains the PhaseAdder template.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import Operation


def _add_k_fourier(k, wires):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(qml.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


class PhaseAdder(Operation):
    r"""Performs the Inplace Phase Addition operation.

    This operator adds the integer :math:`k` modulo :math:`mod` in the Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (m) \rangle = |\phi (m+k \, \text{mod} \, mod) \rangle,

    where :math:`|\phi (m) \rangle` represents the :math:`| m \rangle`: state in the Fourier basis such:

    .. math::

        QFT |m \rangle = |\phi (m) \rangle.

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

    Args:
        k (int): number that wants to be added.
        wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent k and mod.
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
        work_wires (Sequence[int]): the auxiliary wire to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(wires)}}`.

    **Example**

    Sum of two integers :math:`m=5` and :math:`k=4` modulo :math:`mod=7`. Note that to perform this sum using qml.PhaseAdder we need that :math:`m,k < mod`.

    .. code-block::

        m = 5
        k = 4
        mod = 7
        wires_m =[1,2,3]
        work_wires=[0,4]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def adder_modulo(m, k, mod, wires_m, work_wires):
            # Function that performs m + k modulo mod in the computational basis
            qml.BasisEmbedding(m, wires=wires_m)
            qml.QFT(wires=work_wires[:1]+ wires_m)
            PhaseAdder(k, wires_m, mod, work_wires)
            qml.adjoint(qml.QFT)(wires=work_wires[:1]+wires_m)
            return qml.sample(wires=wires_m)

    .. code-block:: pycon

        >>> print(f"The ket representation of {m} + {k} mod {mod} is {adder_modulo(m, k, mod,wires_m,work_wires)}")
        The ket representation of 5 + 4 mod 7 is [0 1 0]

    We can see that the result [0 1 0] corresponds to 2, which comes from :math:`5+4=9 \longrightarrow 9 \, \text{mod} \,  7 = 2`.
    """

    grad_method = None

    def __init__(
        self, k, wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments
        if mod is None:
            mod = 2 ** (len(wires))
        if k >= mod:
            raise ValueError("The module mod must be larger than k.")
        if not hasattr(wires, "__len__") or mod > 2 ** (len(wires)):
            raise ValueError("PhaseAdder must have at least enough wires to represent mod.")
        if work_wires is not None:
            if any(wire in work_wires for wire in wires):
                raise ValueError("None of the wires in work_wires should be included in wires.")
        else:
            work_wires = [wires[-1] + 1, wires[-1] + 2]

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(k, mod, work_wires, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
            wires (Sequence[int]): the wires the operation acts on
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.PhaseAdder.compute_decomposition(k=2,mod=8,wires=[1,2,3],work_wires=[0,4])
        [PhaseShift(6.283185307179586, wires=[1]),
        PhaseShift(3.141592653589793, wires=[2]),
        PhaseShift(1.5707963267948966, wires=[3])]
        """
        op_list = []

        if mod == 2 ** len(wires):
            op_list.extend(_add_k_fourier(k, wires))
        else:
            new_wires = work_wires[:1] + wires
            work_wire = work_wires[1]
            aux_k = new_wires[0]
            op_list.extend(_add_k_fourier(k, new_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, new_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.CNOT(wires=[aux_k, work_wire]))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, new_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, new_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=new_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=new_wires))
            op_list.extend(_add_k_fourier(k, new_wires))

        return op_list


"""
Contains the OutAdder template.
"""

import pennylane as qml
from pennylane.operation import Operation


class OutAdder(Operation):
    r"""Performs the Outplace Addition operation.

    This operator adds the integer :math:`k` modulo :math:`mod` in the computational basis:

    .. math::

        \text{OutAdder}(mod) |m \rangle | k \rangle | 0 \rangle = |m \rangle | k \rangle | m+k \, \text{mod} \, mod \rangle ,

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

    Args:
        x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
        y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
        output_wires (Sequence[int]): the wires that stores the sum modulo mod :math:`x+y \, \text{mod} \, mod`.
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
        work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(wires)}}`

    **Example**

    Sum of two integers :math:`x=8` and :math:`y=5` modulo :math:`mod=15`. Note that to perform this sum using qml.OutAdder we need that :math:`x,y < mod`.

    .. code-block::

        x=5
        y=6
        mod=7
        x_wires=[0,1,2]
        y_wires=[3,4,5]
        output_wires=[7,8,9]
        work_wires=[6,10]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit_OutAdder():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    .. code-block:: pycon

        >>> print(f"The ket representation of {x} + {y} mod {mod} is {circuit_OutAdder()}")
        The ket representation of 5 + 6 mod 7 is [1 0 0]

    We can see that the result [1 0 0] corresponds to 4, which comes from :math:`5+6=11 \longrightarrow 11 \, \text{mod} \, 7 = 4`.
    """

    grad_method = None

    def __init__(self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None):

        if mod is None:
            mod = 2 ** (len(output_wires))
        if (not hasattr(output_wires, "__len__")) or (mod > 2 ** len(output_wires)):
            raise ValueError("OutAdder must have at least enough wires to represent mod.")
        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        else:
            max_wire = max(max(x_wires), max(y_wires), max(output_wires))
            work_wires = [max_wire + 1, max_wire + 2]
        for key in ["x_wires", "y_wires", "output_wires", "work_wires"]:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])
        all_wires = sum(
            self.hyperparameters[key]
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        )
        self.hyperparameters["mod"] = mod
        super().__init__(wires=all_wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires, **kwargs):
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
            y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
            output_wires (Sequence[int]): the wires that stores the sum modulo mod :math:`x+y mod mod`.
            mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(output_wires)``.
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(output_wires)}}`.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutAdder.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [CNOT(wires=[2, 5]),
        CNOT(wires=[3, 6]),
        QFT(wires=[5, 6]),
        ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]),
        Adjoint(QFT(wires=[5, 6]))]
        """
        op_list = []
        if mod != 2 ** (len(output_wires)):
            qft_new_output_wires = work_wires[:1] + output_wires
        else:
            qft_new_output_wires = output_wires
        for i in range(len(y_wires)):
            op_list.append(qml.CNOT(wires=[y_wires[i], output_wires[i]]))
        op_list.append(qml.QFT(wires=qft_new_output_wires))
        op_list.append(
            qml.ControlledSequence(PhaseAdder(1, output_wires, mod, work_wires), control=x_wires)
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_new_output_wires))

        return op_list


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
Tests for the OutAdder template.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np


def test_standard_validity_OutAdder():
    """Check the operation using the assert_valid function."""
    x_wires = [0, 1]
    y_wires = [2, 3]
    output_wires = [4, 5, 8]
    mod = 7
    work_wires = [6, 7]
    op = OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
    qml.ops.functions.assert_valid(op)


class TestMultiplier:
    """Test the qml.OutAdder template."""

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            ([0, 1, 2], [3, 4, 5], [4, 5, 6], 7, [7, 8]),
            ([0, 1], [3, 4, 5], [4, 5, 6, 7], 11, [8, 9]),
            ([0, 1, 2], [3, 4], [5, 6], 3, [7, 8]),
        ],
    )
    def test_operation_result(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod is None:
            max = 2 ** len(output_wires)
        else:
            max = mod
        for x, y in zip(range(len(x_wires)), range(len(y_wires))):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))), (x + y) % max
            )

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                None,
                [9, 10],
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                6,
                None,
            ),
            (
                [0, 1],
                [3, 4, 5],
                [6, 7, 8],
                None,
                None,
            ),
        ],
    )
    def test_operation_result_args_None(
        self, x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=too-many-arguments
        """Test the correctness of the OutAdder template output."""
        dev = qml.device("default.qubit", shots=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        if mod is None:
            max = 2 ** len(output_wires)
        else:
            max = mod
        for x, y in zip(range(len(x_wires)), range(len(y_wires))):
            assert np.allclose(
                sum(bit * (2**i) for i, bit in enumerate(reversed(circuit(x, y)))), (x + y) % max
            )

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                9,
                [9, 10],
                "OutAdder must have at least enough wires to represent mod.",
            ),
        ],
    )
    def test_operation_error(self, x_wires, y_wires, output_wires, mod, work_wires, msg_match):
        """Test an error is raised when k or mod don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            OutAdder(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires", "msg_match"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [1, 10],
                "None of the wires in work_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [3, 10],
                "None of the wires in work_wires should be included in y_wires.",
            ),
            (
                [0, 1, 2],
                [2, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in x_wires.",
            ),
            (
                [0, 1, 2],
                [3, 7, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in y_wires should be included in output_wires.",
            ),
            (
                [0, 1, 7],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
                "None of the wires in x_wires should be included in output_wires.",
            ),
        ],
    )
    def test_wires_error(self, x_wires, y_wires, output_wires, mod, work_wires, msg_match):
        """Test an error is raised when some work_wires don't meet the requirements"""
        with pytest.raises(ValueError, match=msg_match):
            OutAdder(x_wires, y_wires, output_wires, mod, work_wires)

    @pytest.mark.parametrize(
        ("x_wires", "y_wires", "output_wires", "mod", "work_wires"),
        [
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                None,
                [9, 10],
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10],
            ),
            (
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                7,
                [9, 10, 11],
            ),
            (
                [0, 1, 2],
                [3, 5],
                [6, 8],
                2,
                [9, 10],
            ),
        ],
    )
    def test_decomposition(self, x_wires, y_wires, output_wires, mod, work_wires):
        """Test that compute_decomposition and decomposition work as expected."""
        adder_decomposition = OutAdder(
            x_wires, y_wires, output_wires, mod, work_wires
        ).compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires)
        op_list = []
        if mod != 2 ** len(output_wires):
            qft_new_output_wires = work_wires[:1] + output_wires
        else:
            qft_new_output_wires = output_wires
        for i in range(len(y_wires)):
            op_list.append(qml.CNOT(wires=[y_wires[i], output_wires[i]]))
        op_list.append(qml.QFT(wires=qft_new_output_wires))
        op_list.append(
            qml.ControlledSequence(PhaseAdder(1, output_wires, mod, work_wires), control=x_wires)
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_new_output_wires))

        for op1, op2 in zip(adder_decomposition, op_list):
            qml.assert_equal(op1, op2)

    @pytest.mark.jax
    def test_jit_compatible(self):
        """Test that the template is compatible with the JIT compiler."""

        import jax

        jax.config.update("jax_enable_x64", True)

        x, y = 2, 3

        # x, y in binary
        x_list = [0, 1, 0]
        y_list = [0, 1, 1]
        mod = 12
        x_wires = [0, 1]
        y_wires = [2, 3]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]
        dev = qml.device("default.qubit", shots=1)

        @jax.jit
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

        assert jax.numpy.allclose(
            sum(bit * (2**i) for i, bit in enumerate(reversed(circuit()))), (x + y) % mod
        )
