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
"""
Tests for the QubitUnitary decomposition transforms.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.wires import Wires

from pennylane.transforms.decompositions import zyz_decomposition
from pennylane.transforms.decompositions import two_qubit_decomposition
from pennylane.transforms.decompositions.utils import (
    _convert_to_su4,
    _su2su2_to_tensor_products,
    _compute_num_cnots,
)

from test_optimization.utils import compute_matrix_from_ops_two_qubit, check_matrix_equivalence

from gate_data import I, Z, S, T, H, X, CNOT, SWAP

single_qubit_decomps = [
    # First set of gates are diagonal and converted to RZ
    (I, qml.RZ, [0.0]),
    (Z, qml.RZ, [np.pi]),
    (S, qml.RZ, [np.pi / 2]),
    (T, qml.RZ, [np.pi / 4]),
    (qml.RZ(0.3, wires=0).matrix, qml.RZ, [0.3]),
    (qml.RZ(-0.5, wires=0).matrix, qml.RZ, [-0.5]),
    # Next set of gates are non-diagonal and decomposed as Rots
    (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
    (X, qml.Rot, [0.0, np.pi, np.pi]),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix, qml.Rot, [0.2, 0.5, -0.3]),
    (np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix, qml.Rot, [-1.0, 2.0, -3.0]),
]


class TestQubitUnitaryZYZDecomposition:
    """Test that the decompositions are correct."""

    def test_zyz_decomposition_invalid_input(self):
        """Test that non-unitary operations throw errors when we try to decompose."""
        with pytest.raises(ValueError, match="Operator must be unitary"):
            zyz_decomposition(I + H, Wires("a"))

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition(self, U, expected_gate, expected_params):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        obtained_gates = zyz_decomposition(U, Wires("a"))

        assert len(obtained_gates) == 1

        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(obtained_gates[0].parameters, expected_params)

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_torch(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Torch is correctly decomposed."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex64)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(
            [x.detach() for x in obtained_gates[0].parameters], expected_params
        )

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_tf(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Tensorflow is correctly decomposed."""
        tf = pytest.importorskip("tensorflow")

        U = tf.Variable(U, dtype=tf.complex64)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose([x.numpy() for x in obtained_gates[0].parameters], expected_params)

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_jax(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in JAX is correctly decomposed."""
        jax = pytest.importorskip("jax")

        U = jax.numpy.array(U, dtype=jax.numpy.complex64)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")
        assert qml.math.allclose(
            [jax.numpy.asarray(x) for x in obtained_gates[0].parameters], expected_params
        )


# Randomly generated set (scipy.unitary_group) of five U(4) operations.
# These require 3 CNOTs each
samples_u4 = [
    [
        [
            -0.07016275 - 0.11813399j,
            -0.46476569 - 0.36887134j,
            -0.18641714 + 0.66322739j,
            -0.36131479 - 0.15452521j,
        ],
        [
            -0.21347395 - 0.47461873j,
            0.45781338 - 0.21930172j,
            0.16991759 - 0.12283128j,
            -0.59626019 + 0.26831674j,
        ],
        [
            -0.50100034 + 0.47448914j,
            0.14346598 + 0.41837463j,
            -0.39898589 + 0.01284601j,
            -0.40027516 - 0.09308024j,
        ],
        [
            0.47310527 + 0.10157557j,
            -0.4411336 - 0.00466688j,
            -0.2328747 - 0.51752603j,
            -0.46274728 + 0.18717456j,
        ],
    ],
    [
        [
            -0.41189319 + 0.06007113j,
            0.15316396 + 0.1458654j,
            -0.17064243 + 0.33405919j,
            0.60457809 + 0.52513855j,
        ],
        [
            0.08694733 + 0.64367692j,
            -0.17413963 + 0.61038522j,
            -0.27698195 - 0.31363701j,
            -0.01169698 - 0.0012099j,
        ],
        [
            -0.39488729 - 0.41241129j,
            0.2202738 + 0.19315024j,
            -0.45149927 - 0.30746997j,
            0.15876369 - 0.51435213j,
        ],
        [
            -0.27620095 - 0.05049386j,
            0.47854591 + 0.48737626j,
            0.6201202 - 0.03549654j,
            -0.25966725 + 0.03722163j,
        ],
    ],
    [
        [
            -0.34812515 + 0.37427723j,
            -0.11092236 + 0.47565307j,
            0.13724183 - 0.29504039j,
            -0.56249794 - 0.27908375j,
        ],
        [
            -0.14408107 + 0.1693212j,
            -0.20483797 - 0.10707915j,
            -0.85376825 + 0.3175112j,
            -0.1054503 - 0.23726165j,
        ],
        [
            0.03106625 - 0.04236712j,
            0.78292822 + 0.03053768j,
            0.01814738 + 0.16830002j,
            0.03513342 - 0.59451003j,
        ],
        [
            -0.00087219 - 0.82857442j,
            0.04840206 + 0.30294214j,
            -0.1884474 + 0.01468393j,
            -0.41353861 + 0.11227088j,
        ],
    ],
    [
        [
            -0.05780187 - 0.06284269j,
            0.13559069 + 0.19399748j,
            0.12381697 + 0.01612151j,
            0.71416466 - 0.64114599j,
        ],
        [
            -0.31103029 - 0.06658675j,
            -0.50183231 + 0.49812898j,
            -0.58061141 - 0.20451914j,
            -0.07379796 - 0.12030957j,
        ],
        [
            0.47241806 - 0.79298028j,
            0.14041019 + 0.06342211j,
            -0.27789855 + 0.19625469j,
            -0.07716877 - 0.05067088j,
        ],
        [
            0.11114093 - 0.16488557j,
            -0.12688073 + 0.63574829j,
            0.68327072 - 0.15122624j,
            -0.21697355 + 0.05813823j,
        ],
    ],
    [
        [
            0.32457875 - 0.36309659j,
            -0.21084067 + 0.48248995j,
            -0.21588245 - 0.42368088j,
            -0.0474262 + 0.50714809j,
        ],
        [
            -0.24688996 - 0.11890225j,
            0.16113004 + 0.1518989j,
            -0.40132738 - 0.28678782j,
            -0.63810805 - 0.4747406j,
        ],
        [
            -0.14474527 - 0.46561401j,
            0.47151308 - 0.32560877j,
            0.51600239 - 0.28298318j,
            -0.18491473 + 0.23103107j,
        ],
        [
            0.42624962 - 0.51795827j,
            -0.17138618 - 0.56213399j,
            -0.36133453 + 0.23168462j,
            -0.0167845 - 0.14191731j,
        ],
    ],
]

# These are randomly generated matrices that involve a single CNOT
samples_1_cnot = [
    # CNOT10 (A \otimes B)
    [
        [
            -0.40013501 + 0.6346169j,
            0.36588428 + 0.27955713j,
            -0.34061004 - 0.21797615j,
            -0.15200951 + 0.19619931j,
        ],
        [
            0.19619931 - 0.15200951j,
            0.21797615 + 0.34061004j,
            -0.27955713 - 0.36588428j,
            0.6346169 - 0.40013501j,
        ],
        [
            -0.04618249 + 0.40174118j,
            0.24303208 + 0.05036332j,
            0.74472919 + 0.09069221j,
            0.09646726 - 0.45024168j,
        ],
        [
            0.45024168 - 0.09646726j,
            0.09069221 + 0.74472919j,
            0.05036332 + 0.24303208j,
            -0.40174118 + 0.04618249j,
        ],
    ],
    # CNOT01 (A \otimes B)
    [
        [
            -0.65867779 - 0.39965996j,
            0.10480488 - 0.11993597j,
            0.33775094 + 0.50135151j,
            -0.11410732 + 0.0509635j,
        ],
        [
            -0.15620406 + 0.03112864j,
            0.0184181 + 0.77022418j,
            0.12151068 + 0.02920466j,
            0.24022431 - 0.55472634j,
        ],
        [
            0.0509635 - 0.11410732j,
            -0.50135151 - 0.33775094j,
            0.11993597 - 0.10480488j,
            -0.39965996 - 0.65867779j,
        ],
        [
            0.55472634 - 0.24022431j,
            0.02920466 + 0.12151068j,
            0.77022418 + 0.0184181j,
            -0.03112864 + 0.15620406j,
        ],
    ],
    # (A \otimes B) CNOT01
    [
        [
            0.61975417 + 0.05871326j,
            0.18490393 - 0.03060346j,
            0.11206333 - 0.18820514j,
            0.51965346 - 0.50922726j,
        ],
        [
            0.17568798 - 0.06526691j,
            -0.50876306 + 0.35875148j,
            -0.06315069 + 0.72482005j,
            0.07415843 - 0.20610651j,
        ],
        [
            0.72482005 - 0.06315069j,
            0.20610651 - 0.07415843j,
            -0.06526691 + 0.17568798j,
            -0.35875148 + 0.50876306j,
        ],
        [
            0.18820514 - 0.11206333j,
            -0.50922726 + 0.51965346j,
            -0.05871326 - 0.61975417j,
            -0.03060346 + 0.18490393j,
        ],
    ],
    # (A \otimes B) CNOT10
    [
        [
            0.19339868 + 0.08633274j,
            0.01466384 + 0.70171345j,
            -0.63869976 + 0.02205134j,
            0.09769795 - 0.21108831j,
        ],
        [
            0.18782762 - 0.1372004j,
            -0.56109965 - 0.30592619j,
            -0.32755953 + 0.62074274j,
            0.12258652 + 0.17271055j,
        ],
        [
            0.30592619 + 0.56109965j,
            0.1372004 - 0.18782762j,
            0.17271055 + 0.12258652j,
            0.62074274 - 0.32755953j,
        ],
        [
            0.70171345 + 0.01466384j,
            0.08633274 + 0.19339868j,
            0.21108831 - 0.09769795j,
            -0.02205134 + 0.63869976j,
        ],
    ],
    # (A \otimes B) CNOT10 (C \otimes D)
    [
        [
            0.78385585 + 0.24581188j,
            -0.08367428 + 0.34003966j,
            -0.16227836 - 0.08719991j,
            -0.222293 + 0.34520414j,
        ],
        [
            -0.40100084 - 0.10151918j,
            0.07269317 + 0.76372534j,
            -0.26873591 + 0.2188187j,
            -0.34389609 - 0.04434368j,
        ],
        [
            0.17914714 - 0.32406563j,
            0.2474564 - 0.0645854j,
            0.45026225 - 0.23004978j,
            -0.66076185 - 0.32437111j,
        ],
        [
            0.04747281 + 0.12113138j,
            0.4725868 + 0.00816913j,
            -0.53596566 - 0.55049352j,
            0.1451332 - 0.38510071j,
        ],
    ],
    # (A \otimes B) CNOT01 (C \otimes D)
    [
        [
            0.38565489 - 0.50840093j,
            -0.30818989 - 0.11768624j,
            0.27524652 - 0.39173492j,
            -0.5046883 - 0.00636884j,
        ],
        [
            -0.05689053 - 0.29547371j,
            0.51930406 - 0.09499063j,
            -0.40250308 - 0.33321454j,
            0.00587892 - 0.59806656j,
        ],
        [
            0.56972698 - 0.14750915j,
            -0.32434509 + 0.18005613j,
            -0.62729996 + 0.30972331j,
            0.15801266 - 0.04037939j,
        ],
        [
            -0.37603427 + 0.12044912j,
            -0.64676631 + 0.23778647j,
            0.03588508 - 0.08379105j,
            0.02300689 - 0.60033588j,
        ],
    ],
]

# Randomly-generated SU(2) x SU(2) matrices. These can be used to test
# the 0-CNOT decomposition case
samples_su2_su2 = [
    (
        [
            [0.21993927 - 0.1111822j, -0.27174921 - 0.93027824j],
            [0.27174921 - 0.93027824j, 0.21993927 + 0.1111822j],
        ],
        [
            [0.86361715 + 0.11195238j, 0.35098794 - 0.34415046j],
            [-0.35098794 - 0.34415046j, 0.86361715 - 0.11195238j],
        ],
    ),
    (
        [
            [0.08652981 - 0.25976406j, 0.95775011 + 0.08803377j],
            [-0.95775011 + 0.08803377j, 0.08652981 + 0.25976406j],
        ],
        [
            [-0.1668335 + 0.92244968j, -0.34571207 + 0.04166941j],
            [0.34571207 + 0.04166941j, -0.1668335 - 0.92244968j],
        ],
    ),
    (
        [
            [-0.36170896 - 0.92179965j, -0.1390035 + 0.01140442j],
            [0.1390035 + 0.01140442j, -0.36170896 + 0.92179965j],
        ],
        [
            [-0.26587616 + 0.09917768j, 0.731434 + 0.62006287j],
            [-0.731434 + 0.62006287j, -0.26587616 - 0.09917768j],
        ],
    ),
    (
        [
            [0.90778489 + 0.27105877j, 0.17141891 - 0.27031333j],
            [-0.17141891 - 0.27031333j, 0.90778489 - 0.27105877j],
        ],
        [
            [0.72380105 - 0.58082557j, -0.18751423 + 0.32185728j],
            [0.18751423 + 0.32185728j, 0.72380105 + 0.58082557j],
        ],
    ),
    (
        [
            [-0.11704772 + 0.39598718j, -0.89480902 - 0.16973743j],
            [0.89480902 - 0.16973743j, -0.11704772 - 0.39598718j],
        ],
        [[np.exp(-1j * np.pi / 3), 0], [0, np.exp(1j * np.pi / 3)]],
    ),
    (
        [
            [0, -1j],
            [-1j, 0],
        ],
        [
            [-0.05594177 - 0.02155518j, -0.83242096 - 0.55089131j],
            [0.83242096 - 0.55089131j, -0.05594177 + 0.02155518j],
        ],
    ),
    (
        [
            [0.90778489 + 0.27105877j, 0.17141891 - 0.27031333j],
            [-0.17141891 - 0.27031333j, 0.90778489 - 0.27105877j],
        ],
        [
            [0, -1j],
            [-1j, 0],
        ],
    ),
]


class TestTwoQubitUnitaryDecomposition:
    """Test that two-qubit unitary operations are correctly decomposed."""

    @pytest.mark.parametrize("U", samples_u4)
    def test_convert_to_su4(self, U):
        """Test a matrix in U(4) is correct converted to SU(4)."""
        U_su4 = _convert_to_su4(np.array(U))

        # Ensure the determinant is correct and the mats are equivalent up to a phase
        assert qml.math.isclose(qml.math.linalg.det(U_su4), 1.0)
        assert check_matrix_equivalence(np.array(U), U_su4)

    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_su2su2_to_tensor_products(self, U_pair):
        """Test SU(2) x SU(2) can be correctly factored into tensor products."""
        true_matrix = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        A, B = _su2su2_to_tensor_products(true_matrix)

        assert check_matrix_equivalence(qml.math.kron(A, B), true_matrix)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4)
    def test_two_qubit_decomposition_3_cnots(self, U, wires):
        """Test that a two-qubit matrix in isolation is correctly decomposed."""
        U = _convert_to_su4(np.array(U))

        assert _compute_num_cnots(U) == 3

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        # We check with a slightly great tolerance threshold here simply because the
        # test matrices were copied in here with reduced precision.
        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_1_cnot)
    def test_two_qubit_decomposition_1_cnot(self, U, wires):
        """Test that a two-qubit matrix in isolation is correctly decomposed."""
        U = _convert_to_su4(np.array(U))

        assert _compute_num_cnots(U) == 1

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        # We check with a slightly great tolerance threshold here simply because the
        # test matrices were copied in here with reduced precision.
        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    def test_two_qubit_decomposition_invalid_two_cnots(self):
        # Randomly-generated unitary of the form CNOT U1 \otimes U2) CNOT (U3 \otimes U4)
        U = _convert_to_su4(
            np.array(
                [
                    [
                        -0.08737018 - 0.2335028j,
                        -0.10041547 + 0.63791828j,
                        -0.01502992 - 0.54660066j,
                        0.04526445 - 0.46879937j,
                    ],
                    [
                        -0.43062885 - 0.32278065j,
                        -0.08235743 + 0.38761533j,
                        0.42067305 + 0.36484914j,
                        -0.38344969 + 0.31020747j,
                    ],
                    [
                        -0.32496269 - 0.15387743j,
                        -0.2081725 - 0.59008906j,
                        0.14062425 - 0.50123501j,
                        -0.44835313 - 0.08454499j,
                    ],
                    [
                        0.71918496 + 0.04133173j,
                        0.08993026 + 0.16223525j,
                        0.20852686 - 0.27645241j,
                        -0.47595359 + 0.31656869j,
                    ],
                ]
            )
        )

        assert _compute_num_cnots(U) == 2

        with pytest.warns(UserWarning, match="1 or 2 CNOTs is not currently supported"):
            two_qubit_decomposition(U, wires=[0, 1])

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products(self, U_pair, wires):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        U = _convert_to_su4(qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1])))

        assert _compute_num_cnots(U) == 0

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)


class TestTwoQubitUnitaryDecompositionInterfaces:
    """Test the decomposition in the non-autograd interfaces."""

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4 + samples_1_cnot)
    def test_two_qubit_decomposition_torch(self, U, wires):
        """Test that a two-qubit operation in Torch is correctly decomposed."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_torch(self, U_pair, wires):
        """Test that a two-qubit tensor product in Torch is correctly decomposed."""
        torch = pytest.importorskip("torch")

        U1 = torch.tensor(U_pair[0], dtype=torch.complex128)
        U2 = torch.tensor(U_pair[1], dtype=torch.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4 + samples_1_cnot)
    def test_two_qubit_decomposition_tf(self, U, wires):
        """Test that a two-qubit operation in Tensorflow is correctly decomposed."""
        tf = pytest.importorskip("tensorflow")

        U = tf.Variable(U, dtype=tf.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_tf(self, U_pair, wires):
        """Test that a two-qubit tensor product in Tensorflow is correctly decomposed."""
        tf = pytest.importorskip("tensorflow")

        U1 = tf.Variable(U_pair[0], dtype=tf.complex128)
        U2 = tf.Variable(U_pair[1], dtype=tf.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4 + samples_1_cnot)
    def test_two_qubit_decomposition_jax(self, U, wires):
        """Test that a two-qubit operation in JAX is correctly decomposed."""
        jax = pytest.importorskip("jax")

        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_jax(self, U_pair, wires):
        """Test that a two-qubit tensor product in JAX is correctly decomposed."""
        jax = pytest.importorskip("jax")

        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U1 = jax.numpy.array(U_pair[0], dtype=jax.numpy.complex128)
        U2 = jax.numpy.array(U_pair[1], dtype=jax.numpy.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)
