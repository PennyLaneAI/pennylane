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
from pennylane.transforms.decompositions.two_qubit_unitary import (
    _convert_to_su4,
    _su2su2_to_tensor_products,
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


# Randomly generated set (scipy.unitary_group) of five SU(4) operations.
test_u4_unitaries = [
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


test_su4_unitaries = [
    [
        [
            0.07765924 - 0.3053435j,
            -0.00544346 + 0.17785393j,
            -0.37819506 - 0.06769805j,
            -0.75382791 - 0.39140988j,
        ],
        [
            0.45849817 - 0.20446929j,
            0.40911688 - 0.66827471j,
            -0.29633505 - 0.14611682j,
            0.05442742 + 0.14790906j,
        ],
        [
            -0.36161974 + 0.09483265j,
            0.15292432 - 0.52479805j,
            0.59885223 - 0.05933094j,
            -0.35549461 - 0.27003429j,
        ],
        [
            -0.64499777 + 0.30483098j,
            0.18721944 - 0.1432504j,
            -0.59682685 - 0.15771633j,
            0.17092981 - 0.1587149j,
        ],
    ],
    [
        [
            -0.26266621 - 0.02053405j,
            -0.07876123 - 0.66084098j,
            -0.45369467 + 0.06028697j,
            -0.29947378 + 0.43418056j,
        ],
        [
            -0.15519815 - 0.58508147j,
            -0.10044503 + 0.27880786j,
            0.24651271 - 0.33162726j,
            0.03370782 + 0.61146448j,
        ],
        [
            0.60582544 + 0.25418538j,
            0.37446127 - 0.25617924j,
            0.07773588 - 0.18239669j,
            0.37852024 + 0.42417838j,
        ],
        [
            0.20461256 + 0.30112632j,
            -0.48479767 + 0.16847851j,
            -0.33212878 - 0.68568253j,
            -0.146386 - 0.04630265j,
        ],
    ],
    [
        [
            -0.19357534 + 0.05415218j,
            0.33159102 + 0.00393342j,
            -0.08674344 - 0.48530026j,
            0.77860264 - 0.01911054j,
        ],
        [
            -0.03953123 - 0.55929504j,
            -0.06065906 + 0.25475833j,
            -0.72320791 - 0.19893665j,
            -0.14649365 + 0.18160162j,
        ],
        [
            0.79678459 + 0.07820277j,
            0.29286639 + 0.05133j,
            -0.20355269 + 0.33959993j,
            0.26188743 + 0.21275882j,
        ],
        [
            0.00690914 - 0.06475621j,
            0.62854762 + 0.58135445j,
            0.18380884 - 0.06169495j,
            -0.29156472 - 0.37431518j,
        ],
    ],
    [
        [
            -0.74150838 - 0.31836676j,
            -0.19824177 + 0.11411013j,
            0.2105543 - 0.44404134j,
            0.23175558 - 0.03564448j,
        ],
        [
            0.14464037 + 0.01962981j,
            -0.32534365 + 0.34440049j,
            -0.0263518 - 0.38725266j,
            -0.60030466 + 0.49316216j,
        ],
        [
            -0.16758688 - 0.12435341j,
            -0.17350706 - 0.74956269j,
            -0.53983324 - 0.08065496j,
            -0.11493267 + 0.2310136j,
        ],
        [
            0.07028002 - 0.52821714j,
            -0.07096366 + 0.35529443j,
            -0.55262688 + 0.0671827j,
            -0.18126627 - 0.49194507j,
        ],
    ],
    [
        [
            -0.29171103 - 0.0636464j,
            -0.48846467 - 0.36067209j,
            -0.22793317 + 0.4548702j,
            0.27088407 - 0.45818459j,
        ],
        [
            -0.27108482 + 0.27382381j,
            0.15501384 + 0.26807221j,
            -0.69215901 - 0.30088389j,
            -0.25574673 - 0.34730381j,
        ],
        [
            0.16554448 - 0.21880187j,
            -0.54233066 + 0.48166767j,
            -0.39824354 + 0.06232072j,
            0.13836446 + 0.46579821j,
        ],
        [
            0.45296585 - 0.69421183j,
            0.07965187 + 0.05436182j,
            -0.08924185 + 0.03284008j,
            -0.32488413 - 0.43474127j,
        ],
    ],
]

test_su2_su2_pairs = [
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
        [[0, -1j], [-1j, 0],],
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
        [[0, -1j], [-1j, 0],],
    ),
]


class TestTwoQubitUnitaryDecomposition:
    """Test that two-qubit unitary operations are correctly decomposed."""

    @pytest.mark.parametrize("U", test_u4_unitaries)
    def test_convert_to_su4(self, U):
        """Test SU(2) x SU(2) can be correctly factored into tensor products."""
        U_su4 = _convert_to_su4(U)

        # Ensure the determinant is correct
        assert qml.math.isclose(qml.math.linalg.det(U_su4), 1.0)

        # Ensure the matrix is equivalent up to a global phase
        up_to_id = qml.math.dot(U, qml.math.conj(qml.math.T(U_su4)))
        assert qml.math.allclose(qml.math.eye(4), up_to_id / up_to_id[0, 0])

    @pytest.mark.parametrize("U_pair", test_su2_su2_pairs)
    def test_su2su2_to_tensor_products(self, U_pair):
        """Test SU(2) x SU(2) can be correctly factored into tensor products."""
        true_matrix = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        A, B = _su2su2_to_tensor_products(true_matrix)

        assert check_matrix_equivalence(qml.math.kron(A, B), true_matrix)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", test_su4_unitaries + test_u4_unitaries)
    def test_two_qubit_decomposition(self, U, wires):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        U = np.array(U)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        # We check with a slightly great tolerance threshold here simply because the
        # test matrices were copied in here with reduced precision.
        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", test_su2_su2_pairs)
    def test_two_qubit_decomposition_tensor_products(self, U_pair, wires):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        U = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)


class TestTwoQubitUnitaryDecompositionInterfaces:
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", test_su4_unitaries + test_u4_unitaries)
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
    @pytest.mark.parametrize("U", test_su4_unitaries + test_u4_unitaries)
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
    @pytest.mark.parametrize("U", test_su4_unitaries + test_u4_unitaries)
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
