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

# Randomly-generated U(4) elements that require 2 CNOTs
samples_u4_2_cnots = [
    # (U1 \otimes U2) CNOT01 (U3 \otimes U4) CNOT10
    [
        [
            0.21519064 + 0.03018591j,
            -0.23207023 + 0.14092543j,
            -0.53689353 - 0.2946668j,
            0.51445261 + 0.489204j,
        ],
        [
            0.74139151 - 0.11264103j,
            0.31749154 + 0.50217117j,
            0.0925797 - 0.08233009j,
            -0.26285568 + 0.01521574j,
        ],
        [
            0.52541986 + 0.14967089j,
            0.00882372 - 0.58434132j,
            0.37824777 + 0.25806244j,
            0.37103602 + 0.11253721j,
        ],
        [
            0.30028938 + 0.04232397j,
            -0.10120803 - 0.47067076j,
            -0.63182808 - 0.02632943j,
            -0.42481116 - 0.30967153j,
        ],
    ],
    # (U1 \otimes U2) CNOT01 (U3 \otimes U4) CNOT01
    [
        [
            -0.04472714 + 0.01919032j,
            -0.57602746 + 0.70197946j,
            -0.38567309 - 0.02769067j,
            0.01695965 - 0.15248041j,
        ],
        [
            0.88781894 - 0.06836099j,
            -0.06641031 + 0.11848866j,
            0.08169273 + 0.02798384j,
            0.25129416 + 0.34358238j,
        ],
        [
            0.35060107 + 0.21292471j,
            -0.01047688 - 0.06841495j,
            0.20642614 + 0.0491272j,
            -0.01499298 - 0.88413888j,
        ],
        [
            -0.18802799 + 0.03351033j,
            -0.18365632 + 0.34416172j,
            0.88848782 - 0.09271919j,
            0.03261441 + 0.11079005j,
        ],
    ],
    #  CNOT01 (U1 \otimes U2) CNOT01 (U3 \otimes U4)
    [
        [
            0.27790456 - 0.52018253j,
            0.05793464 + 0.38411653j,
            -0.15069552 - 0.25864196j,
            -0.58470268 - 0.26418787j,
        ],
        [
            -0.33486277 + 0.08420967j,
            -0.36088137 + 0.48971641j,
            0.20761839 + 0.63264891j,
            -0.21743312 - 0.14174246j,
        ],
        [
            0.03505601 - 0.6247698j,
            0.32838083 + 0.10638574j,
            0.28089091 + 0.26934482j,
            0.55756423 - 0.16418792j,
        ],
        [
            -0.04913217 + 0.37279303j,
            0.53282896 + 0.27564823j,
            -0.54177 + 0.14860717j,
            0.1360818 - 0.40571624j,
        ],
    ],
    #  (U1 \otimes U2) CNOT01 (U3 \otimes U4)  CNOT01 (U5 \otimes U6)
    [
        [
            0.25652439 - 0.0462842j,
            0.67390778 - 0.31620797j,
            0.4650931 - 0.2908297j,
            0.05553787 - 0.27191151j,
        ],
        [
            -0.40551672 + 0.66099793j,
            0.0045791 - 0.48095684j,
            0.24088629 + 0.22488846j,
            -0.05822632 + 0.23517259j,
        ],
        [
            0.55224042 - 0.03087806j,
            0.15616954 - 0.1797145j,
            -0.15345415 + 0.51479092j,
            -0.5573751 + 0.19536115j,
        ],
        [
            0.14789508 + 0.05380565j,
            0.34517639 - 0.19669112j,
            -0.54821596 - 0.03750822j,
            0.5720388 + 0.43384543j,
        ],
    ],
    #  (U1 \otimes U2) CNOT10 (U3 \otimes U4)  CNOT01 (U5 \otimes U6)
    [
        [
            0.80915358 + 0.17873994j,
            -0.01135281 + 0.27634758j,
            -0.32600107 + 0.20030231j,
            0.2127876 + 0.21248383j,
        ],
        [
            -0.14325396 + 0.23855296j,
            0.78044867 + 0.10607667j,
            -0.38158405 - 0.07363528j,
            0.10640225 - 0.37398987j,
        ],
        [
            -0.00320404 - 0.14542437j,
            0.49548943 + 0.03676195j,
            0.11351799 - 0.12191885j,
            -0.35677139 + 0.75956824j,
        ],
        [
            0.29664416 + 0.35600146j,
            0.23301292 - 0.04465991j,
            0.80142543 - 0.18038982j,
            0.18367409 - 0.1428856j,
        ],
    ],
]

# samples_u4_2_cnots = []

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
        U_su4 = _convert_to_su4(U)

        # Ensure the determinant is correct
        assert qml.math.isclose(qml.math.linalg.det(U_su4), 1.0)

        # Ensure the matrix is equivalent up to a global phase
        up_to_id = qml.math.dot(U, qml.math.conj(qml.math.T(U_su4)))
        assert qml.math.allclose(qml.math.eye(4), up_to_id / up_to_id[0, 0])

    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_su2su2_to_tensor_products(self, U_pair):
        """Test SU(2) x SU(2) can be correctly factored into tensor products."""
        true_matrix = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        A, B = _su2su2_to_tensor_products(true_matrix)

        assert check_matrix_equivalence(qml.math.kron(A, B), true_matrix)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4_2_cnots + samples_u4)
    def test_two_qubit_decomposition(self, U, wires):
        """Test that a two-qubit matrix in isolation is correctly decomposed."""
        U = np.array(U)

        print(U)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        print(obtained_decomposition)

        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        # We check with a slightly great tolerance threshold here simply because the
        # test matrices were copied in here with reduced precision.
        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products(self, U_pair, wires):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        U = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        obtained_matrix = compute_matrix_from_ops_two_qubit(obtained_decomposition, wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)


class TestTwoQubitUnitaryDecompositionInterfaces:
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4_2_cnots + samples_u4)
    def test_two_qubit_decomposition_torch(self, U, wires):
        """Test that a two-qubit operation in Torch is correctly decomposed."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        obtained_matrix = compute_matrix_from_ops_two_qubit(
            obtained_decomposition, wire_order=wires
        )

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    # @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    # @pytest.mark.parametrize("U", samples_u4_2_cnots + samples_u4)
    # def test_two_qubit_decomposition_tf(self, U, wires):
    #     """Test that a two-qubit operation in Tensorflow is correctly decomposed."""
    #     tf = pytest.importorskip("tensorflow")

    #     U = tf.Variable(U, dtype=tf.complex128)

    #     obtained_decomposition = two_qubit_decomposition(U, wires=wires)

    #     obtained_matrix = compute_matrix_from_ops_two_qubit(
    #         obtained_decomposition, wire_order=wires
    #     )

    #     assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_u4_2_cnots + samples_u4)
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
