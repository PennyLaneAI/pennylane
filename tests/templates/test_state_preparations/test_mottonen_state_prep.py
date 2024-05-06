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
Unit tests for the ArbitraryStatePreparation template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates.state_preparations.mottonen import _get_alpha_y, gray_code


def test_standard_validity():
    """Check the operation using the assert_valid function."""

    state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
    state = state / np.linalg.norm(state)

    op = qml.MottonenStatePreparation(state_vector=state, wires=range(3))

    qml.ops.functions.assert_valid(op)


class TestHelpers:
    """Tests the helper functions for classical pre-processsing."""

    # fmt: off
    @pytest.mark.parametrize("rank,expected_gray_code", [
        (1, ['0', '1']),
        (2, ['00', '01', '11', '10']),
        (3, ['000', '001', '011', '010', '110', '111', '101', '100']),
    ])
    # fmt: on
    def test_gray_code(self, rank, expected_gray_code):
        """Tests that the function gray_code generates the proper
        Gray code of given rank."""

        assert gray_code(rank) == expected_gray_code

    @pytest.mark.parametrize(
        "current_qubit, expected",
        [
            (1, np.array([0, 0, 0, 1.23095942])),
            (2, np.array([2.01370737, 3.14159265])),
            (3, np.array([1.15927948])),
        ],
    )
    def test_get_alpha_y(self, current_qubit, expected, tol):
        """Test the _get_alpha_y helper function."""

        state = np.array([np.sqrt(0.2), 0, np.sqrt(0.5), 0, 0, 0, np.sqrt(0.2), np.sqrt(0.1)])
        res = _get_alpha_y(state, 3, current_qubit)
        assert np.allclose(res, expected, atol=tol)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires,target_state", [
        ([1, 0], 0, [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [0], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [1], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1], [0], [0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 1], [1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 1], [2], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 2], [0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 1], [1, 2], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1j, 0, 0, 0], [0, 1, 2], [0, 0, 0, 0, 1j, 0, 0, 0]),
        ([1 / 2, 0, 0, 0, 1 / 2, 1j / 2, -1 / 2, 0], [0, 1, 2], [1 / 2, 0, 0, 0, 1 / 2, 1j / 2, -1 / 2, 0]),
        ([1 / 3, 0, 0, 0, 2j / 3, 2j / 3, 0, 0], [0, 1, 2], [1 / 3, 0, 0, 0, 2j / 3, 2j / 3, 0, 0]),
        ([2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3], [0, 1, 2], [2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3]),
        (
                [1 / np.sqrt(8), 1j / np.sqrt(8), 1 / np.sqrt(8), -1j / np.sqrt(8), 1 / np.sqrt(8),
                 1 / np.sqrt(8), 1 / np.sqrt(8), 1j / np.sqrt(8)],
                [0, 1, 2],
                [1 / np.sqrt(8), 1j / np.sqrt(8), 1 / np.sqrt(8), -1j / np.sqrt(8), 1 / np.sqrt(8),
                 1 / np.sqrt(8), 1 / np.sqrt(8), 1j / np.sqrt(8)],
        ),
        (
                [-0.17133152 - 0.18777771j, 0.00240643 - 0.40704011j, 0.18684538 - 0.36315606j, -0.07096948 + 0.104501j,
                 0.30357755 - 0.23831927j, -0.38735106 + 0.36075556j, 0.12351096 - 0.0539908j,
                 0.27942828 - 0.24810483j],
                [0, 1, 2],
                [-0.17133152 - 0.18777771j, 0.00240643 - 0.40704011j, 0.18684538 - 0.36315606j, -0.07096948 + 0.104501j,
                 0.30357755 - 0.23831927j, -0.38735106 + 0.36075556j, 0.12351096 - 0.0539908j,
                 0.27942828 - 0.24810483j],
        ),
        (
                [-0.29972867 + 0.04964242j, -0.28309418 + 0.09873227j, 0.00785743 - 0.37560696j,
                 -0.3825148 + 0.00674343j, -0.03008048 + 0.31119167j, 0.03666351 - 0.15935903j,
                 -0.25358831 + 0.35461265j, -0.32198531 + 0.33479292j],
                [0, 1, 2],
                [-0.29972867 + 0.04964242j, -0.28309418 + 0.09873227j, 0.00785743 - 0.37560696j,
                 -0.3825148 + 0.00674343j, -0.03008048 + 0.31119167j, 0.03666351 - 0.15935903j,
                 -0.25358831 + 0.35461265j, -0.32198531 + 0.33479292j],
        ),
        (
                [-0.39340123 + 0.05705932j, 0.1980509 - 0.24234781j, 0.27265585 - 0.0604432j, -0.42641249 + 0.25767258j,
                 0.40386614 - 0.39925987j, 0.03924761 + 0.13193724j, -0.06059103 - 0.01753834j,
                 0.21707136 - 0.15887973j],
                [0, 1, 2],
                [-0.39340123 + 0.05705932j, 0.1980509 - 0.24234781j, 0.27265585 - 0.0604432j, -0.42641249 + 0.25767258j,
                 0.40386614 - 0.39925987j, 0.03924761 + 0.13193724j, -0.06059103 - 0.01753834j,
                 0.21707136 - 0.15887973j],
        ),
        (
                [-1.33865287e-01 + 0.09802308j, 1.25060033e-01 + 0.16087698j, -4.14678130e-01 - 0.00774832j,
                 1.10121136e-01 + 0.37805482j, -3.21284864e-01 + 0.21521063j, -2.23121454e-04 + 0.28417422j,
                 5.64131205e-02 + 0.38135286j, 2.32694503e-01 + 0.41331133j],
                [0, 1, 2],
                [-1.33865287e-01 + 0.09802308j, 1.25060033e-01 + 0.16087698j, -4.14678130e-01 - 0.00774832j,
                 1.10121136e-01 + 0.37805482j, -3.21284864e-01 + 0.21521063j, -2.23121454e-04 + 0.28417422j,
                 5.64131205e-02 + 0.38135286j, 2.32694503e-01 + 0.41331133j],
        ),
        ([1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0], [0, 1, 2],
         [1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0]),
        ([1 / 2, 0, 1j / 2, 1j / np.sqrt(2)], [0, 1], [1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0]),
    ])
    # fmt: on
    def test_state_preparation(self, state_vector, wires, target_state):
        """Tests that the template produces correct states."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit():
            qml.MottonenStatePreparation(state_vector, wires)
            return qml.state()

        state = circuit()

        assert np.allclose(state, target_state)

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires,target_state", [
        ([1, 0], 0, [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [0], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [1], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1], [0], [0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 1], [1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 1], [2], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 2], [0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 1], [1, 2], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1j, 0, 0, 0], [0, 1, 2], [0, 0, 0, 0, 1j, 0, 0, 0]),
        ([1 / 2, 0, 0, 0, 1 / 2, 1j / 2, -1 / 2, 0], [0, 1, 2], [1 / 2, 0, 0, 0, 1 / 2, 1j / 2, -1 / 2, 0]),
        ([1 / 3, 0, 0, 0, 2j / 3, 2j / 3, 0, 0], [0, 1, 2], [1 / 3, 0, 0, 0, 2j / 3, 2j / 3, 0, 0]),
        ([2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3], [0, 1, 2], [2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3]),
        (
                [1 / np.sqrt(8), 1j / np.sqrt(8), 1 / np.sqrt(8), -1j / np.sqrt(8), 1 / np.sqrt(8),
                 1 / np.sqrt(8), 1 / np.sqrt(8), 1j / np.sqrt(8)],
                [0, 1, 2],
                [1 / np.sqrt(8), 1j / np.sqrt(8), 1 / np.sqrt(8), -1j / np.sqrt(8), 1 / np.sqrt(8),
                 1 / np.sqrt(8), 1 / np.sqrt(8), 1j / np.sqrt(8)],
        ),
        (
                [-0.17133152 - 0.18777771j, 0.00240643 - 0.40704011j, 0.18684538 - 0.36315606j, -0.07096948 + 0.104501j,
                 0.30357755 - 0.23831927j, -0.38735106 + 0.36075556j, 0.12351096 - 0.0539908j,
                 0.27942828 - 0.24810483j],
                [0, 1, 2],
                [-0.17133152 - 0.18777771j, 0.00240643 - 0.40704011j, 0.18684538 - 0.36315606j, -0.07096948 + 0.104501j,
                 0.30357755 - 0.23831927j, -0.38735106 + 0.36075556j, 0.12351096 - 0.0539908j,
                 0.27942828 - 0.24810483j],
        ),
        (
                [-0.29972867 + 0.04964242j, -0.28309418 + 0.09873227j, 0.00785743 - 0.37560696j,
                 -0.3825148 + 0.00674343j, -0.03008048 + 0.31119167j, 0.03666351 - 0.15935903j,
                 -0.25358831 + 0.35461265j, -0.32198531 + 0.33479292j],
                [0, 1, 2],
                [-0.29972867 + 0.04964242j, -0.28309418 + 0.09873227j, 0.00785743 - 0.37560696j,
                 -0.3825148 + 0.00674343j, -0.03008048 + 0.31119167j, 0.03666351 - 0.15935903j,
                 -0.25358831 + 0.35461265j, -0.32198531 + 0.33479292j],
        ),
        (
                [-0.39340123 + 0.05705932j, 0.1980509 - 0.24234781j, 0.27265585 - 0.0604432j, -0.42641249 + 0.25767258j,
                 0.40386614 - 0.39925987j, 0.03924761 + 0.13193724j, -0.06059103 - 0.01753834j,
                 0.21707136 - 0.15887973j],
                [0, 1, 2],
                [-0.39340123 + 0.05705932j, 0.1980509 - 0.24234781j, 0.27265585 - 0.0604432j, -0.42641249 + 0.25767258j,
                 0.40386614 - 0.39925987j, 0.03924761 + 0.13193724j, -0.06059103 - 0.01753834j,
                 0.21707136 - 0.15887973j],
        ),
        (
                [-1.33865287e-01 + 0.09802308j, 1.25060033e-01 + 0.16087698j, -4.14678130e-01 - 0.00774832j,
                 1.10121136e-01 + 0.37805482j, -3.21284864e-01 + 0.21521063j, -2.23121454e-04 + 0.28417422j,
                 5.64131205e-02 + 0.38135286j, 2.32694503e-01 + 0.41331133j],
                [0, 1, 2],
                [-1.33865287e-01 + 0.09802308j, 1.25060033e-01 + 0.16087698j, -4.14678130e-01 - 0.00774832j,
                 1.10121136e-01 + 0.37805482j, -3.21284864e-01 + 0.21521063j, -2.23121454e-04 + 0.28417422j,
                 5.64131205e-02 + 0.38135286j, 2.32694503e-01 + 0.41331133j],
        ),
        ([1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0], [0, 1, 2],
         [1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0]),
        ([1 / 2, 0, 1j / 2, 1j / np.sqrt(2)], [0, 1], [1 / 2, 0, 0, 0, 1j / 2, 0, 1j / np.sqrt(2), 0]),
    ])
    # fmt: on
    def test_state_preparation_probability_distribution(
        self, tol, state_vector, wires, target_state
    ):
        """Tests that the template produces states with correct probability distribution."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit():
            qml.MottonenStatePreparation(state_vector, wires)
            return (
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2)),
                qml.state(),
            )

        results = circuit()

        state = results[-1].ravel()

        probabilities = np.abs(state) ** 2
        target_probabilities = np.abs(target_state) ** 2

        assert np.allclose(probabilities, target_probabilities, atol=tol, rtol=0)

    # fmt: off
    @pytest.mark.parametrize("state_vector, n_wires", [
        ([1 / 2, 1 / 2, 1 / 2, 1 / 2], 2),
        ([1, 0, 0, 0], 2),
        ([0, 1, 0, 0], 2),
        ([0, 0, 0, 1], 2),
        ([0, 1, 0, 0, 0, 0, 0, 0], 3),
        ([0, 0, 0, 0, 1, 0, 0, 0], 3),
        ([2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3], 3),
        ([1 / 2, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, 0], 3),
        ([1 / 3, 0, 0, 0, 2 / 3, 2 / 3, 0, 0], 3),
        ([2 / 3, 0, 0, 0, 1 / 3, 0, 0, 2 / 3], 3),
    ])
    # fmt: on
    def test_RZ_skipped(self, mocker, state_vector, n_wires):
        """Tests that the cascade of RZ gates is skipped for real-valued states."""

        n_CNOT = 2**n_wires - 2

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="autograd")
        def circuit(state_vector):
            qml.MottonenStatePreparation(state_vector, wires=range(n_wires))
            return qml.expval(qml.PauliX(wires=0))

        # when the RZ cascade is skipped, CNOT gates should only be those required for RY cascade
        spy = mocker.spy(circuit.device, "execute")
        circuit(state_vector)
        tape = spy.call_args[0][0][0]

        assert tape.specs["resources"].gate_types["CNOT"] == n_CNOT

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        state = np.array([1 / 2, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 0])

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.MottonenStatePreparation(state, wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.MottonenStatePreparation(state, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    def test_batched_decomposition_fails(self):
        """Test that attempting to decompose a MottonenStatePreparation operation with
        broadcasting raises an error."""
        state = np.array([[1 / 2, 1 / 2, 1 / 2, 1 / 2], [0.0, 0.0, 0.0, 1.0]])

        op = qml.MottonenStatePreparation(state, wires=[0, 1])
        with pytest.raises(ValueError, match="Broadcasting with MottonenStatePreparation"):
            _ = op.decomposition()

        with pytest.raises(ValueError, match="Broadcasting with MottonenStatePreparation"):
            _ = qml.MottonenStatePreparation.compute_decomposition(state, qml.wires.Wires([0, 1]))

    def test_decomposition_includes_global_phase(self):
        """Test that the decomposition includes the correct global phase."""
        state = np.array([-0.5, 0.2, 0.3, 0.9, 0.5, 0.2, 0.3, 0.9])
        state = state / np.linalg.norm(state)
        decomp = qml.MottonenStatePreparation(state, [0, 1, 2]).decomposition()
        gphase = decomp[-1]
        assert isinstance(gphase, qml.GlobalPhase)
        assert qml.math.allclose(gphase.data[0], qml.math.mean(-1 * qml.math.angle(state)))


class TestInputs:
    """Test inputs and pre-processing."""

    # fmt: off
    @pytest.mark.parametrize("state_vector, wires", [
        ([1/2, 1/2], [0]),
        ([2/3, 0, 2j/3, -2/3], [0, 1]),
    ])
    # fmt: on
    def test_error_state_vector_not_normalized(self, state_vector, wires):
        """Tests that the correct error messages is raised if
        the given state vector is not normalized."""

        with pytest.raises(ValueError, match="State vectors have to be of norm"):
            qml.MottonenStatePreparation(state_vector, wires)

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires", [
        ([0, 1, 0], [0, 1]),
        ([0, 1, 0, 0, 0], [0]),
    ])
    # fmt: on
    def test_error_num_entries(self, state_vector, wires):
        """Tests that the correct error messages is raised  if
        the number of entries in the given state vector does not match
        with the number of wires in the system."""

        with pytest.raises(ValueError, match="State vectors must be of (length|shape)"):
            qml.MottonenStatePreparation(state_vector, wires)

    @pytest.mark.parametrize(
        "state_vector",
        [
            ([[[0, 0, 1, 0]]]),
            ([[[0, 1], [1, 0], [0, 0], [0, 0]]]),
        ],
    )
    def test_exception_wrong_shape(self, state_vector):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""

        with pytest.raises(ValueError, match="State vectors must be one-dimensional"):
            qml.MottonenStatePreparation(state_vector, 2)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.MottonenStatePreparation(np.array([0, 1]), wires=[0], id="a")
        assert template.id == "a"


class TestGradient:
    """Tests gradients."""

    # TODO: Currently the template fails for more elaborate gradient
    # tests, i.e. when the state contains zeros.
    # Make the template fully differentiable and test it.

    @pytest.mark.parametrize(
        "state_vector",
        [
            pnp.array([0.70710678, 0.70710678], requires_grad=True),
            pnp.array([0.70710678, 0.70710678j], requires_grad=True),
        ],
    )
    def test_gradient_evaluated(self, state_vector):
        """Test that the gradient is successfully calculated for a simple example. This test only
        checks that the gradient is calculated without an error."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(state_vector):
            qml.MottonenStatePreparation(state_vector, wires=range(1))
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(state_vector)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            [0.0, 0.7, 0.7, 0.0],
            [0.0, 0.5, 0.5, 0.0],
        ),
        ([0.1, 0.0, 0.0, 0.1], [0.5, 0.0, 0.0, 0.5]),
    ],
)
class TestCasting:
    """Test that the Mottonen state preparation ensures the compatibility with
    interfaces by using casting'"""

    @pytest.mark.jax
    def test_jax(self, inputs, expected):
        """Test that MottonenStatePreparation can be correctly used with the JAX interface."""
        from jax import numpy as jnp

        inputs = jnp.array(inputs)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(inputs):
            qml.MottonenStatePreparation(inputs, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        inputs = inputs / jnp.linalg.norm(inputs)
        res = circuit(inputs)
        assert np.allclose(res, expected, atol=1e-6, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, inputs, expected):
        """Test that MottonenStatePreparation can be correctly used with the JAX-JIT interface."""
        import jax
        from jax import numpy as jnp

        inputs = jnp.array(inputs)
        dev = qml.device("default.qubit", wires=2)

        @jax.jit
        @qml.qnode(dev)
        def circuit(inputs):
            qml.MottonenStatePreparation(inputs, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        inputs = inputs / jnp.linalg.norm(inputs)
        res = circuit(inputs)
        assert np.allclose(res, expected, atol=1e-6, rtol=0)

    @pytest.mark.tf
    def test_tensorflow(self, inputs, expected):
        """Test that MottonenStatePreparation can be correctly used with the TensorFlow interface."""
        import tensorflow as tf

        inputs = tf.Variable(inputs)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(inputs):
            qml.MottonenStatePreparation(inputs, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        inputs = inputs / tf.linalg.norm(inputs)
        res = circuit(inputs)
        assert np.allclose(res, expected, atol=1e-6, rtol=0)

    @pytest.mark.torch
    def test_torch(self, inputs, expected):
        """Test that MottonenStatePreparation can be correctly used with the Torch interface."""
        import torch

        inputs = torch.tensor(inputs, requires_grad=True)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(inputs):
            qml.MottonenStatePreparation(inputs, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        inputs = inputs / torch.linalg.norm(inputs)
        res = circuit(inputs)
        assert np.allclose(res.detach().numpy(), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("adj_base_op", [qml.StatePrep, qml.MottonenStatePreparation])
def test_adjoint_brings_back_to_zero(adj_base_op):
    """Test that a StatePrep then its adjoint returns the device to the zero state."""

    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit(state):
        qml.StatePrep(state, wires=[0, 1, 2])
        qml.adjoint(adj_base_op(state, wires=[0, 1, 2]))
        return qml.state()

    state = np.array([-0.5, 0.2, 0.3, 0.9, 0.5, 0.2, 0.3, 0.9])
    state = state / np.linalg.norm(state)
    actual = circuit(state)
    expected = np.zeros(8)
    expected[0] = 1.0
    assert qml.math.allclose(actual, expected)
