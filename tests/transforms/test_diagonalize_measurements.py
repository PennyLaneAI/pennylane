# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for the transform diagonalize_tape_measurements, which diagonalizes unsupported
observables in measurements on a tape.
"""

from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane import Hadamard, X, Y, Z
from pennylane.tape import QuantumScript
from pennylane.transforms.diagonalize_measurements import (
    _check_if_diagonalizing,
    _diagonalize_observable,
    diagonalize_tape_measurements,
    null_postprocessing,
)


class TestDiagonalizeObservable:
    """Tests for the _diagonalize_observable method"""

    @pytest.mark.parametrize("obs", [X(1), Y(2), Z(3), Hadamard(0)])
    def test_diagonalize_observable_single_observable(self, obs):
        """Test the diagonalize_observables function for X, Y, Z and H"""

        diagonalizing_gates, new_obs, _diag_obs = _diagonalize_observable(obs)

        assert diagonalizing_gates == obs.diagonalizing_gates()
        assert new_obs == Z(obs.wires)
        assert _diag_obs == ([obs], [obs.wires[0]])

    @pytest.mark.parametrize("obs, diagonalize", [(X(1), False), (Y(3), True), (Hadamard(0), True)])
    def test_supported_base_obs_arg(self, obs, diagonalize):
        """Test that observables are diagonalized or not depending on whether the observable
        type is included in supported_base_obs"""

        device_supported_obs = ["PauliX", "PauliZ"]

        diagonalizing_gates, new_obs, _visited_obs = _diagonalize_observable(
            obs, supported_base_obs=device_supported_obs
        )
        assert _visited_obs == ([obs], [obs.wires[0]])

        if diagonalize:
            assert diagonalizing_gates == obs.diagonalizing_gates()
            assert new_obs == Z(obs.wires)
        else:
            assert diagonalizing_gates == []
            assert new_obs == obs

    @pytest.mark.parametrize("obs, apply_gates", [(X(0), True), (Y(2), True), (Y(3), False)])
    def test_visited_obs_arg(self, obs, apply_gates):
        """Test that if _visited_obs includes previously encountered observables that overlap
        with the observable to be diagonalized, this is taken into account"""

        visited_obs = ([Y(3), Z(1)], [3, 1])

        diagonalizing_gates, new_obs, new_visited_obs = _diagonalize_observable(
            obs, _visited_obs=visited_obs
        )

        if apply_gates:
            assert diagonalizing_gates == obs.diagonalizing_gates()
            assert new_obs == Z(obs.wires)
        else:
            assert diagonalizing_gates == []
            assert new_obs == Z(obs.wires)

        if obs in visited_obs[0]:
            assert new_visited_obs == visited_obs
        else:
            assert new_visited_obs == ([Y(3), Z(1), obs], [3, 1, obs.wires[0]])

    @pytest.mark.parametrize(
        "compound_obs, expected_res, base_obs",
        [
            (X(0) @ Y(2), Z(0) @ Z(2), [X(0), Y(2)]),  # prod
            (
                qml.operation.Tensor(X(0), Y(2)),
                qml.operation.Tensor(Z(0), Z(2)),
                [X(0), Y(2)],
            ),  # tensor
            (X(1) + Y(2), Z(1) + Z(2), [X(1), Y(2)]),  # sum
            (2 * X(1), 2 * Z(1), [X(1)]),  # sprod
            (
                qml.ops.LinearCombination([2, 3], [X(0), Y(1)]),
                qml.ops.LinearCombination([2, 3], [Z(0), Z(1)]),
                [X(0), Y(1)],
            ),  # hamiltonian
            (
                qml.ops.LinearCombination([2, 3], [X(0) @ Y(2), 3 * Y(1)]),
                qml.ops.LinearCombination([2, 3], [Z(0) @ Z(2), 3 * Z(1)]),
                [X(0), Y(2), Y(1)],
            ),  # hamiltonian with composite terms
            (
                X(1) + 2 * Y(3) + X(0) @ (3 * Z(4)) + X(0) @ (Y(3) + Z(4)),
                Z(1) + 2 * Z(3) + Z(0) @ (3 * Z(4)) + Z(0) @ (Z(3) + Z(4)),
                [X(1), Y(3), X(0), Z(4)],
            ),  # nested messy sum
        ],
    )
    def test_compound_observables(self, compound_obs, expected_res, base_obs):
        """Test that _diagonalize_observable works on compound observables"""

        diagonalizing_gates, new_obs, visited_obs = _diagonalize_observable(compound_obs)

        expected_diag_gates = np.concatenate([o.diagonalizing_gates() for o in base_obs])

        assert new_obs == expected_res
        assert visited_obs == (base_obs, [o.wires[0] for o in base_obs])
        assert diagonalizing_gates == list(expected_diag_gates)

    def test_legacy_hamiltonian(self):
        """Test that _diagonalize_observable works on legacy Hamiltonians observables"""
        with pytest.warns():
            compound_obs = qml.ops.Hamiltonian([2, 3], [Y(0), X(1)])
            expected_res = qml.ops.Hamiltonian([2, 3], [Z(0), Z(1)])

        diagonalizing_gates, new_obs, visited_obs = _diagonalize_observable(compound_obs)
        base_obs = [Y(0), X(1)]
        expected_diag_gates = np.concatenate([o.diagonalizing_gates() for o in base_obs])

        assert new_obs == expected_res
        assert visited_obs == (base_obs, [o.wires[0] for o in base_obs])
        assert diagonalizing_gates == list(expected_diag_gates)

    @pytest.mark.parametrize(
        "compound_obs, expected_res, base_obs",
        [
            (X(0) @ Y(2), X(0) @ Z(2), [X(0), Y(2)]),  # prod
            (
                qml.operation.Tensor(X(0), Y(2)),
                qml.operation.Tensor(X(0), Z(2)),
                [X(0), Y(2)],
            ),  # tensor
            (X(1) + Y(2), X(1) + Z(2), [X(1), Y(2)]),  # sum
            (2 * X(1), 2 * X(1), [X(1)]),  # sprod
            (
                qml.ops.LinearCombination([2, 3], [X(0), Y(1)]),
                qml.ops.LinearCombination([2, 3], [X(0), Z(1)]),
                [X(0), Y(1)],
            ),  # hamiltonian
            (
                qml.ops.LinearCombination([2, 3], [X(0) @ Y(2), 3 * Y(1)]),
                qml.ops.LinearCombination([2, 3], [X(0) @ Z(2), 3 * Z(1)]),
                [X(0), Y(2), Y(1)],
            ),  # hamiltonian with composite terms
            (
                X(1) + 2 * Hadamard(3) + X(0) @ (3 * Z(4)) + X(0) @ (Hadamard(3) + Z(4)),
                X(1) + 2 * Z(3) + X(0) @ (3 * Z(4)) + X(0) @ (Z(3) + Z(4)),
                [X(1), Hadamard(3), X(0), Z(4)],
            ),  # nested messy sum
        ],
    )
    def test_compound_observables_supported_base_obs(self, compound_obs, expected_res, base_obs):
        """Test supported_base_obs argument works as expected for compound observables"""

        device_supported_obs = ["PauliX", "PauliZ"]
        diagonalizing_gates, new_obs, visited_obs = _diagonalize_observable(
            compound_obs, supported_base_obs=device_supported_obs
        )

        diag_obs = [o for o in base_obs if isinstance(o, (Y, Hadamard))]

        expected_diag_gates = (
            np.concatenate([o.diagonalizing_gates() for o in diag_obs]) if diag_obs else []
        )

        assert new_obs == expected_res
        assert visited_obs == (base_obs, [o.wires[0] for o in base_obs])
        assert diagonalizing_gates == list(expected_diag_gates)

    def test_compound_observable_with_duplicate_terms(self):
        """Test that when a compound observable includes duplicate terms, it only adds
        the diagonalizing gates once"""

        obs = X(0) + Z(2) + X(0) @ Z(3)

        diagonalizing_gates, new_obs, new_visited_obs = _diagonalize_observable(obs)

        assert new_visited_obs == ([X(0), Z(2), Z(3)], [0, 2, 3])
        assert diagonalizing_gates == X(0).diagonalizing_gates()
        assert new_obs == Z(0) + Z(2) + Z(0) @ Z(3)

    @pytest.mark.parametrize("obs", [X(0) + 2 * qml.Identity(), X(0) + 2 * qml.Identity(wires=2)])
    def test_with_identity(self, obs):
        """Test that observables with Identity are supported and Identity remains unchanged"""

        gates, new_obs, visited_obs = _diagonalize_observable(obs)

        assert gates == X(0).diagonalizing_gates()
        assert new_obs[0] == Z(0)  # X(0) is diagonalized
        assert new_obs[1] == obs[1]  # Identity is unchanged
        assert visited_obs == ([X(0)], [0])

    @pytest.mark.parametrize(
        "obs", [X(0) + 1.7 * X(2) + X(0) @ Y(2), X(0) + 2.3 * X(2) + X(0) @ Z(2)]
    )
    def test_non_commuting_measurements(self, obs):
        """Test that when a compound observable includes non-commuting observables, it raises
        an error"""

        obs = X(0) + Z(2) + X(0) @ Y(2)

        with pytest.raises(ValueError, match="Expected only a single observable per wire"):
            _ = _diagonalize_observable(obs)

    @pytest.mark.parametrize(
        "obs", [X(0) + 1.7 * X(2) + X(0) @ Y(2), X(0) + 2.3 * X(2) + X(0) @ Z(2)]
    )
    def test_non_commuting_measurements_with_supported_obs(self, obs):
        """Test that when a compound observable includes non-commuting observables, it raises
        an error, even if some of those observables aren't being diagonalized"""

        device_supported_obs = ["PauliX", "PauliZ"]

        with pytest.raises(ValueError, match="Expected only a single observable per wire"):
            _ = _diagonalize_observable(obs, supported_base_obs=device_supported_obs)

    def test_diagonalizing_unknown_observable_raises_error(self):
        """Test that an unknown observable raises an error when diagonalizing"""

        # pylint: disable=too-few-public-methods
        class MyObs(qml.operation.Observable):

            @property
            def name(self):
                return f"MyObservable[{self.wires}]"

        with pytest.raises(NotImplementedError, match="Unable to convert observable"):
            _ = _diagonalize_observable(MyObs(wires=[2]))

    @pytest.mark.parametrize(
        "obs, input_visited_obs, switch_basis, expected_res",
        [
            (X(0), ([], []), True, (True, ([X(0)], [0]))),
            (X(0), ([], []), False, (False, ([X(0)], [0]))),
            (X(0), ([X(0)], [0]), False, (False, ([X(0)], [0]))),
            (X(0), ([X(0)], [0]), True, (False, ([X(0)], [0]))),
        ],
    )
    def test_check_if_diagonalising(self, obs, input_visited_obs, switch_basis, expected_res):
        """Test that _check_if_diagonalizing returns True or False based on whether the
        observable is supported, and whether its already been diagonalized previously"""
        diagonalize, output_visited_obs = _check_if_diagonalizing(
            obs, input_visited_obs, switch_basis
        )

        assert (diagonalize, output_visited_obs) == expected_res

    @pytest.mark.parametrize(
        "obs, _visited_obs, raise_error",
        [(Y(1), ([X(0)], [0]), False), (Y(1), ([Y(1)], [1]), False), (Y(1), ([X(1)], [1]), True)],
    )
    def test_check_if_diagonalizing_raises_error(self, obs, _visited_obs, raise_error):
        """Test that _check_if_diagonalizing raises an error if the observable should be
        diagonalized, but a different observable on that wire has already been diagonalized"""
        if raise_error:
            with pytest.raises(ValueError, match="overlaps with another observable on the tape"):
                _ = _check_if_diagonalizing(obs, _visited_obs, switch_basis=True)
            with pytest.raises(ValueError, match="overlaps with another observable on the tape"):
                _ = _check_if_diagonalizing(obs, _visited_obs, switch_basis=False)

        else:
            _ = _check_if_diagonalizing(obs, _visited_obs, switch_basis=True)
            _ = _check_if_diagonalizing(obs, _visited_obs, switch_basis=False)


class TestDiagonalizeTapeMeasurements:
    """Tests the diagonalize_tape_measurements transform"""

    def test_diagonalize_tape_measurements(self):
        """Test that the diagonalize_tape_measurements transform diagonalizes the measurements on the tape"""
        measurements = [qml.expval(X(0)), qml.var(X(1) + Y(2))]

        tape = QuantumScript([], measurements=measurements)
        tapes, fn = diagonalize_tape_measurements(tape)
        new_tape = tapes[0]

        assert new_tape.measurements == [qml.expval(Z(0)), qml.var(Z(1) + Z(2))]
        assert (
            new_tape.operations
            == X(0).diagonalizing_gates() + X(1).diagonalizing_gates() + Y(2).diagonalizing_gates()
        )

        assert fn == null_postprocessing

    def test_with_duplicate_measurements(self):
        """Test that the diagonalize_tape_measurements transform diagonalizes the measurements
        on the tape correctly when the same observable is used more than once"""
        measurements = [qml.expval(X(0)), qml.var(X(1) + Y(2)), qml.sample(X(0) @ Y(2))]

        tape = QuantumScript([], measurements=measurements)
        tapes, fn = diagonalize_tape_measurements(tape)
        new_tape = tapes[0]

        assert new_tape.measurements == [
            qml.expval(Z(0)),
            qml.var(Z(1) + Z(2)),
            qml.sample(Z(0) @ Z(2)),
        ]
        assert (
            new_tape.operations
            == X(0).diagonalizing_gates() + X(1).diagonalizing_gates() + Y(2).diagonalizing_gates()
        )

        assert fn == null_postprocessing

    def test_non_commuting_observables_raise_an_error(self):
        """Test that the diagonalize_tape_measurements raises an error as expected if the tape contains
        non-commuting observables"""
        measurements = [qml.expval(X(0)), qml.var(Z(0) + Y(2))]

        tape = QuantumScript([], measurements=measurements)

        with pytest.raises(ValueError, match="overlaps with another observable on the tape"):
            _ = diagonalize_tape_measurements(tape)

    def test_measurements_with_no_obs(self):
        """Test that the transform correctly handles tapes where some measurements don't
        have an observable"""

        measurements = [qml.expval(X(0)), qml.var(X(1) + Y(2)), qml.sample()]

        tape = QuantumScript([], measurements=measurements)
        tapes, fn = diagonalize_tape_measurements(tape)
        new_tape = tapes[0]

        assert new_tape.measurements == [qml.expval(Z(0)), qml.var(Z(1) + Z(2)), qml.sample()]
        assert (
            new_tape.operations
            == X(0).diagonalizing_gates() + X(1).diagonalizing_gates() + Y(2).diagonalizing_gates()
        )

        assert fn == null_postprocessing

    def test_decomposing_subset_of_obs(self):
        """Test that passing a list of supported obs to the diagonalize_tape_measurements transform
        diagonalizes only the unsupported base observables"""
        measurements = [
            qml.expval(X(0)),
            qml.var(X(1) + Y(2) @ X(1)),
            qml.counts(X(0) @ (2.3 * Y(2))),
        ]

        tape = QuantumScript([], measurements=measurements)

        tapes, fn = diagonalize_tape_measurements(tape, supported_base_obs=["PauliX", "PauliZ"])

        new_tape = tapes[0]

        assert new_tape.measurements == [
            qml.expval(X(0)),
            qml.var(X(1) + Z(2) @ X(1)),
            qml.counts(X(0) @ (2.3 * Z(2))),
        ]
        assert new_tape.operations == Y(2).diagonalizing_gates()

        assert fn == null_postprocessing

    @pytest.mark.parametrize("supported_base_obs", (["PauliC", "PauliZ"], [X, Z], [X(0), qml.Z(1)]))
    def test_bad_obs_input_raises_error(self, supported_base_obs):
        """Test that if a value is passed to supported_base_obs that can't be interpreted, a clear error is raised"""

        with pytest.raises(ValueError, match="Supported base observables must be a subset of"):
            _ = diagonalize_tape_measurements(
                QuantumScript([], measurements=[]), supported_base_obs=supported_base_obs
            )

    @pytest.mark.parametrize("supported_base_obs", (["PauliZ"],))
    @pytest.mark.parametrize("shots", [None, 2000, (4000, 5000, 6000)])
    def test_qnode_integration(self, supported_base_obs, shots):

        dev = qml.device("default.qubit", shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.23, 1)
            qml.RY(2.46, 0)
            return qml.expval(X(0)), qml.var(X(1) + Y(2))

        @partial(diagonalize_tape_measurements, supported_base_obs=supported_base_obs)
        @qml.qnode(dev)
        def circuit_diagonalized():
            qml.RX(1.23, 1)
            qml.RY(2.46, 0)
            return qml.expval(X(0)), qml.var(X(1) + Y(2))

        expected_res = circuit()
        res = circuit_diagonalized()

        if len(dev.shots.shot_vector) > 1:
            for r_diagonalized, r in zip(res, expected_res):
                assert np.allclose(r_diagonalized, r, atol=0.1)
        else:
            assert np.allclose(expected_res, res, atol=0.1)
