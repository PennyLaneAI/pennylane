# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the gradients.parameter_shift module using the new return types."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.gradients.parameter_shift import _get_operation_recipe, _put_zeros_in_pdA2_involutory
from pennylane.devices import DefaultQubit
from pennylane.operation import Observable, AnyWires


shot_vec_tol = 10e-3


class TestParamShiftShotVector:
    """Unit tests for the param_shift function used with a device that has a
    shot vector defined"""

    def test_multi_measure_probs_expval(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device("default.qubit", wires=2, shots=(1000000, 10000000))
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tapes, fn = qml.gradients.param_shift(tape)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert len(res) == 2

        for r in res:
            assert len(r) == 2

        expval_expected = [-2 * np.sin(x) / 2, 0]
        probs_expected = (
            np.array(
                [
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )

        for r in res:

            # Expvals
            r_to_check = r[0][0]
            exp = expval_expected[0]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            r_to_check = r[0][1]
            exp = expval_expected[1]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == ()

            # Probs

            r_to_check = r[1][0]
            exp = probs_expected[:, 0]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

            r_to_check = r[1][1]
            exp = probs_expected[:, 1]
            assert np.allclose(r_to_check, exp, atol=shot_vec_tol)
            assert isinstance(r_to_check, np.ndarray)
            assert r_to_check.shape == (4,)

    def test_involutory_variance(self, tol):
        """Tests qubit observables that are involutory"""
        dev = qml.device("default.qubit", wires=1, shots=(1000000, 10000000))
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.PauliZ(0))

        res = dev.execute(tape)
        expected = 1 - np.cos(a) ** 2
        for r in res:
            assert np.allclose(r, expected, atol=shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        for _gA in gradA:
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()

        assert len(tapes) == 1 + 2 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = 2 * np.sin(a) * np.cos(a)

        # TODO: finite diff shot-vector update
        # assert gradF == pytest.approx(expected, abs=tol)
        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=shot_vec_tol)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        dev = qml.device("default.qubit", wires=1, shots=(int(10e5), int(10e5)))
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        herm_shot_vec_tol = shot_vec_tol * 100
        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.var(qml.Hermitian(A, 0))

        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        for r in res:
            assert np.allclose(r, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4 * 1

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 2

        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        for _gA in gradA:
            assert _gA == pytest.approx(expected, abs=herm_shot_vec_tol)
            assert isinstance(_gA, np.ndarray)
            assert _gA.shape == ()
            # TODO: finite diff shot-vector update
            # assert gradF == pytest.approx(expected, abs=tol)

    def test_involutory_and_noninvolutory_variance_single_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable when there's a single trainable parameter."""
        dev = qml.device("default.qubit", wires=2, shots=(1000000, 10000000))
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        herm_shot_vec_tol = shot_vec_tol * 100
        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        # Note: only the first param is trainable
        tape.trainable_params = {0}

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for r in res:
            assert np.allclose(r, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 1

        expected = [2 * np.sin(a) * np.cos(a), 0]

        # Param-shift
        for shot_vec_result in gradA:
            for param_res in shot_vec_result:
                assert isinstance(param_res, np.ndarray)
                assert param_res.shape == ()

            assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
            assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

        # TODO: finite diff shot-vector update
        # for shot_vec_result in gradF:
        #     for param_res in shot_vec_result:
        #         assert isinstance(param_res, np.ndarray)
        #         assert param_res.shape == ()

        #     assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
        #     assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

    def test_involutory_and_noninvolutory_variance_multi_param(self, tol):
        """Tests a qubit Hermitian observable that is not involutory alongside
        an involutory observable."""
        dev = qml.device("default.qubit", wires=2, shots=(1000000, 10000000))
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        a = 0.54

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RX(a, wires=1)
            qml.var(qml.PauliZ(0))
            qml.var(qml.Hermitian(A, 1))

        tape.trainable_params = {0, 1}
        herm_shot_vec_tol = shot_vec_tol * 100

        res = dev.execute(tape)
        expected = [1 - np.cos(a) ** 2, (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)]
        for res_shot_item in res:
            assert np.allclose(res_shot_item, expected, atol=herm_shot_vec_tol, rtol=0)

        # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        assert isinstance(gradA, tuple)
        assert len(gradA) == 2
        for shot_vec_res in gradA:
            for meas_res in shot_vec_res:
                for param_res in meas_res:
                    assert isinstance(param_res, np.ndarray)
                    assert param_res.shape == ()

        assert len(tapes) == 1 + 2 * 4

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))
        assert len(tapes) == 1 + 2

        expected = [2 * np.sin(a) * np.cos(a), 0, 0, -35 * np.sin(2 * a) - 12 * np.cos(2 * a)]

        # Param-shift
        for shot_vec_result in gradA:
            assert isinstance(shot_vec_result[0][0], np.ndarray)
            assert shot_vec_result[0][0].shape == ()
            assert shot_vec_result[0][0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[0][1], np.ndarray)
            assert shot_vec_result[0][1].shape == ()
            assert shot_vec_result[0][1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][0], np.ndarray)
            assert shot_vec_result[1][0].shape == ()
            assert shot_vec_result[1][0] == pytest.approx(expected[2], abs=herm_shot_vec_tol)

            assert isinstance(shot_vec_result[1][1], np.ndarray)
            assert shot_vec_result[1][1].shape == ()
            assert shot_vec_result[1][1] == pytest.approx(expected[3], abs=herm_shot_vec_tol)

        # TODO: finite diff shot-vector update
        # for shot_vec_result in gradF:
        #     for param_res in shot_vec_result:
        #         assert isinstance(param_res, np.ndarray)
        #         assert param_res.shape == ()

        #     assert shot_vec_result[0] == pytest.approx(expected[0], abs=herm_shot_vec_tol)
        #     assert shot_vec_result[1] == pytest.approx(expected[1], abs=herm_shot_vec_tol)

    # TODO: finite diff shot-vector update
    @pytest.mark.xfail(reason="Uses finite diff")
    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3, shots=(1000000, 10000000))

        a = 0.54
        b = -0.423
        c = 0.123

        with qml.tape.QuantumTape() as tape:
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))
            qml.var(qml.PauliZ(2))

        res = dev.execute(tape)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )

        assert isinstance(res, tuple)
        assert np.allclose(res, expected, atol=shot_vec_tol, rtol=0)

        # # circuit jacobians
        tapes, fn = qml.gradients.param_shift(tape)
        gradA = fn(dev.batch_execute(tapes))

        tapes, fn = qml.gradients.finite_diff(tape)
        gradF = fn(dev.batch_execute(tapes))

        expected = np.array(
            [
                [2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0],
                [
                    0,
                    -np.cos(a) * np.sin(b),
                    0.5 * (2 * np.cos(b) * np.cos(2 * c) * np.sin(b) + np.sin(2 * b)),
                ],
                [0, 0, np.cos(b) ** 2 * np.sin(2 * c)],
            ]
        ).T
        assert isinstance(gradA, tuple)
        for a, e in zip(gradA, expected):
            for a_comp, e_comp in zip(a, e):
                assert isinstance(a_comp, np.ndarray)
                assert a_comp.shape == ()
                assert np.allclose(a_comp, e_comp, atol=shot_vec_tol, rtol=0)
        assert gradF == pytest.approx(expected, abs=shot_vec_tol)
