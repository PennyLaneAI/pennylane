# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the AmplitudeAmplification template.
"""
# pylint: disable=private-access, protected-access
import copy
from functools import reduce

import pytest

import pennylane as qml
import numpy as np

from pennylane import numpy as qnp
from pennylane.math import allclose, get_interface

# auxiliar operators


@qml.prod
def generator(wires):
    for wire in wires:
        qml.Hadamard(wire)


@qml.prod
def oracle(items, wires):
    for item in items:
        qml.FlipSign(item, wires=wires)


class TestInitialization:
    """Test the AmplitudeAmplification class initializes correctly."""

    @pytest.mark.parametrize(
        "fixed_point, aux_wire, raise_error",
        (
            (True, 3, False),
            (True, "a", False),
            (False, 4, False),
            (True, None, True),
        ),
    )
    def test_error_none_wire(self, fixed_point, aux_wire, raise_error):
        """Test an error is raised if aux_wire is None and fixed_point is True."""

        U = generator(wires=range(3))
        O = oracle([0, 2], wires=range(3))

        if raise_error:
            with pytest.raises(
                TypeError, match="aux_wire must be specified if fixed_point == True."
            ):
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, aux_wire=aux_wire
                )

        else:
            try:
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, aux_wire=aux_wire
                )
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize(
        "U, O, raise_error",
        (
            (generator(wires=[0, 1, 2]), oracle([0, 1], wires=[0, 1, 2]), False),
            (generator(wires=[0, 2, 1]), oracle([0, 1], wires=[0, 1, 2]), False),
            (generator(wires=[0, 1, 3]), oracle([0, 1], wires=[0, 1, 2]), True),
            (generator(wires=[0, 1, 2, 3]), oracle([0, 1], wires=[0, 1, 2]), True),
        ),
    )
    def test_error_wrong_wires(self, U, O, raise_error):
        """Test an error is raised if the wires of U and O are not the same."""

        if raise_error:
            with pytest.raises(TypeError, match="U and O must act on the same wires."):
                qml.AmplitudeAmplification(U, O)

        else:
            try:
                qml.AmplitudeAmplification(U, O)
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize(
        "wires, fixed_point, aux_wire, raise_error",
        (
            ([0, 1, 2], True, 2, True),
            ([0, 1, 2], True, "a", False),
            (["a", "b"], True, "a", True),
            ([0, 1], False, 0, False),
        ),
    )
    def test_error_wrong_wire(self, wires, fixed_point, aux_wire, raise_error):
        """Test an error is raised if aux_wire is part of the U wires."""

        U = generator(wires=wires)
        O = oracle([0], wires=wires)

        if raise_error:
            with pytest.raises(TypeError, match="aux_wire must be different from the wires of U."):
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, aux_wire=aux_wire
                )

        else:
            try:
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, aux_wire=aux_wire
                )
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize(
        "U, O, iters, fixed_point, aux_wire",
        (
            (generator(wires=range(3)), oracle([0, 1], wires=range(3)), True, 3),
            (generator(wires=range(2)), oracle([0, 1], wires=range(2)), False, 2),
            (generator(wires=range(4)), oracle([0, 1], wires=range(4)), True, 5),
        ),
    )
    def test_init_correctly(self, U, O, iters, fixed_point, aux_wire):
        """Test that all of the attributes are initalized correctly."""

        op = qml.AmplitudeAmplification(U, O, iters, fixed_point, aux_wire)

        assert op.wires == U.wires + qml.wires.Wires(aux_wire)
        assert op.U == U
        assert op.O == O
        assert op.iters == iters
        assert op.fixed_point == fixed_point
        assert op.aux_wire == aux_wire


class TestResults:
    @pytest.mark.parametrize(
        "n_wires, items, iters",
        (
            (3, [0, 2], 1),
            (3, [1, 2], 2),
            (5, [4, 5, 7, 12], 3),
            (5, [0, 1, 2, 3, 4], 4),
        ),
    )
    def test_compare_grover(self, n_wires, items, iters):
        U = generator(wires=range(n_wires))
        O = oracle(items, wires=range(n_wires))

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit_amplitude_amplification():
            generator(wires=range(n_wires))
            qml.AmplitudeAmplification(U, O, iters)
            return qml.probs(wires=range(n_wires))

        @qml.qnode(dev)
        def circuit_grover():
            generator(wires=range(n_wires))

            for _ in range(iters):
                oracle(items, wires=range(n_wires))
                qml.GroverOperator(wires=range(n_wires))

            return qml.probs(wires=range(n_wires))

        assert np.allclose(circuit_amplitude_amplification(), circuit_grover(), atol=1e-5)
