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
Tests for the TrotterProduct template and helper functions.
"""
# pylint: disable=private-access, protected-access, too-many-arguments
import copy
from collections import defaultdict
from functools import partial, reduce

import pytest

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.math import allclose, get_interface
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.resource import Resources
from pennylane.resource.error import SpectralNormError
from pennylane.templates.subroutines.time_evolution.trotter import (
    TrotterizedQfunc,
    _recursive_expression,
    _recursive_qfunc,
    _scalar,
)

test_hamiltonians = (
    qml.dot([1.0, 1.0, 1.0], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]),
    qml.dot(
        [1.23, -0.45], [qml.s_prod(0.1, qml.PauliX(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1))]
    ),  # op arith
    qml.dot(
        [1, -0.5, 0.5], [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(0)]
    ),  # H = Identity
    qml.dot([2.0, 2.0, 2.0], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]),
)

p_4 = (4 - 4 ** (1 / 3)) ** -1
p_6 = (4 - 4 ** (1 / 5)) ** -1
p4_comp = 1 - (4 * p_4)
p6_comp = 1 - (4 * p_6)

test_decompositions = (
    {  # (hamiltonian_index, order): decomposition assuming t = 4.2, computed by hand
        (0, 1): [
            qml.evolve(qml.PauliX(0), -4.2),
            qml.evolve(qml.PauliY(0), -4.2),
            qml.evolve(qml.PauliZ(1), -4.2),
        ],
        (0, 2): [
            qml.evolve(qml.PauliX(0), -4.2 / 2),
            qml.evolve(qml.PauliY(0), -4.2 / 2),
            qml.evolve(qml.PauliZ(1), -4.2 / 2),
            qml.evolve(qml.PauliZ(1), -4.2 / 2),
            qml.evolve(qml.PauliY(0), -4.2 / 2),
            qml.evolve(qml.PauliX(0), -4.2 / 2),
        ],
        (0, 4): [
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),  # S_2(p * t) ^ 2
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -4.2 / 2),  # S_2((1 - 4p) * t)
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -4.2 / 2),  # S_2(p * t) ^ 2
        ],
        (1, 1): [
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * -4.2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * -4.2),
        ],
        (1, 2): [
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * -4.2 / 2),
        ],
        (1, 4): [
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), (1 - 4 * p_4) * 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), (1 - 4 * p_4) * -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), (1 - 4 * p_4) * -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), (1 - 4 * p_4) * 1.23 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * -4.2 / 2),
            qml.evolve(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * -4.2 / 2),
        ],
        (2, 1): [
            qml.evolve(qml.Identity(wires=[0, 1]), -4.2),
            qml.evolve(qml.PauliZ(0), -0.5 * -4.2),
            qml.evolve(qml.PauliZ(0), 0.5 * -4.2),
        ],
        (2, 2): [
            qml.evolve(qml.Identity(wires=[0, 1]), -4.2 / 2),
            qml.evolve(qml.PauliZ(0), -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), -4.2 / 2),
        ],
        (2, 4): [
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), (1 - 4 * p_4) * -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), (1 - 4 * p_4) * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), (1 - 4 * p_4) * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), (1 - 4 * p_4) * -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), (1 - 4 * p_4) * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * 0.5 * -4.2 / 2),
            qml.evolve(qml.PauliZ(0), p_4 * -0.5 * -4.2 / 2),
            qml.evolve(qml.Identity(wires=[0, 1]), p_4 * -4.2 / 2),
        ],
        (3, 1): [
            qml.evolve(qml.PauliX(0), -8.4),
            qml.evolve(qml.PauliY(0), -8.4),
            qml.evolve(qml.PauliZ(1), -8.4),
        ],
        (3, 2): [
            qml.evolve(qml.PauliX(0), -8.4 / 2),
            qml.evolve(qml.PauliY(0), -8.4 / 2),
            qml.evolve(qml.PauliZ(1), -8.4 / 2),
            qml.evolve(qml.PauliZ(1), -8.4 / 2),
            qml.evolve(qml.PauliY(0), -8.4 / 2),
            qml.evolve(qml.PauliX(0), -8.4 / 2),
        ],
        (3, 4): [
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),  # S_2(p * t) ^ 2
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -8.4 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -8.4 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -8.4 / 2),
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -8.4 / 2),  # S_2((1 - 4p) * t)
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -8.4 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -8.4 / 2),  # S_2(p * t) ^ 2
        ],
    }
)

test_resources_data = {
    # (hamiltonian_index, order): Resources computed by hand
    (0, 1): Resources(
        num_wires=2,
        num_gates=3,
        gate_types=defaultdict(int, {"Evolution": 3}),
        gate_sizes=defaultdict(int, {1: 3}),
        depth=2,
    ),
    (0, 2): Resources(
        num_wires=2,
        num_gates=6,
        gate_types=defaultdict(int, {"Evolution": 6}),
        gate_sizes=defaultdict(int, {1: 6}),
        depth=4,
    ),
    (0, 4): Resources(
        num_wires=2,
        num_gates=30,
        gate_types=defaultdict(int, {"Evolution": 30}),
        gate_sizes=defaultdict(int, {1: 30}),
        depth=20,
    ),
    (1, 1): Resources(
        num_wires=2,
        num_gates=2,
        gate_types=defaultdict(int, {"Evolution": 2}),
        gate_sizes=defaultdict(int, {1: 1, 2: 1}),
        depth=2,
    ),
    (1, 2): Resources(
        num_wires=2,
        num_gates=4,
        gate_types=defaultdict(int, {"Evolution": 4}),
        gate_sizes=defaultdict(int, {1: 2, 2: 2}),
        depth=4,
    ),
    (1, 4): Resources(
        num_wires=2,
        num_gates=20,
        gate_types=defaultdict(int, {"Evolution": 20}),
        gate_sizes=defaultdict(int, {1: 10, 2: 10}),
        depth=20,
    ),
    (2, 1): Resources(
        num_wires=2,
        num_gates=3,
        gate_types=defaultdict(int, {"Evolution": 3}),
        gate_sizes=defaultdict(int, {1: 2, 2: 1}),
        depth=3,
    ),
    (2, 2): Resources(
        num_wires=2,
        num_gates=6,
        gate_types=defaultdict(int, {"Evolution": 6}),
        gate_sizes=defaultdict(int, {1: 4, 2: 2}),
        depth=6,
    ),
    (2, 4): Resources(
        num_wires=2,
        num_gates=30,
        gate_types=defaultdict(int, {"Evolution": 30}),
        gate_sizes=defaultdict(int, {1: 20, 2: 10}),
        depth=30,
    ),
    (3, 1): Resources(
        num_wires=2,
        num_gates=3,
        gate_types=defaultdict(int, {"Evolution": 3}),
        gate_sizes=defaultdict(int, {1: 3}),
        depth=2,
    ),
    (3, 2): Resources(
        num_wires=2,
        num_gates=6,
        gate_types=defaultdict(int, {"Evolution": 6}),
        gate_sizes=defaultdict(int, {1: 6}),
        depth=4,
    ),
    (3, 4): Resources(
        num_wires=2,
        num_gates=30,
        gate_types=defaultdict(int, {"Evolution": 30}),
        gate_sizes=defaultdict(int, {1: 30}),
        depth=20,
    ),
}


def _generate_simple_decomp(coeffs, ops, time, order, n):
    """Given coeffs, ops and a time argument in a given framework, generate the
    Trotter product for order and number of trotter steps."""
    decomp = []
    if order == 1:
        decomp.extend(qml.evolve(op, -coeff * (time / n)) for coeff, op in zip(coeffs, ops))

    coeffs_ops = zip(coeffs, ops)

    if get_interface(coeffs) == "torch":
        import torch

        coeffs_ops_reversed = zip(torch.flip(coeffs, dims=(0,)), ops[::-1])
    else:
        coeffs_ops_reversed = zip(coeffs[::-1], ops[::-1])

    if order == 2:
        decomp.extend(qml.evolve(op, -coeff * (time / n) / 2) for coeff, op in coeffs_ops)
        decomp.extend(qml.evolve(op, -coeff * (time / n) / 2) for coeff, op in coeffs_ops_reversed)

    if order == 4:
        s_2 = []
        s_2_p = []

        for coeff, op in coeffs_ops:
            s_2.append(qml.evolve(op, -(p_4 * coeff) * (time / n) / 2))
            s_2_p.append(qml.evolve(op, -((1 - (4 * p_4)) * coeff) * (time / n) / 2))

        for coeff, op in coeffs_ops_reversed:
            s_2.append(qml.evolve(op, -(p_4 * coeff) * (time / n) / 2))
            s_2_p.append(qml.evolve(op, -((1 - (4 * p_4)) * coeff) * (time / n) / 2))

        decomp = (s_2 * 2) + s_2_p + (s_2 * 2)

    return decomp * n


class TestInitialization:
    """Test the TrotterProduct class initializes correctly."""

    @pytest.mark.parametrize(
        "hamiltonian, raise_error",
        (
            (qml.PauliX(0), True),
            (qml.prod(qml.PauliX(0), qml.PauliZ(1)), True),
            (qml.Hamiltonian([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)]), False),
            (qml.dot([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)]), False),
        ),
    )
    def test_error_type(self, hamiltonian, raise_error):
        """Test an error is raised of an incorrect type is passed"""
        if raise_error:
            with pytest.raises(
                TypeError,
                match="The given operator must be a PennyLane ~.Sum or ~.SProd",
            ):
                qml.TrotterProduct(hamiltonian, time=1.23)

        else:
            qml.TrotterProduct(hamiltonian, time=1.23)

    @pytest.mark.parametrize(
        "hamiltonian",
        (
            qml.Hamiltonian([1.23, 4 + 5j], [qml.PauliX(0), qml.PauliZ(1)]),
            qml.dot([1.23, 4 + 5j], [qml.PauliX(0), qml.PauliZ(1)]),
            qml.dot([1.23, 0.5], [qml.RY(1.23, 0), qml.RZ(3.45, 1)]),
        ),
    )
    def test_error_hermiticity(self, hamiltonian):
        """Test that an error is raised if any terms in
        the Hamiltonian are not Hermitian and check_hermitian is True."""

        with pytest.raises(
            ValueError, match="One or more of the terms in the Hamiltonian may not be Hermitian"
        ):
            qml.TrotterProduct(hamiltonian, time=0.5)

        try:
            qml.TrotterProduct(hamiltonian, time=0.5, check_hermitian=False)
        except ValueError:
            assert False  # No error should be raised if the check_hermitian flag is disabled

    @pytest.mark.parametrize(
        "hamiltonian",
        (
            qml.Hamiltonian([1.0], [qml.PauliX(0)]),
            qml.dot([2.0], [qml.PauliY(0)]),
        ),
    )
    def test_error_hamiltonian(self, hamiltonian):
        """Test that an error is raised if the input Hamiltonian has only 1 term."""
        with pytest.raises(
            ValueError, match="There should be at least 2 terms in the Hamiltonian."
        ):
            qml.TrotterProduct(hamiltonian, 1.23, n=2, order=4)

    @pytest.mark.parametrize("order", (-1, 0, 0.5, 3, 7.0))
    def test_error_order(self, order):
        """Test that an error is raised if 'order' is not one or positive even number."""
        time = 0.5
        hamiltonian = qml.dot([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)])

        with pytest.raises(
            ValueError, match="The order of a TrotterProduct must be 1 or a positive even integer,"
        ):
            qml.TrotterProduct(hamiltonian, time, order=order)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_init_correctly(self, hamiltonian):
        """Test that all of the attributes are initialized correctly."""
        time, n, order = (4.2, 10, 4)
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order, check_hermitian=False)

        if isinstance(hamiltonian, qml.ops.op_math.SProd):
            hamiltonian = hamiltonian.simplify()

        assert op.wires == hamiltonian.wires
        assert op.parameters == [*hamiltonian.data, time]
        assert op.data == (*hamiltonian.data, time)
        assert op.hyperparameters == {
            "base": hamiltonian,
            "n": n,
            "order": order,
            "check_hermitian": False,
        }

    @pytest.mark.parametrize("n", (1, 2, 5, 10))
    @pytest.mark.parametrize("time", (0.5, 1.2))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_copy(self, hamiltonian, time, n, order):
        """Test that we can make deep and shallow copies of TrotterProduct correctly."""
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order)
        new_op = copy.copy(op)

        assert op.wires == new_op.wires
        assert op.parameters == new_op.parameters
        assert op.data == new_op.data
        assert op.hyperparameters == new_op.hyperparameters
        assert op is not new_op

    @pytest.mark.jax
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_standard_validity(self, hamiltonian):
        """Test standard validity criteria using assert_valid."""
        time, n, order = (4.2, 10, 4)
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order)
        qml.ops.functions.assert_valid(op, skip_differentiation=True)

    @pytest.mark.jax
    @pytest.mark.xfail(reason="https://github.com/PennyLaneAI/pennylane/issues/6333", strict=False)
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_standard_validity_with_differentiation(self, hamiltonian):
        """Test standard validity criteria using assert_valid."""
        time, n, order = (4.2, 10, 4)
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order)
        qml.ops.functions.assert_valid(op)

    # TODO: Remove test when we deprecate ApproxTimeEvolution
    @pytest.mark.parametrize("n", (1, 2, 5, 10))
    @pytest.mark.parametrize("time", (0.5, 1.2))
    def test_convention_approx_time_evolv(self, time, n):
        """Test that TrotterProduct matches ApproxTimeEvolution"""
        hamiltonian = qml.Hamiltonian(
            [1.23, -0.45, 6], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        )
        op1 = qml.TrotterProduct(hamiltonian, time, order=1, n=n)
        op2 = qml.adjoint(qml.ApproxTimeEvolution(hamiltonian, time, n=n))

        assert qnp.allclose(
            qml.matrix(op1, wire_order=hamiltonian.wires),
            qml.matrix(op2, wire_order=hamiltonian.wires),
        )

        op1 = qml.adjoint(qml.TrotterProduct(hamiltonian, time, order=1, n=n))
        op2 = qml.ApproxTimeEvolution(hamiltonian, time, n=n)

        assert qnp.allclose(
            qml.matrix(op1, wire_order=hamiltonian.wires),
            qml.matrix(op2, wire_order=hamiltonian.wires),
        )

    @pytest.mark.parametrize(
        "make_H",
        [
            lambda: qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(1)]),
            lambda: qml.sum(qml.PauliX(0), qml.PauliY(1)),
            lambda: qml.s_prod(1.2, qml.PauliX(0) + qml.PauliY(1)),
        ],
    )
    def test_queuing(self, make_H):
        """Test that the target operator is removed from the queue."""

        with qml.queuing.AnnotatedQueue() as q:
            H = make_H()
            op = qml.TrotterProduct(H, time=2)

        assert len(q.queue) == 1
        assert q.queue[0] is op


class TestPrivateFunctions:
    """Test the private helper functions."""

    @pytest.mark.parametrize(
        "order, result",
        (
            (4, 0.4144907717943757),
            (6, 0.3730658277332728),
            (8, 0.35958464934999224),
        ),
    )  # Computed by hand
    def test_private_scalar(self, order, result):
        """Test the _scalar function correctly computes the parameter scalar."""
        s = _scalar(order)
        assert qnp.isclose(s, result)

    expected_expansions = (  # for H = X0 + Y0 + Z1, t = 1.23, computed by hand
        [  # S_1(t)
            qml.evolve(qml.PauliX(0), -1.23),
            qml.evolve(qml.PauliY(0), -1.23),
            qml.evolve(qml.PauliZ(1), -1.23),
        ],
        [  # S_2(t)
            qml.evolve(qml.PauliX(0), -1.23 / 2),
            qml.evolve(qml.PauliY(0), -1.23 / 2),
            qml.evolve(qml.PauliZ(1), -1.23 / 2),
            qml.evolve(qml.PauliZ(1), -1.23 / 2),
            qml.evolve(qml.PauliY(0), -1.23 / 2),
            qml.evolve(qml.PauliX(0), -1.23 / 2),
        ],
        [  # S_4(t)
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),  # S_2(p * t) ^ 2
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -1.23 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), (1 - 4 * p_4) * -1.23 / 2),
            qml.evolve(qml.PauliY(0), (1 - 4 * p_4) * -1.23 / 2),
            qml.evolve(qml.PauliX(0), (1 - 4 * p_4) * -1.23 / 2),  # S_2((1 - 4p) * t)
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliZ(1), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliY(0), p_4 * -1.23 / 2),
            qml.evolve(qml.PauliX(0), p_4 * -1.23 / 2),  # S_2(p * t) ^ 2
        ],
    )

    @pytest.mark.parametrize("order, expected_expansion", zip((1, 2, 4), expected_expansions))
    def test_recursive_expression_no_queue(self, order, expected_expansion):
        """Test the _recursive_expression function correctly generates the decomposition"""
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]

        with qml.queuing.AnnotatedQueue() as q:
            decomp = _recursive_expression(1.23, order, ops)

        assert len(q) == 0  # No queuing!
        for op1, op2 in zip(decomp, expected_expansion):
            qml.assert_equal(op1, op2)


class TestError:
    """Test the error method of the TrotterProduct class"""

    @pytest.mark.parametrize("fast", (True, False))
    def test_invalid_method(self, fast):
        """Test that passing an invalid method raises an error."""
        method = "crazy"
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), 1.23)
        with pytest.raises(ValueError, match=f"The '{method}' method is not supported"):
            _ = op.error(method, fast=fast)

    def test_one_norm_error_method(self):
        """Test that the one-norm error method works as expected."""
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), time=0.05, order=4)
        expected_error = ((10**5 + 1) / 120) * (0.1**5)

        for computed_error in (
            op.error(method="one-norm-bound"),
            op.error(method="one-norm-bound", fast=False),
        ):
            assert isinstance(computed_error, SpectralNormError)
            assert qnp.isclose(computed_error.error, expected_error)

    def test_commutator_error_method(self):
        """Test that the commutator error method works as expected."""
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), time=0.05, order=2, n=10)
        expected_error = (32 / 3) * (0.05**3) * (1 / 100)

        for computed_error in (
            op.error(method="commutator-bound"),
            op.error(method="commutator-bound", fast=False),
        ):
            assert isinstance(computed_error, SpectralNormError)
            assert qnp.isclose(computed_error.error, expected_error)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "method, expected_error", (("one-norm-bound", 0.001265625), ("commutator-bound", 0.001))
    )
    @pytest.mark.parametrize("interface", ("autograd", "jax", "torch"))
    def test_error_interfaces(self, method, interface, expected_error):
        """Test that the error method works with all interfaces"""

        time = qml.math.array(0.1, like=interface)
        coeffs = qml.math.array([1.0, 0.5], like=interface)

        hamiltonian = qml.dot(coeffs, [qml.X(0), qml.Y(0)])

        op = qml.TrotterProduct(hamiltonian, time, n=2, order=2)
        computed_error = op.error(method=method)

        assert isinstance(computed_error, SpectralNormError)
        assert qml.math.get_interface(computed_error.error) == interface

        assert qnp.isclose(computed_error.error, qml.math.array(expected_error, like=interface))

    @pytest.mark.tf
    def test_tensorflow_interface(self):
        """Test that an error is raised if a TrotterProduct with
        tensorflow parameters is used to compute error."""

        coeffs = qml.math.array([1.0, 0.5], like="tensorflow")
        hamiltonian = qml.dot(coeffs, [qml.X(0), qml.Y(0)])

        op = qml.TrotterProduct(hamiltonian, 1.23, order=2, n=5)
        with pytest.raises(TypeError, match="Calculating error bound for Tensorflow objects"):
            _ = op.error()


class TestResources:
    """Test the resources method of the TrotterProduct class"""

    def test_resources_no_queuing(self):
        """Test that no operations are queued when computing resources."""
        time = 0.5
        hamiltonian = qml.sum(qml.PauliX(0), qml.PauliZ(0))
        op = qml.TrotterProduct(hamiltonian, time, n=5, order=2)

        with qml.queuing.AnnotatedQueue() as q:
            _ = op.resources()

        assert len(q.queue) == 0

    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_resources(self, hamiltonian, hamiltonian_index, order):
        """Test that the resources are tracked accurately."""
        op = qml.TrotterProduct(hamiltonian, 4.2, order=order)

        tracked_resources = op.resources()
        expected_resources = test_resources_data[(hamiltonian_index, order)]

        assert expected_resources == tracked_resources

    @pytest.mark.parametrize("n", (1, 5, 10))
    def test_resources_with_trotter_steps(self, n):
        """Test that the resources are tracked accurately with number of steps."""
        order = 2
        hamiltonian_index = 0

        op = qml.TrotterProduct(test_hamiltonians[hamiltonian_index], 0.5, order=order, n=n)
        tracked_resources = op.resources()

        expected_resources = Resources(
            num_wires=2,
            num_gates=6 * n,
            gate_types=defaultdict(int, {"Evolution": 6 * n}),
            gate_sizes=defaultdict(int, {1: 6 * n}),
            depth=4 * n,
        )

        assert expected_resources == tracked_resources

    def test_resources_integration(self):
        """Test that the resources integrate well with qml.tracker and qml.specs
        resource tracking."""
        time = 0.5
        hamiltonian = qml.sum(qml.X(0), qml.Y(0), qml.Z(1))

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time, n=5, order=2)
            return qml.expval(qml.Z(0))

        expected_resources = Resources(
            num_wires=2,
            num_gates=30,
            gate_types=defaultdict(int, {"Evolution": 30}),
            gate_sizes=defaultdict(int, {1: 30}),
            depth=20,
        )

        with qml.Tracker(dev) as tracker:
            circ()

        spec_resources = qml.specs(circ)()["resources"]
        tracker_resources = tracker.history["resources"][0]

        assert expected_resources == spec_resources
        assert expected_resources == tracker_resources

    def test_resources_and_error(self):
        """Test that we can compute the resources and error together"""
        time = 0.1
        coeffs = qml.math.array([1.0, 0.5])
        hamiltonian = qml.dot(coeffs, [qml.X(0), qml.Y(0)])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time, n=2, order=2)
            return qml.expval(qml.Z(0))

        specs = qml.specs(circ)()

        computed_error = (specs["errors"])["SpectralNormError"]
        computed_resources = specs["resources"]

        # Expected resources and errors (computed by hand)
        expected_resources = Resources(
            num_wires=1,
            num_gates=8,
            gate_types=defaultdict(int, {"Evolution": 8}),
            gate_sizes=defaultdict(int, {1: 8}),
            depth=8,
        )
        expected_error = 0.001

        assert computed_resources == expected_resources
        assert isinstance(computed_error, SpectralNormError)
        assert qnp.isclose(computed_error.error, qml.math.array(expected_error))


class TestDecomposition:
    """Test the decomposition of the TrotterProduct class."""

    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_compute_decomposition(self, hamiltonian, hamiltonian_index, order):
        """Test the decomposition is correct and queues"""
        op = qml.TrotterProduct(hamiltonian, 4.2, order=order)
        with qml.tape.QuantumTape() as tape:
            decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)

        assert decomp == tape.operations  # queue matches decomp with circuit ordering

        decomp = [qml.simplify(op) for op in decomp]
        true_decomp = [
            qml.simplify(op) for op in test_decompositions[(hamiltonian_index, order)][::-1]
        ]
        for op1, op2 in zip(decomp, true_decomp):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_decomposition_new(
        self, hamiltonian, hamiltonian_index, order
    ):  # pylint: disable=unused-argument
        """Tests the decomposition rule implemented with the new system."""
        op = qml.TrotterProduct(hamiltonian, 4.2, order=order)
        for rule in qml.list_decomps(qml.TrotterProduct):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("order", (1, 2))
    @pytest.mark.parametrize("num_steps", (1, 2, 3))
    def test_compute_decomposition_n_steps(self, num_steps, order):
        """Test the decomposition is correct when we set the number of trotter steps"""
        time = 0.5
        hamiltonian = qml.sum(qml.PauliX(0), qml.PauliZ(0))

        if order == 1:
            base_decomp = [
                qml.evolve(qml.PauliZ(0), -0.5 / num_steps),
                qml.evolve(qml.PauliX(0), -0.5 / num_steps),
            ]
        elif order == 2:
            base_decomp = [
                qml.evolve(qml.PauliX(0), -0.25 / num_steps),
                qml.evolve(qml.PauliZ(0), -0.25 / num_steps),
                qml.evolve(qml.PauliZ(0), -0.25 / num_steps),
                qml.evolve(qml.PauliX(0), -0.25 / num_steps),
            ]
        else:
            assert False, "Order must be 1 or 2"

        true_decomp = base_decomp * num_steps

        op = qml.TrotterProduct(hamiltonian, time, n=num_steps, order=order)
        decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)
        for op1, op2 in zip(decomp, true_decomp):
            qml.assert_equal(op1, op2)


class TestIntegration:
    """Test that the TrotterProduct can be executed and differentiated
    through all interfaces."""

    #   Circuit execution tests:
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_execute_circuit(self, hamiltonian, hamiltonian_index, order):
        """Test that the gate executes correctly in a circuit."""
        wires = hamiltonian.wires
        dev = qml.device("reference.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time=4.2, order=order)
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [
                    qml.matrix(op, wire_order=wires)
                    for op in test_decompositions[(hamiltonian_index, order)]
                ],
            )
            @ initial_state
        )
        state = circ()

        assert qnp.allclose(expected_state, state)

    @pytest.mark.parametrize("order", (1, 2))
    @pytest.mark.parametrize("num_steps", (1, 2, 3))
    def test_execute_circuit_n_steps(self, num_steps, order):
        """Test that the circuit executes as expected when we set the number of trotter steps"""
        time = 0.5
        hamiltonian = qml.sum(qml.PauliX(0), qml.PauliZ(0))

        if order == 1:
            base_decomp = [
                qml.exp(qml.PauliZ(0), 0.5j / num_steps),
                qml.exp(qml.PauliX(0), 0.5j / num_steps),
            ]
        elif order == 2:
            base_decomp = [
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
            ]
        else:
            assert False, "Order must be 1 or 2"

        true_decomp = base_decomp * num_steps

        wires = hamiltonian.wires
        dev = qml.device("reference.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time, n=num_steps, order=order)
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y, [qml.matrix(op, wire_order=wires) for op in true_decomp[::-1]]
            )
            @ initial_state
        )
        state = circ()
        assert qnp.allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_jax_execute(self, time):
        """Test that the gate executes correctly in the jax interface."""
        from jax import numpy as jnp

        time = jnp.array(time)
        coeffs = jnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_jaxjit_execute(self, time):
        """Test that the gate executes correctly in the jax interface with jit."""
        import jax
        from jax import numpy as jnp

        time = jnp.array(time)
        c1 = jnp.array(1.23)
        c2 = jnp.array(-0.45)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=2)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circ(time, c1, c2):
            h = qml.sum(
                qml.s_prod(c1, terms[0]),
                qml.s_prod(c2, terms[1]),
            )
            qml.TrotterProduct(h, time, n=2, order=2, check_hermitian=False)
            return qml.state()

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp([c1, c2], terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, c1, c2)
        assert allclose(expected_state, state)

    @pytest.mark.tf
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_tf_execute(self, time):
        """Test that the gate executes correctly in the tensorflow interface."""
        import tensorflow as tf

        time = tf.Variable(time, dtype=tf.complex128)
        coeffs = tf.Variable([1.23, -0.45], dtype=tf.complex128)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.sum(
                qml.s_prod(coeffs[0], terms[0]),
                qml.s_prod(coeffs[1], terms[1]),
            )
            qml.TrotterProduct(h, time, n=2, order=2)

            return qml.state()

        initial_state = tf.Variable([1.0, 0.0, 0.0, 0.0], dtype=tf.complex128)

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = tf.linalg.matvec(
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            ),
            initial_state,
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state)

    @pytest.mark.torch
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_torch_execute(self, time):
        """Test that the gate executes correctly in the torch interface."""
        import torch

        time = torch.tensor(time, dtype=torch.complex64, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state, atol=1e-5)  # float 32 precision issues

    @pytest.mark.autograd
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_autograd_execute(self, time):
        """Test that the gate executes correctly in the autograd interface."""
        time = qnp.array(time)
        coeffs = qnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = qnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert qnp.allclose(expected_state, state)

    @pytest.mark.autograd
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_autograd_gradient(self, order, n):
        """Test that the gradient is computed correctly"""
        time = qnp.array(1.5)
        coeffs = qnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        measured_time_grad, measured_coeff_grad = qml.grad(circ)(time, coeffs)
        reference_time_grad, reference_coeff_grad = qml.grad(reference_circ)(time, coeffs)
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_torch_gradient(self, order, n):
        """Test that the gradient is computed correctly using torch"""
        import torch

        time = torch.tensor(1.5, dtype=torch.complex64, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        time_reference = torch.tensor(1.5, dtype=torch.complex64, requires_grad=True)
        coeffs_reference = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        res_circ = circ(time, coeffs)
        res_circ.backward()
        measured_time_grad = time.grad
        measured_coeff_grad = coeffs.grad

        ref_circ = reference_circ(time_reference, coeffs_reference)
        ref_circ.backward()
        reference_time_grad = time_reference.grad
        reference_coeff_grad = coeffs_reference.grad

        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_tf_gradient(self, order, n):
        """Test that the gradient is computed correctly using tensorflow"""
        import tensorflow as tf

        time = tf.Variable(1.5, dtype=tf.complex128)
        coeffs = tf.Variable([1.23, -0.45], dtype=tf.complex128)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.sum(
                qml.s_prod(coeffs[0], terms[0]),
                qml.s_prod(coeffs[1], terms[1]),
            )
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        with tf.GradientTape() as tape:
            result = circ(time, coeffs)

        measured_time_grad, measured_coeff_grad = tape.gradient(result, (time, coeffs))

        with tf.GradientTape() as tape:
            result = reference_circ(time, coeffs)

        reference_time_grad, reference_coeff_grad = tape.gradient(result, (time, coeffs))
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_jax_gradient(self, order, n):
        """Test that the gradient is computed correctly"""
        import jax
        from jax import numpy as jnp

        time = jnp.array(1.5)
        coeffs = jnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("reference.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        measured_time_grad, measured_coeff_grad = jax.grad(circ, argnums=[0, 1])(time, coeffs)
        reference_time_grad, reference_coeff_grad = jax.grad(reference_circ, argnums=[0, 1])(
            time, coeffs
        )
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)


class TestTrotterizedQfuncInitialization:
    """Test the TrotterizedQfunc class initializes correctly."""

    @staticmethod
    def my_qfunc(time, arg1, arg2, wires, kwarg1=None, kwarg2=0):  # Dummy qfunc for testing
        qml.RX(time * arg1, wires[0])
        qml.RY(time * arg2, wires[0])

        if kwarg1:
            qml.CNOT(wires)
            qml.RZ(time * kwarg2, wires[1])

    def test_error_qfunc(self):
        """Test that an error is raised if a qfunc is not provided."""
        with pytest.raises(ValueError, match="The qfunc must be provided to be trotterized."):
            time = 0.1
            args = (1, 2, 3)
            kwargs = {"kwarg1": 1, "kwarg2": 2, "kwarg3": 3}
            TrotterizedQfunc(time, *args, **kwargs)

    def test_infer_wires(self):
        """Test that if the wires are not passes as kwargs then the last arg is
        assumed to be the wires."""
        time = 0.1
        args = (2.34, -5.6, ["a", "b"])
        kwargs = {"kwarg1": True, "kwarg2": 78.9}

        op = TrotterizedQfunc(time, *args, qfunc=self.my_qfunc, **kwargs)
        assert op.wires == qml.wires.Wires(["a", "b"])

    @pytest.mark.parametrize("order", [0, -1, 3])
    def test_error_order(self, order):
        """Test that an error is raised if the order is not a multiple of 2 or less than 1."""
        time = 0.1
        args = (2.34, -5.6)
        kwargs = {"kwarg1": True, "kwarg2": 78.9}

        with pytest.raises(ValueError, match="The order must be 1 or a positive even integer,"):
            TrotterizedQfunc(time, *args, qfunc=self.my_qfunc, order=order, wires=[0, 1], **kwargs)

    wire_data = (
        [0, 1],
        ["a", "b"],
        (0, "a"),
    )
    args_kwargs_data = (
        ((1.0, 2), {"kwarg1": True, "kwarg2": 34.5}),
        ((0.0, -0.2), {"kwarg1": False, "kwarg2": 678}),
        ((-11.0, 0), {"kwarg1": False, "kwarg2": -0.999}),
    )
    hyperparams_data = (
        (1, 0.1, 1, True),
        (2, 1, 2, True),
        (10, 0.1, 4, False),
        (100, 3.5, 2, False),
    )

    @pytest.mark.parametrize("wires", wire_data)
    @pytest.mark.parametrize("n, time, order, reverse", hyperparams_data)
    @pytest.mark.parametrize("qfunc_args, qfunc_kwargs", args_kwargs_data)
    def test_parameters_and_hyperparameters(
        self, time, qfunc_args, wires, n, order, reverse, qfunc_kwargs
    ):
        """Test that the parameters and hyperparameters are set correctly"""
        op = TrotterizedQfunc(
            time,
            *qfunc_args,
            qfunc=self.my_qfunc,
            n=n,
            order=order,
            reverse=reverse,
            wires=wires,
            **qfunc_kwargs,
        )

        expected_hyperparams = copy.deepcopy(qfunc_kwargs)
        expected_hyperparams["n"] = n
        expected_hyperparams["order"] = order
        expected_hyperparams["reverse"] = reverse
        expected_hyperparams["qfunc"] = self.my_qfunc

        assert op.wires == qml.wires.Wires(wires)
        assert op.data == (time,) + qfunc_args
        assert op.parameters == list((time,) + qfunc_args)
        assert op.hyperparameters == expected_hyperparams

    @pytest.mark.parametrize("wires", wire_data)
    @pytest.mark.parametrize("n, time, order, reverse", hyperparams_data)
    @pytest.mark.parametrize("qfunc_args, qfunc_kwargs", args_kwargs_data)
    def test_trotterize_parameters_and_hyperparameters(
        self, time, qfunc_args, wires, n, order, reverse, qfunc_kwargs
    ):
        """Test that the parameters and hyperparameters are set correctly"""
        op = qml.trotterize(self.my_qfunc, n=n, order=order, reverse=reverse)(
            time,
            *qfunc_args,
            wires=wires,
            **qfunc_kwargs,
        )

        expected_hyperparams = copy.deepcopy(qfunc_kwargs)
        expected_hyperparams["n"] = n
        expected_hyperparams["order"] = order
        expected_hyperparams["reverse"] = reverse
        expected_hyperparams["qfunc"] = self.my_qfunc

        assert op.wires == qml.wires.Wires(wires)
        assert op.data == (time,) + qfunc_args
        assert op.parameters == list((time,) + qfunc_args)
        assert op.hyperparameters == expected_hyperparams

    def test_trotterize_error_if_repeated_kwarg(self):
        """Test that an error is raised if the named kwargs for the qfunc match the
        names of the kwargs of the TrotterizedQfunc class."""

        def my_dummy_qfunc(time, wires, **kwargs):  # pylint:disable=unused-argument
            qml.RZ(time, wires[0])

        for special_key in ["n", "order", "qfunc", "reverse"]:
            with pytest.raises(ValueError, match="Cannot use any of the specailized names:"):
                kwargs = {special_key: 1}
                qml.trotterize(my_dummy_qfunc)(0.1, wires=[0, 1], **kwargs)

    def test_standard_validity(self):
        """Test standard validity criteria using assert_valid."""

        def first_order_expansion(time, theta, phi, wires=(0, 1, 2), flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time * theta, wires[0])
            qml.RY(time * phi, wires[1])
            if flip:
                qml.CNOT(wires=wires[:2])

        op = TrotterizedQfunc(
            0.1,
            *(0.12, -3.45),
            qfunc=first_order_expansion,
            n=1,
            order=2,
            wires=["a", "b", "c"],
            flip=True,
        )
        qml.ops.functions.assert_valid(op, skip_pickle=True)


class TestTrotterizedQfuncIntegration:
    """Test the TrotterizedQfunc decomposes correctly."""

    expected_decomps_order_reverse = (  # computed by hand assuming t = 0.1
        (False, 1, [qml.RX(0.1 * 1.23, "a"), qml.RY(0.1 * 1.23, "a"), qml.CNOT(["a", "b"])]),
        (
            False,
            2,
            [
                qml.RX((0.1 / 2) * 1.23, "a"),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.RX((0.1 / 2) * 1.23, "a"),
            ],
        ),
        (
            False,
            4,
            [
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
            ],
        ),
        (
            False,
            6,
            [
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p6_comp * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p6_comp * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p6_comp * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_6 * p_4 * 0.1 / 2) * 1.23, "a"),
            ],
        ),
        (True, 1, [qml.CNOT(["a", "b"]), qml.RY(0.1 * 1.23, "a"), qml.RX(0.1 * 1.23, "a")]),
        (
            True,
            2,
            [
                qml.CNOT(["a", "b"]),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.RX((0.1 / 2) * 1.23, "a"),
                qml.RX((0.1 / 2) * 1.23, "a"),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
            ],
        ),
        (
            True,
            4,
            [
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RX((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.RY((p4_comp * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RX((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.RY((p_4 * 0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
            ],
        ),
    )

    @pytest.mark.parametrize("reverse, order, expected_decomp", expected_decomps_order_reverse)
    def test_decomposition(self, reverse, order, expected_decomp):
        """Test the decompose method works as expected."""

        def first_order_expansion(time, theta, wires, flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time * theta, wires[0])
            qml.RY(time * theta, wires[0])
            if flip:
                qml.CNOT(wires)

        op = TrotterizedQfunc(
            0.1,
            1.23,
            qfunc=first_order_expansion,
            reverse=reverse,
            order=order,
            wires=["a", "b"],
            flip=True,
        )

        assert op.decomposition() == expected_decomp

    @pytest.mark.parametrize("reverse, order, expected_decomp", expected_decomps_order_reverse)
    def test_private_recursive_qfunc(self, reverse, order, expected_decomp):
        """Test the private _recursive_qfunc function works as expected."""

        def first_order_expansion(time, theta, wires, flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time * theta, wires[0])
            qml.RY(time * theta, wires[0])
            if flip:
                qml.CNOT(wires)

        theta = 1.23
        wires = ["a", "b"]

        decomp = _recursive_qfunc(
            0.1, order, first_order_expansion, wires, reverse, theta, flip=True
        )
        assert decomp == expected_decomp

    expected_decomps_n = (
        (
            1,
            [
                qml.RX((0.1 / 2) * 1.23, "a"),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((0.1 / 2) * 1.23, "a"),
                qml.RX((0.1 / 2) * 1.23, "a"),
            ],
        ),
        (
            2,
            [
                qml.RX((0.1 / 4) * 1.23, "a"),
                qml.RY((0.1 / 4) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((0.1 / 4) * 1.23, "a"),
                qml.RX((0.1 / 4) * 1.23, "a"),
            ],
        ),
        (
            10,
            [
                qml.RX((0.1 / 20) * 1.23, "a"),
                qml.RY((0.1 / 20) * 1.23, "a"),
                qml.CNOT(["a", "b"]),
                qml.CNOT(["a", "b"]),
                qml.RY((0.1 / 20) * 1.23, "a"),
                qml.RX((0.1 / 20) * 1.23, "a"),
            ],
        ),
    )

    @pytest.mark.parametrize("n, expected_decomp", expected_decomps_n)
    def test_decomposition_num_steps(self, n, expected_decomp):
        """Test the decomposition is correct given a number of steps."""

        def first_order_expansion(time, theta, wires, flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time * theta, wires[0])
            qml.RY(time * theta, wires[0])
            if flip:
                qml.CNOT(wires)

        op = TrotterizedQfunc(
            0.1,
            1.23,
            qfunc=first_order_expansion,
            n=n,
            order=2,
            wires=["a", "b"],
            flip=True,
        )

        expected_decomp = expected_decomp * n
        assert op.decomposition() == expected_decomp

    def _generate_simple_decomp_trotterize(self, time, order, reverse, args, wires):
        arg1, arg2 = args

        if order == 1:
            expected_decomp = [
                qml.RX(time * arg1, wires=wires[0]),
                qml.RY(time * arg2, wires=wires[0]),
                qml.MultiRZ(arg1, wires=wires),
                qml.CNOT([wires[0], wires[1]]),
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]]),
                qml.CNOT([wires[0], wires[1]]),
                qml.QFT(wires=wires[1:-1]),
            ]

            if reverse:
                expected_decomp = expected_decomp[::-1]

        if order == 2:
            expected_decomp = [
                qml.RX((time / 2) * arg1, wires=wires[0]),
                qml.RY((time / 2) * arg2, wires=wires[0]),
                qml.MultiRZ(arg1, wires=wires),
                qml.CNOT([wires[0], wires[1]]),
                qml.ControlledPhaseShift((time / 2) * arg2, wires=[wires[1], wires[0]]),
                qml.CNOT([wires[0], wires[1]]),
                qml.QFT(wires=wires[1:-1]),
            ]

            if reverse:
                expected_decomp = expected_decomp[::-1]

            expected_decomp = expected_decomp + expected_decomp[::-1]

        if order == 4:
            expected_decomp1 = [
                qml.RX((p_4 * time / 2) * arg1, wires=wires[0]),
                qml.RY((p_4 * time / 2) * arg2, wires=wires[0]),
                qml.MultiRZ(arg1, wires=wires),
                qml.CNOT([wires[0], wires[1]]),
                qml.ControlledPhaseShift((p_4 * time / 2) * arg2, wires=[wires[1], wires[0]]),
                qml.CNOT([wires[0], wires[1]]),
                qml.QFT(wires=wires[1:-1]),
            ]

            expected_decomp1 = (
                expected_decomp1[::-1] + expected_decomp1
                if reverse
                else expected_decomp1 + expected_decomp1[::-1]
            )

            expected_decomp2 = [
                qml.RX((p4_comp * time / 2) * arg1, wires=wires[0]),
                qml.RY((p4_comp * time / 2) * arg2, wires=wires[0]),
                qml.MultiRZ(arg1, wires=wires),
                qml.CNOT([wires[0], wires[1]]),
                qml.ControlledPhaseShift((p4_comp * time / 2) * arg2, wires=[wires[1], wires[0]]),
                qml.CNOT([wires[0], wires[1]]),
                qml.QFT(wires=wires[1:-1]),
            ]

            expected_decomp2 = (
                expected_decomp2[::-1] + expected_decomp2
                if reverse
                else expected_decomp2 + expected_decomp2[::-1]
            )

            expected_decomp = expected_decomp1 * 2 + expected_decomp2 + expected_decomp1 * 2

        return expected_decomp

    #   Circuit execution tests:
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("reverse", (True, False))
    def test_execute_circuit(self, n, order, reverse):
        """Test that the gate executes correctly in a circuit."""
        time = 0.1
        wires = ["aux1", "aux2", 0, 1, "target"]
        arg1 = 2.34
        arg2 = -6.78
        args = (arg1, arg2)
        kwargs = {"kwarg1": True, "kwarg2": 1}

        def my_qfunc(time, arg1, arg2, wires, kwarg1=False, kwarg2=None):
            """Arbitrarily complex qfunc"""
            qml.RX(time * arg1, wires=wires[0])
            qml.RY(time * arg2, wires=wires[0])
            qml.MultiRZ(arg1, wires=wires)

            if kwarg1:
                qml.CNOT([wires[0], wires[1]])
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]])
                qml.CNOT([wires[0], wires[1]])

            for _ in range(kwarg2):
                qml.QFT(wires=wires[1:-1])

        expected_t = time / n
        expected_decomp = self._generate_simple_decomp_trotterize(
            expected_t, order, reverse, args, wires
        )
        expected_decomp = expected_decomp * n

        @qml.qnode(qml.device("default.qubit", wires=wires))
        def circ(time, alpha, beta, wires, **kwargs):
            TrotterizedQfunc(
                time,
                alpha,
                beta,
                qfunc=my_qfunc,
                n=n,
                order=order,
                reverse=reverse,
                wires=wires,
                **kwargs,
            )
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=wires) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )

        state = circ(time, *args, wires=wires, **kwargs)
        assert qnp.allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("reverse", (True, False))
    def test_jax_execute(self, n, order, reverse):
        """Test that the gate executes correctly in the jax interface."""
        from jax import numpy as jnp

        time = jnp.array(0.1)
        wires = ["aux1", "aux2", 0, 1, "target"]
        arg1 = jnp.array(2.34)
        arg2 = jnp.array(-6.78)
        args = (arg1, arg2)
        kwargs = {"kwarg1": True, "kwarg2": 1}

        def my_qfunc(time, arg1, arg2, wires, kwarg1=False, kwarg2=None):
            """Arbitrarily complex qfunc"""
            qml.RX(time * arg1, wires=wires[0])
            qml.RY(time * arg2, wires=wires[0])
            qml.MultiRZ(arg1, wires=wires)

            if kwarg1:
                qml.CNOT([wires[0], wires[1]])
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]])
                qml.CNOT([wires[0], wires[1]])

            for _ in range(kwarg2):
                qml.QFT(wires=wires[1:-1])

        expected_t = time / n
        expected_decomp = self._generate_simple_decomp_trotterize(
            expected_t, order, reverse, args, wires
        )
        expected_decomp = expected_decomp * n

        @qml.qnode(qml.device("default.qubit", wires=wires))
        def circ(time, alpha, beta, wires, **kwargs):
            TrotterizedQfunc(
                time,
                alpha,
                beta,
                qfunc=my_qfunc,
                n=n,
                order=order,
                reverse=reverse,
                wires=wires,
                **kwargs,
            )
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        initial_state = jnp.array(initial_state)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=wires) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )

        state = circ(time, *args, wires=wires, **kwargs)
        assert allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("reverse", (True, False))
    @pytest.mark.parametrize("method", ("backprop", "parameter-shift"))
    def test_jaxjit_execute(self, n, order, reverse, method):
        """Test that the gate executes correctly in the jax interface."""
        import jax
        from jax import numpy as jnp

        time = jnp.array(0.1)
        wires = ("aux1", "aux2", 0, 1, "target")
        arg1 = jnp.array(2.34)
        arg2 = jnp.array(-6.78)
        args = (arg1, arg2)
        kwargs = {"kwarg1": True, "kwarg2": 1}

        def my_qfunc(time, arg1, arg2, wires, kwarg1=False, kwarg2=None):
            """Arbitrarily complex qfunc"""
            qml.RX(time * arg1, wires=wires[0])
            qml.RY(time * arg2, wires=wires[0])
            qml.MultiRZ(arg1, wires=wires)

            if kwarg1:
                qml.CNOT([wires[0], wires[1]])
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]])
                qml.CNOT([wires[0], wires[1]])

            for _ in range(kwarg2):
                qml.QFT(wires=wires[1:-1])

        expected_t = time / n
        expected_decomp = self._generate_simple_decomp_trotterize(
            expected_t, order, reverse, args, wires
        )
        expected_decomp = expected_decomp * n

        @partial(jax.jit, static_argnames=["wires", "kwarg1", "kwarg2"])
        @qml.qnode(qml.device("default.qubit", wires=wires), interface="jax", diff_method=method)
        def circ(time, alpha, beta, wires, **kwargs):
            TrotterizedQfunc(
                time,
                alpha,
                beta,
                qfunc=my_qfunc,
                n=n,
                order=order,
                reverse=reverse,
                wires=wires,
                **kwargs,
            )
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        initial_state = jnp.array(initial_state)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=wires) for op in expected_decomp[::-1]],
            )
            @ initial_state
        )

        state = circ(time, *args, wires=wires, **kwargs)
        assert allclose(expected_state, state)

    #   Circuit gradient tests:
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("reverse", (True, False))
    @pytest.mark.parametrize("method", ("backprop", "parameter-shift"))
    def test_gradient(self, n, order, reverse, method):
        """Test that the gradient is computed correctly"""
        time = qnp.array(0.1)
        wires = ["aux1", "aux2", 0, 1, "target"]
        arg1 = qnp.array(2.34)
        arg2 = qnp.array(-6.78)
        kwargs = {"kwarg1": True, "kwarg2": 1}

        def my_qfunc(time, arg1, arg2, wires, kwarg1=False, kwarg2=None):
            """Arbitrarily complex qfunc"""
            qml.RX(time * arg1, wires=wires[0])
            qml.RY(time * arg2, wires=wires[0])
            qml.MultiRZ(arg1, wires=wires)

            if kwarg1:
                qml.CNOT([wires[0], wires[1]])
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]])
                qml.CNOT([wires[0], wires[1]])

            for _ in range(kwarg2):
                qml.QFT(wires=wires[1:-1])

        @qml.qnode(qml.device("default.qubit", wires=wires), diff_method=method)
        def circ(time, alpha, beta, wires, **kwargs):
            TrotterizedQfunc(
                time,
                alpha,
                beta,
                qfunc=my_qfunc,
                n=n,
                order=order,
                reverse=reverse,
                wires=wires,
                **kwargs,
            )
            return qml.expval(qml.Hadamard(wires[0]))

        @qml.qnode(qml.device("default.qubit", wires=wires), diff_method=method)
        def reference_circ(time, alpha, beta, wires):
            with qml.QueuingManager.stop_recording():
                expected_t = time / n
                expected_decomp = self._generate_simple_decomp_trotterize(
                    expected_t, order, reverse, (alpha, beta), wires
                )
                expected_decomp = expected_decomp * n

            for op in expected_decomp:
                qml.apply(op)

            return qml.expval(qml.Hadamard(wires[0]))

        measured_time_grad, measured_arg1_grad, measured_arg2_grad = qml.grad(circ)(
            time, arg1, arg2, wires, **kwargs
        )
        reference_time_grad, reference_arg1_grad, reference_arg2_grad = qml.grad(reference_circ)(
            time, arg1, arg2, wires
        )
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_arg1_grad, reference_arg1_grad)
        assert allclose(measured_arg2_grad, reference_arg2_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("reverse", (True, False))
    @pytest.mark.parametrize("method", ("backprop", "parameter-shift"))
    def test_jax_gradient(self, n, order, reverse, method):
        """Test that the gradient is computed correctly"""
        import jax
        from jax import numpy as jnp

        time = jnp.array(0.1)
        wires = ["aux1", "aux2", 0, 1, "target"]
        arg1 = jnp.array(2.34)
        arg2 = jnp.array(-6.78)
        args = (arg1, arg2)
        kwargs = {"kwarg1": True, "kwarg2": 1}

        def my_qfunc(time, arg1, arg2, wires, kwarg1=False, kwarg2=None):
            """Arbitrarily complex qfunc"""
            qml.RX(time * arg1, wires=wires[0])
            qml.RY(time * arg2, wires=wires[0])
            qml.MultiRZ(arg1, wires=wires)

            if kwarg1:
                qml.CNOT([wires[0], wires[1]])
                qml.ControlledPhaseShift(time * arg2, wires=[wires[1], wires[0]])
                qml.CNOT([wires[0], wires[1]])

            for _ in range(kwarg2):
                qml.QFT(wires=wires[1:-1])

        expected_t = time / n
        expected_decomp = self._generate_simple_decomp_trotterize(
            expected_t, order, reverse, args, wires
        )
        expected_decomp = expected_decomp * n

        @qml.qnode(qml.device("default.qubit", wires=wires), diff_method=method)
        def circ(time, alpha, beta, wires, **kwargs):
            TrotterizedQfunc(
                time,
                alpha,
                beta,
                qfunc=my_qfunc,
                n=n,
                order=order,
                reverse=reverse,
                wires=wires,
                **kwargs,
            )
            return qml.expval(qml.Hadamard(wires[0]))

        @qml.qnode(qml.device("default.qubit", wires=wires), diff_method=method)
        def reference_circ(time, alpha, beta, wires):
            with qml.QueuingManager.stop_recording():
                expected_t = time / n
                expected_decomp = self._generate_simple_decomp_trotterize(
                    expected_t, order, reverse, (alpha, beta), wires
                )
                expected_decomp = expected_decomp * n

            for op in expected_decomp:
                qml.apply(op)

            return qml.expval(qml.Hadamard(wires[0]))

        measured_time_grad, measured_arg1_grad, measured_arg2_grad = jax.grad(
            circ, argnums=[0, 1, 2]
        )(time, arg1, arg2, wires, **kwargs)
        reference_time_grad, reference_arg1_grad, reference_arg2_grad = jax.grad(
            reference_circ, argnums=[0, 1, 2]
        )(time, arg1, arg2, wires)
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_arg1_grad, reference_arg1_grad)
        assert allclose(measured_arg2_grad, reference_arg2_grad)
