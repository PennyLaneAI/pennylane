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
Test private functions computing Trotter-Suzuki product formula errors.
"""

import pytest

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.resource.error.trotter_error import (
    _commutator_error,
    _compute_repetitions,
    _flatten_trotter,
    _generate_combinations,
    _one_norm_error,
    _recursive_flatten,
    _recursive_nested_commutator,
    _simplify,
    _spectral_norm,
)

p_4 = (4 - 4 ** (1 / 3)) ** -1


@pytest.mark.parametrize(
    "order, expected_val",
    (
        (1, 1),
        (2, 2),
        (4, 10),
        (6, 50),  # Computed by hand
    ),
)
def test_compute_repetitions(order, expected_val):
    assert _compute_repetitions(order) == expected_val


@pytest.mark.parametrize(
    "op, fast, expected_norm",
    (
        (qml.s_prod(0, qml.I(0)), False, 0),
        (qml.s_prod(1.23, qml.X(0)), False, 1.23),
        (
            qml.sum(qml.Z(1), qml.s_prod(-1.23, qml.X(0))),
            True,
            2.23,
        ),
        (
            qml.sum(qml.Z(1), qml.s_prod(-1.23, qml.I(0))),
            False,
            2.23,
        ),
    ),
)
def test_spectral_norm_pauli(op, fast, expected_norm):
    assert _spectral_norm(op, fast=fast) == expected_norm


@pytest.mark.parametrize(
    "op, fast, expected_norm",  # computed by hand
    (
        (qml.Hadamard(0), False, 1),
        (qml.Hadamard(0), True, 1),
        (qml.RX(1.23, 0), True, 1),
        (qml.s_prod(-0.5, qml.Hadamard(0)), False, 0.5),
    ),
)
def test_spectral_norm_non_pauli(op, fast, expected_norm):
    if not fast:
        assert _spectral_norm(op, False) == expected_norm
    else:
        assert _spectral_norm(op, True) >= expected_norm


combination_data = (
    (0, 0, ()),
    (0, 123, ()),
    (1, 0, ((0,),)),
    (5, 0, ((0, 0, 0, 0, 0),)),
    (1, 123, ((123,),)),
    (2, 1, ((0, 1), (1, 0))),
    (2, 2, ((0, 2), (1, 1), (2, 0))),
    (2, 3, ((0, 3), (1, 2), (2, 1), (3, 0))),
    (3, 1, ((0, 0, 1), (0, 1, 0), (1, 0, 0))),
)


@pytest.mark.parametrize("num_var, req_sum, expected_tup", combination_data)
def test_generate_combinations(num_var, req_sum, expected_tup):
    assert _generate_combinations(num_var, req_sum) == expected_tup


@pytest.mark.parametrize(
    "A, B, alpha, final_op",  # computed by hand
    (
        (qml.X(0), qml.Y(0), 0, qml.Y(0)),
        (qml.X(0), qml.Y(0), 1, qml.s_prod(2j, qml.Z(0))),
        (qml.X(0), qml.Y(0), 2, qml.s_prod(4, qml.Y(0))),
        (qml.X(0), qml.Y(0), 3, qml.s_prod(8j, qml.Z(0))),
        (qml.X(0), qml.Y(1), 3, qml.s_prod(0, qml.I(wires=(0, 1)))),
        (
            qml.RX(1.23, 0),
            qml.RZ(-0.5, 0),
            1,
            qml.sum(
                qml.prod(qml.RX(1.23, 0), qml.RZ(-0.5, 0)),
                qml.s_prod(-1, qml.prod(qml.RZ(-0.5, 0), qml.RX(1.23, 0))),
            ),
        ),
        (
            qml.s_prod(-0.5, qml.prod(qml.X(0), qml.RX(123, 1))),
            qml.RZ(-0.5, 2),
            4,
            qml.s_prod(0, qml.I(wires=(0, 1, 2))),
        ),
    ),
)
def test_recursive_nested_commutator(A, B, alpha, final_op):
    m_expected = qml.matrix(final_op)
    computed_op = _recursive_nested_commutator(A, B, alpha)

    try:
        m_computed = qml.matrix(computed_op, wire_order=final_op.wires)
    except qml.operation.DecompositionUndefinedError:
        pr = computed_op.pauli_rep
        pr.simplify()
        m_computed = pr.to_mat(wire_order=final_op.wires.tolist())

    assert qnp.allclose(m_computed, m_expected)


@pytest.mark.parametrize(
    "order, num_ops, t, expected_indicies_and_coeffs",
    (
        (1, 2, 1, ([0, 1], [1, 1])),
        (1, 3, 0.25, ([0, 1, 2], [0.25, 0.25, 0.25])),
        (2, 3, 1, ([0, 1, 2, 2, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])),
        (2, 2, 8, ([0, 1, 1, 0], [4, 4, 4, 4])),
        (
            4,
            2,
            1,
            (
                [0, 1, 1, 0] * 5,
                [0.5 * p_4] * 8 + [0.5 * (1 - 4 * p_4)] * 4 + [0.5 * p_4] * 8,
            ),
        ),
    ),
)
def test_recursive_flatten(order, num_ops, t, expected_indicies_and_coeffs):
    computed_indicies, computed_coeffs = _recursive_flatten(order, num_ops, t)
    expected_indicies, expected_coeffs = expected_indicies_and_coeffs

    assert computed_coeffs == expected_coeffs
    assert computed_indicies == expected_indicies


@pytest.mark.parametrize(
    "index_lst, coeffs_lst, simplified_index_lst, simplified_coeffs_lst",
    (  # computed by hand
        (
            [0, 1, 2, 3],
            [123, -0.45, -6.7, 89],
            [0, 1, 2, 3],
            [123, -0.45, -6.7, 89],
        ),  # no simplification
        ([0, 1, 1, 0], [1.23, -4, 5, 6.7], [0, 1, 0], [1.23, 1, 6.7]),
        (
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 2, 3, 4, -5, -6, -7, -8],
            [0, 1, 0, 1, 0],
            [1, 5, -1, -13, -8],
        ),
    ),
)
def test_simplify_flatten(index_lst, coeffs_lst, simplified_index_lst, simplified_coeffs_lst):
    computed_index_lst, computed_coeffs_lst = _simplify(index_lst, coeffs_lst)
    assert computed_index_lst == simplified_index_lst
    assert computed_coeffs_lst == simplified_coeffs_lst


@pytest.mark.parametrize(
    "order, num_ops, expected_indicies_and_coeffs",
    (
        (1, 3, ([0, 1, 2], [1] * 3)),
        (2, 3, ([0, 1, 2, 1, 0], [0.5, 0.5, 1, 0.5, 0.5])),
        (2, 2, ([0, 1, 0], [1 / 2, 1, 1 / 2])),
        (
            4,
            2,
            (
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [
                    p_4 / 2,
                    p_4,
                    p_4,
                    p_4,
                    (1 - 3 * p_4) / 2,
                    1 - 4 * p_4,
                    (1 - 3 * p_4) / 2,
                    p_4,
                    p_4,
                    p_4,
                    p_4 / 2,
                ],
            ),
        ),
    ),
)
def test_flatten_trotter(order, num_ops, expected_indicies_and_coeffs):
    computed_indicies, computed_coeffs = _flatten_trotter(num_ops, order)
    expected_indicies, expected_coeffs = expected_indicies_and_coeffs

    assert qnp.allclose(computed_coeffs, expected_coeffs)
    assert computed_indicies == expected_indicies


class TestErrorFunctions:
    """Test the error estimation functionality"""

    one_norm_error_dict = {
        1: lambda a1, a2, t, n: (1 / n) * (t * (abs(a1) + abs(a2))) ** 2,
        2: lambda a1, a2, t, n: (1 / n**2) * (3 / 2) * (t * (abs(a1) + abs(a2))) ** 3,
        4: lambda a1, a2, t, n: (1 / n**4) * ((10**5 + 1) / 120) * (t * (abs(a1) + abs(a2))) ** 5,
    }

    @pytest.mark.parametrize(
        "steps, order",
        (
            (1, 1),
            (10, 1),
            (100, 1),
            (1, 2),
            (10, 2),
            (100, 2),
            (1, 4),
            (10, 4),
            (100, 4),
        ),
    )
    @pytest.mark.parametrize(
        "h_coeffs",
        (
            (1, 1),
            (0.5, 2),
            (1.23, -0.45),
            (1j, 3.14j),
        ),
    )
    @pytest.mark.parametrize("time", (1, 0.5, 0.25, 0.01))
    def test_one_norm_error(self, steps, order, time, h_coeffs):
        h_ops = (qml.s_prod(h_coeffs[0], qml.X(0)), qml.s_prod(h_coeffs[1], qml.Z(0)))
        expected_error = self.one_norm_error_dict[order](h_coeffs[0], h_coeffs[1], time, steps)

        computed_error = _one_norm_error(h_ops, time, order, steps, fast=False)
        assert qnp.isclose(computed_error, expected_error, atol=1e-12)

        computed_error_fast = _one_norm_error(h_ops, time, order, steps, fast=True)
        assert (
            qnp.isclose(computed_error_fast, expected_error, atol=1e-12)
            or computed_error_fast > expected_error
        )

    commutator_error_dict = {
        1: lambda a1, a2, t, n: (1 / n) * 4 * t**2 * abs(a1) * abs(a2),
        2: lambda a1, a2, t, n: (1 / n**2)
        * (16 / 3)
        * t**3
        * abs(a1)
        * abs(a2)
        * (abs(a1) + abs(a2)),
    }

    @pytest.mark.parametrize(
        "steps, order",
        (
            (1, 1),
            (10, 1),
            (100, 1),
            (1, 2),
            (10, 2),
            (100, 2),
        ),
    )
    @pytest.mark.parametrize(
        "h_coeffs",
        (
            (1, 1),
            (0.5, 2),
            (1.23, -0.45),
            (1j, 3.14j),
        ),
    )
    @pytest.mark.parametrize("time", (1, 0.5, 0.25, 0.01))
    def test_commutator_error(self, steps, order, time, h_coeffs):
        h_ops = (qml.s_prod(h_coeffs[0], qml.X(0)), qml.s_prod(h_coeffs[1], qml.Z(0)))
        expected_error = self.commutator_error_dict[order](h_coeffs[0], h_coeffs[1], time, steps)

        computed_error = _commutator_error(h_ops, time, order, steps, fast=False)
        assert qnp.isclose(computed_error, expected_error, atol=1e-12)

        computed_error_fast = _commutator_error(h_ops, time, order, steps, fast=True)
        assert (
            qnp.isclose(computed_error_fast, expected_error, atol=1e-12)
            or computed_error_fast > expected_error
        )
