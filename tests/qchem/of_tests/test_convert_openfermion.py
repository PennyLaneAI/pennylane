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
Unit tests for functions needed for converting ``QubitOperator``from OpenFermion to
PennyLane ``LinearCombination`` and vice versa.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np

openfermion = pytest.importorskip("openfermion")


class TestFromOpenFermion:

    OPS = (
        (
            (openfermion.QubitOperator("X0", 1.2) + openfermion.QubitOperator("Z1", 2.4)),
            (1.2 * qml.X(0) + 2.4 * qml.Z(1)),
        ),
        (
            (openfermion.QubitOperator("X0 X2", 0.3) + openfermion.QubitOperator("Z1 Z0", 0.5)),
            (0.3 * qml.X(0) @ qml.X(2) + 0.5 * qml.Z(1) @ qml.Z(0)),
        ),
        (openfermion.QubitOperator(), (0.0 * qml.I(0))),
        (
            (
                0.1 * openfermion.QubitOperator("X0")
                + 0.5 * openfermion.QubitOperator("Y1")
                + 0.2 * openfermion.QubitOperator("Z2")
            ),
            (0.1 * qml.X(0) + 0.2 * qml.Z(2) + 0.5 * qml.Y(1)),
        ),
        (
            (openfermion.QubitOperator("Z0", 0.1) - openfermion.QubitOperator("Z1", 0.5)),
            (qml.ops.Sum(0.1 * qml.Z(0), -0.5 * qml.Z(1))),
        ),
        (
            (openfermion.QubitOperator("X0 X1", 1.0) + openfermion.QubitOperator("Y0 Y1", 0.5)),
            (qml.ops.Sum(1.0 * qml.X(0) @ qml.X(1), 0.5 * qml.Y(0) @ qml.Y(1))),
        ),
    )

    @pytest.mark.parametrize("of_op, pl_op", OPS)
    def test_conversion(self, of_op, pl_op):
        """Test conversion from ``QubitOperator`` to PennyLane."""
        converted_pl_op = qml.from_openfermion(of_op)
        assert converted_pl_op.compare(pl_op)

    def test_tol(self):
        """Test with complex coefficients."""
        q_op = openfermion.QubitOperator("X0", complex(1.0, 1e-8)) + openfermion.QubitOperator(
            "Z1", complex(1.3, 1e-8)
        )

        pl_op = qml.from_openfermion(q_op, tol=1e-6)
        assert ~np.any(pl_op.coeffs.imag)

        pl_op = qml.from_openfermion(q_op, tol=1e-10)
        assert np.any(pl_op.coeffs.imag)

    def test_sum(self):
        """Test that the from_openfermion method yields a :class:`~.Sum` object if requested."""
        q_op = openfermion.QubitOperator("X0 X1", 0.25) + openfermion.QubitOperator("Z1 Z0", 0.75)

        assert type(qml.from_openfermion(q_op)) == qml.ops.LinearCombination
        assert (
            type(qml.from_openfermion(q_op, format="LinearCombination"))
            == qml.ops.LinearCombination
        )
        assert type(qml.from_openfermion(q_op, format="Sum")) == qml.ops.Sum

    def test_invalid_format(self):
        """Test if error is raised if format is invalid."""
        q_op = openfermion.QubitOperator("X0")

        with pytest.raises(
            ValueError,
            match="format must be a Sum or LinearCombination, got: invalid_format",
        ):
            qml.from_openfermion(q_op, format="invalid_format")


class TestToOpenFermion:

    FERMI_AND_OF_OPS = (
        ((qml.fermi.FermiWord({(0, 0): "+", (1, 1): "-"})), (openfermion.FermionOperator("0^ 1"))),
        (
            (qml.fermi.FermiWord({(1, 0): "+", (0, 1): "-", (2, 3): "+", (3, 2): "-"})),
            (openfermion.FermionOperator("1 0^ 3^ 2")),
        ),
        (
            (
                qml.fermi.FermiSentence(
                    {
                        qml.fermi.FermiWord(
                            {(1, 0): "+", (0, 1): "-", (2, 3): "+", (3, 2): "-"}
                        ): 0.5,
                        qml.fermi.FermiWord({(0, 0): "+", (1, 1): "-"}): 0.3,
                    }
                )
            ),
            (
                0.3 * openfermion.FermionOperator("0^ 1")
                + 0.5 * openfermion.FermionOperator("1 0^ 3^ 2")
            ),
        ),
        (
            (0.5 * qml.Z(0) @ qml.X(2) + 0.1 * qml.X(1) @ qml.Z(0)),
            (openfermion.QubitOperator("Z0 X2", 0.5) + openfermion.QubitOperator("X1 Z0", 0.1)),
        ),
        (
            (1.0 * qml.Z(0) + 0.2 * qml.Y(2) + 0.25 * qml.X(1)),
            (
                1.0 * openfermion.QubitOperator("Z0")
                + 0.25 * openfermion.QubitOperator("X1")
                + 0.2 * openfermion.QubitOperator("Y2")
            ),
        ),
        (
            (qml.ops.Sum(qml.Y(1) @ qml.X(0), qml.X(0) @ qml.Z(2))),
            (openfermion.QubitOperator("Y1 X0") + openfermion.QubitOperator("X0 Z2")),
        ),
    )

    @pytest.mark.parametrize("fermi_op, of_op", FERMI_AND_OF_OPS)
    def test_conversion(self, fermi_op, of_op):
        converted_of_op = qml.to_openfermion(fermi_op)
        assert converted_of_op == of_op

    def test_tol(self):
        """Test the to_openfermion function with complex coefficients."""
        pl_op = complex(1.2, 1e-08) * qml.X(0) + complex(2.4, 1e-08) * qml.Z(1)

        q_op = qml.to_openfermion(pl_op, tol=1e-6)
        coeffs = np.array(list(q_op.terms.values()))
        assert ~np.any(coeffs.imag)

        q_op = qml.to_openfermion(pl_op, tol=1e-10)
        coeffs = np.array(list(q_op.terms.values()))
        assert np.any(coeffs.imag)

    INVALID_OPS = (
        qml.operation.Tensor(qml.PauliZ(0), qml.QuadOperator(0.1, wires=1)),
        qml.prod(qml.PauliX(0), qml.Hadamard(1)),
        qml.sum(qml.PauliZ(0), qml.Hadamard(1)),
    )

    @pytest.mark.parametrize("op", INVALID_OPS)
    def test_not_xyz(self, op):
        r"""Test if the conversion complains about non Pauli matrix observables in the ``LinearCombination``."""
        _match = "Expected a Pennylane operator with a valid Pauli word representation,"

        pl_op = qml.ops.LinearCombination(
            np.array([0.1 + 0.0j, 0.0]), [qml.operation.Tensor(qml.PauliX(0)), op]
        )
        with pytest.raises(ValueError, match=_match):
            qml.to_openfermion(qml.to_openfermion(pl_op))

    def test_wires_not_covered(self):
        r"""Test if the conversion complains about supplied wires not covering ops wires."""
        pl_op = qml.ops.LinearCombination(
            np.array([0.1, 0.2]),
            [
                qml.operation.Tensor(qml.PauliX(wires=["w0"])),
                qml.operation.Tensor(qml.PauliY(wires=["w0"]), qml.PauliZ(wires=["w2"])),
            ],
        )

        with pytest.raises(
            ValueError,
            match="Supplied `wires` does not cover all wires defined in `ops`.",
        ):
            qml.to_openfermion(
                pl_op,
                wires=qml.wires.Wires(["w0", "w1"]),
            )

    def test_custom_wires(self):
        """Test custom (swapped) wires."""
        q_op_ref = (
            openfermion.QubitOperator("X2", 1.2)
            + openfermion.QubitOperator("Z1", 2.4)
            + openfermion.QubitOperator("Y0", 0.2)
        )

        pl_op = 1.2 * qml.X(0) + 2.4 * qml.Z(1) + 0.2 * qml.Y(2)
        # Custom mapping where the first and third qubits are swapped.
        q_op = qml.to_openfermion(pl_op, wires={0: 2, 1: 1, 2: 0})

        assert q_op == q_op_ref
