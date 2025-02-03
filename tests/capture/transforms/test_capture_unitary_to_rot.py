# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the `UnitaryToRotInterpreter` class."""

# pylint:disable=protected-access, wrong-import-position

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import qnode_prim
from pennylane.transforms.unitary_to_rot import (
    UnitaryToRotInterpreter,
    unitary_to_rot_plxpr_to_plxpr,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


def check_jaxpr_eqn(eqn, expected_op, expected_wire) -> bool:
    return eqn.primitive == expected_op._primitive and eqn.primitive.invars[1] == expected_wire


class TestUnitaryToRotInterpreter:
    """Unit tests for the UnitaryToRotInterpreter class for decomposing plxpr."""

    def test_one_qubit_conversion(self):
        """Test that a simple one qubit unitary can be decomposed correctly."""

        @UnitaryToRotInterpreter()
        def f(U):
            qml.QubitUnitary(U, 0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        jaxpr = jax.make_jaxpr(f)(U)
        assert jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert jaxpr.eqns[-3].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    # TODO: Currently only supports three CNOT decomposition
    def test_two_qubit_three_cnot_conversion(self):
        """Test that a two qubit unitary can be decompose correctly."""
        U1 = qml.Rot(1.0, 2.0, 3.0, wires=0)
        U2 = qml.Rot(1.0, 2.0, 3.0, wires=1)
        U = qml.prod(U1, U2).matrix()

        @UnitaryToRotInterpreter()
        def f(U):
            qml.QubitUnitary(U, [0, 1])
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(U)
        # C
        assert jaxpr.eqns[-20].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-19].primitive == qml.RY._primitive
        assert jaxpr.eqns[-18].primitive == qml.RZ._primitive
        # D
        assert jaxpr.eqns[-17].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-16].primitive == qml.RY._primitive
        assert jaxpr.eqns[-15].primitive == qml.RZ._primitive

        # CNOT 1
        assert jaxpr.eqns[-14].primitive == qml.CNOT._primitive

        # RZ RY
        assert jaxpr.eqns[-13].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-12].primitive == qml.RY._primitive
        # CNOT 2
        assert jaxpr.eqns[-11].primitive == qml.CNOT._primitive

        # RY
        assert jaxpr.eqns[-10].primitive == qml.RY._primitive
        # CNOT 3
        assert jaxpr.eqns[-9].primitive == qml.CNOT._primitive

        # A
        assert jaxpr.eqns[-8].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-7].primitive == qml.RY._primitive
        assert jaxpr.eqns[-6].primitive == qml.RZ._primitive

        # B
        assert jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


class TestQNodeIntegration:
    """Test that transform works at the QNode level."""

    def test_one_qubit_conversion_qnode(self):
        """Test that you can integrate the transform at the QNode level."""
        dev = qml.device("default.qubit", wires=1)

        @UnitaryToRotInterpreter()
        @qml.qnode(dev)
        def f(U):
            qml.QubitUnitary(U, 0)
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
        jaxpr = jax.make_jaxpr(f)(U)
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

        # Qubit unitary decomposition
        assert qfunc_jaxpr.eqns[-6].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qml.RZ._primitive

        # Hadamard
        assert qfunc_jaxpr.eqns[-3].primitive == qml.Hadamard._primitive

        # Measurement
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive

    # TODO: Currently only supports three CNOT decomposition
    def test_two_qubit_three_cnot_conversion_qnode(self):
        """Test that a two qubit unitary can be decompose correctly."""
        dev = qml.device("default.qubit", wires=1)

        U1 = qml.Rot(1.0, 2.0, 3.0, wires=0)
        U2 = qml.Rot(1.0, 2.0, 3.0, wires=1)
        U = qml.prod(U1, U2).matrix()

        @UnitaryToRotInterpreter()
        @qml.qnode(dev)
        def f(U):
            qml.QubitUnitary(U, [0, 1])
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(U)
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        # C
        assert qfunc_jaxpr.eqns[-20].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-19].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-18].primitive == qml.RZ._primitive
        # D
        assert qfunc_jaxpr.eqns[-17].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-16].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-15].primitive == qml.RZ._primitive

        # CNOT 1
        assert qfunc_jaxpr.eqns[-14].primitive == qml.CNOT._primitive

        # RZ RY
        assert qfunc_jaxpr.eqns[-13].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-12].primitive == qml.RY._primitive
        # CNOT 2
        assert qfunc_jaxpr.eqns[-11].primitive == qml.CNOT._primitive

        # RY
        assert qfunc_jaxpr.eqns[-10].primitive == qml.RY._primitive
        # CNOT 3
        assert qfunc_jaxpr.eqns[-9].primitive == qml.CNOT._primitive

        # A
        assert qfunc_jaxpr.eqns[-8].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-7].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-6].primitive == qml.RZ._primitive

        # B
        assert qfunc_jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[-4].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[-3].primitive == qml.RZ._primitive

        # Measurement
        assert qfunc_jaxpr.eqns[-2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[-1].primitive == qml.measurements.ExpectationMP._obs_primitive


def test_unitary_to_rot_plxpr_to_plxpr():
    """Test that transforming plxpr works correctly."""

    def circuit(U):
        qml.QubitUnitary(U, 0)
        return qml.expval(qml.Z(0))

    U = qml.Rot(1.0, 2.0, 3.0, wires=0).matrix()
    args = (U,)
    jaxpr = jax.make_jaxpr(circuit)(*args)
    transformed_jaxpr = unitary_to_rot_plxpr_to_plxpr(jaxpr.jaxpr, jaxpr.consts, [], {}, *args)

    assert isinstance(transformed_jaxpr, jax.core.ClosedJaxpr)

    NUM_OF_DECOMP_EQNS = 39
    assert transformed_jaxpr.eqns[NUM_OF_DECOMP_EQNS + 1].primitive == qml.RZ._primitive
    assert transformed_jaxpr.eqns[NUM_OF_DECOMP_EQNS + 2].primitive == qml.RY._primitive
    assert transformed_jaxpr.eqns[NUM_OF_DECOMP_EQNS + 3].primitive == qml.RZ._primitive
    assert transformed_jaxpr.eqns[NUM_OF_DECOMP_EQNS + 4].primitive == qml.PauliZ._primitive
    assert (
        transformed_jaxpr.eqns[NUM_OF_DECOMP_EQNS + 5].primitive
        == qml.measurements.ExpectationMP._obs_primitive
    )
