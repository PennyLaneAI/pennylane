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
"""Unit tests for the ``SingleQubitFusionInterpreter`` class."""

# pylint:disable=wrong-import-position,protected-access
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.transforms.optimization.single_qubit_fusion import (
    SingleQubitFusionInterpreter,
    single_qubit_plxpr_to_plxpr,
)

from pennylane.transforms.optimization import single_qubit_fusion

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


def check_matrix_equivalence(matrix_expected, matrix_obtained, atol=1e-8):
    """Takes two matrices and checks if multiplying one by the conjugate
    transpose of the other gives the identity."""

    mat_product = qml.math.dot(qml.math.conj(qml.math.T(matrix_obtained)), matrix_expected)
    mat_product = mat_product / mat_product[0, 0]

    return qml.math.allclose(mat_product, qml.math.eye(matrix_expected.shape[0]), atol=atol)


def extract_abstract_operator_eqns(jaxpr):
    """Extracts all JAXPR equations that correspond to abstract operators."""
    abstract_op_eqns = []

    for eqn in jaxpr.eqns:

        primitive = eqn.primitive

        if getattr(primitive, "prim_type", "") == "operator":

            abstract_op_eqns.append(eqn)

    return abstract_op_eqns


class TestSingleQubitFusionInterpreter:
    """Unit tests for the SingleQubitFusionInterpreter class"""

    def test_single_qubit_full_fusion(self):
        """Test that a sequence of single-qubit gates all fuse."""

        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1

        expected_primitive = {qml.Rot._primitive}
        actual_primitives = {jaxpr.eqns[0].primitive}
        assert expected_primitive == actual_primitives

        assert qml.math.allclose(jaxpr.eqns[0].invars[0].val, -4.369330)
        assert qml.math.allclose(jaxpr.eqns[0].invars[1].val, 1.983815)
        assert qml.math.allclose(jaxpr.eqns[0].invars[2].val, -0.959211)
        assert qml.math.allclose(jaxpr.eqns[0].invars[3].val, 0)

    def test_single_qubit_partial_fusion_qnode(self):
        """Test that a sequence of single-qubit gates partially fuse."""

        @qml.qnode(device=qml.device("default.qubit", wires=1))
        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        transformed_circuit = SingleQubitFusionInterpreter()(circuit)

        # This circuit should be transformed to a single Rot(-4.37,1.98,-0.96) gate

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(qfunc_jaxpr.eqns) == 3

        expected_primitive = {
            qml.Rot._primitive,
            qml.PauliZ._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        }
        actual_primitives = {
            qfunc_jaxpr.eqns[0].primitive,
            qfunc_jaxpr.eqns[1].primitive,
            qfunc_jaxpr.eqns[2].primitive,
        }
        assert expected_primitive == actual_primitives

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.literals)

        with qml.capture.pause():
            # pylint: disable=not-callable
            expected_result = single_qubit_fusion(circuit)()

        assert qml.math.allclose(result, expected_result)
