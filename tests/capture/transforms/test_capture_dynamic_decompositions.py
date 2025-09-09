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
"""Unit tests for the ``DecomposeInterpreter`` class with dynamic decompositions."""
import numpy as np

# pylint:disable=protected-access,unused-argument, wrong-import-position, no-value-for-parameter, too-few-public-methods, wrong-import-order, too-many-arguments
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from functools import partial

from pennylane.capture import expand_plxpr_transforms, run_autograph
from pennylane.capture.primitives import cond_prim, for_loop_prim, qnode_prim, while_loop_prim
from pennylane.operation import Operation
from pennylane.transforms.decompose import DecomposeInterpreter

pytestmark = [pytest.mark.jax, pytest.mark.capture]


def check_jaxpr_eqns(qfunc_jaxpr_eqns, operations):
    """Assert that the primitives of the jaxpr equations match the provided operations."""

    assert len(qfunc_jaxpr_eqns) == len(operations)

    for eqn, op in zip(qfunc_jaxpr_eqns, operations):
        assert eqn.primitive == op._primitive


def get_jaxpr_eqns(jaxpr):
    """Get the equations of the JAXPR."""

    return jaxpr.eqns


def get_qnode_eqns(jaxpr):
    """Get the equations of the QNode from the JAXPR."""

    assert jaxpr.eqns[0].primitive == qnode_prim

    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

    return qfunc_jaxpr.eqns


def get_eqns_cond_branches(jaxpr_cond, false_branch=True):
    """Get the equations of the true and false branches of the cond primitive."""

    assert jaxpr_cond.primitive == cond_prim

    true_branch_eqns = jaxpr_cond.params["jaxpr_branches"][0].eqns

    false_branch_eqns = jaxpr_cond.params["jaxpr_branches"][1].eqns if false_branch else []

    return true_branch_eqns, false_branch_eqns


def get_eqns_for_loop(jaxpr_for_loop):
    """Get the equations of the body of the for loop primitive."""

    assert jaxpr_for_loop.primitive == for_loop_prim

    return jaxpr_for_loop.params["jaxpr_body_fn"].eqns


def get_eqns_while_loop(jaxpr_while_loop):
    """Get the equations of the body of the while loop primitive."""

    assert jaxpr_while_loop.primitive == while_loop_prim

    return jaxpr_while_loop.params["jaxpr_body_fn"].eqns


class SimpleCustomOp(Operation):
    """Simple custom operation that contains a single gate in its decomposition"""

    num_wires = 1
    num_params = 0

    def _init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        return [qml.Hadamard(wires=wires), qml.Hadamard(wires=wires)]

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(wires):
        qml.Hadamard(wires=wires)
        qml.Hadamard(wires=wires)


class SimpleCustomOpReturn(Operation):
    """Simple custom operation that contains a single gate in its decomposition"""

    num_wires = 1
    num_params = 0

    def _init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        raise NotImplementedError

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(wires):
        return qml.Hadamard(wires=wires)


const = jax.numpy.array(0.1)


class CustomOpConstHyperparams(Operation):
    """Custom operation that contains constants and hyperparameters in its decomposition"""

    num_wires = 4
    num_params = 1

    def __init__(self, phi, wires, id=None):

        self._hyperparameters = {
            "key": const,
            "CNOT": qml.CNOT,
            "RX": qml.RX,
            "phi": phi,
        }

        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        raise NotImplementedError

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(*args, **hyperparameters):

        phi = args[0]
        wires = args[1:]

        hyperparameters["CNOT"](wires=[wires[0], wires[1]])
        hyperparameters["RX"](phi, wires=wires[2])
        hyperparameters["RX"](hyperparameters["key"], wires=wires[0])
        hyperparameters["RX"](const, wires=wires[3])

        qml.RY(hyperparameters["key"], wires[0])
        qml.RZ(hyperparameters["phi"], wires[2])


class CustomOpMultiWire(Operation):
    """Custom operation that acts on multiple wires"""

    num_wires = 4
    num_params = 1

    def __init__(self, phi, wires, id=None):

        self.hyperparameters["key_1"] = 0.1
        self.hyperparameters["key_2"] = 0.2

        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*args, **hyperparameters):
        raise NotImplementedError

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(*args, **hyperparameters):

        phi = args[0]
        wires = args[1:]

        qml.CNOT([wires[0], wires[1]])
        qml.DoubleExcitation(phi, wires)
        qml.CNOT([wires[0], wires[1]])
        qml.RX(hyperparameters["key_1"], wires[0])
        qml.RY(phi, wires[1])
        qml.RZ(phi, wires[2])
        qml.RX(hyperparameters["key_2"], wires[3])


class CustomOpCond(Operation):
    """Custom operation that contains a conditional in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        def true_fn(phi, wires):
            qml.RX(phi, wires=wires)

        def false_fn(phi, wires):
            qml.RY(phi, wires=wires)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi, wires)


class CustomOpCondNoFalseBranch(Operation):
    """Custom operation that contains a conditional in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        def true_fn(phi, wires):
            qml.RX(phi, wires=wires)

        qml.cond(phi > 0.5, true_fn)(phi, wires)


class CustomOpForLoop(Operation):
    """Custom operation that contains a for loop in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        @qml.for_loop(0, 3, 1)
        def loop_rx(i, phi):
            qml.RX(phi, wires)
            return jax.numpy.sin(phi)

        # pylint: disable=unused-variable
        loop_rx(phi)


class CustomOpWhileLoop(Operation):
    """Custom operation that contains a while loop in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        def while_f(i):
            return i < 3

        @qml.while_loop(while_f)
        def loop_fn(i):
            qml.RX(phi, wires)
            return i + 1

        _ = loop_fn(0)


class CustomOpNestedCond(Operation):
    """Custom operation that contains a nested conditional in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        def true_fn(phi, wires):

            @qml.for_loop(0, 3, 1)
            def loop_rx(i, phi):
                qml.RX(phi, wires)
                return jax.numpy.sin(phi)

            # pylint: disable=unused-variable
            loop_rx(phi)

        def false_fn(phi, wires):

            def while_f(i):
                return i < 3

            @qml.while_loop(while_f)
            def loop_fn(i):
                qml.RZ(phi, wires)

                return i + 1

            _ = loop_fn(0)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi, wires)

        qml.RX(phi, wires=wires)


class CustomOpAutograph(Operation):
    """Custom operation that contains a nested conditional in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        if phi > 0.5:
            qml.RX(phi, wires=wires)

        else:
            qml.RY(phi, wires=wires)


class CustomOpNestedOp(Operation):
    """Custom operation that contains a nested decomposition in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [qml.RX(phi, wires=wires), SimpleCustomOp(wires=wires)]

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):
        qml.RX(phi, wires=wires)
        SimpleCustomOp(wires=wires)


class CustomOpNestedOpControlFlow(Operation):
    """Custom operation that contains a nested decomposition in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        return [qml.S(wires=wires)]

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_qfunc_decomposition(phi, wires):

        qml.Rot(0.1, 0.2, 0.3, wires=wires)
        CustomOpNestedOp(phi, wires)

        def true_fn(phi, wires):

            @qml.for_loop(0, 3, 1)
            def loop_rx(i, phi):
                CustomOpNestedOp(phi, wires)
                return jax.numpy.sin(phi)

            # pylint: disable=unused-variable
            loop_rx(phi)

        def false_fn(phi, wires):

            def while_f(i):
                return i < 3

            @qml.while_loop(while_f)
            def loop_fn(i):
                SimpleCustomOp(wires)
                return i + 1

            _ = loop_fn(0)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi, wires)


class CustomOpNoPlxprDecomposition(Operation):
    """Custom operation that does not have a plxpr decomposition and returns an operator with a plxpr decomposition in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        return np.array([[1, 0], [0, 1]])

    @staticmethod
    def compute_decomposition(phi, wires):
        return [CustomOpNestedOpControlFlow(phi, wires)]


# pylint: disable=too-many-public-methods
class TestDynamicDecomposeInterpreter:
    """Tests for the DynamicDecomposeInterpreter class"""

    def test_error_no_qfunc_decomposition(self):
        """Test that an error is raised if an operator does not have a plxpr decomposition."""

        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.RX(0.1, 0).compute_qfunc_decomposition()

    def test_no_qfunc_decomposition(self):
        """Test that a function with a custom operation that does not have a plxpr decomposition is not decomposed."""

        @DecomposeInterpreter()
        def f(x):
            qml.RY(x, wires=0)

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == qml.RY._primitive

    def test_function_simple(self):
        """Test that a function with a custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        def f():
            qml.RY(0.1, wires=0)
            SimpleCustomOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_simple_return(self):
        """Test that a function with a custom operation that returns a value is correctly decomposed."""

        @DecomposeInterpreter()
        def f():
            SimpleCustomOpReturn(wires=0)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == qml.Hadamard._primitive

    ############################
    ### QNode tests
    ############################

    def test_qnode_simple(self):
        """Test that a QNode with a custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit():
            qml.RY(0.1, wires=0)
            SimpleCustomOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = qml.capture.make_plxpr(circuit)()
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)
        check_jaxpr_eqns(qfunc_jaxpr_eqns[0:3], [qml.RY, qml.Hadamard, qml.Hadamard])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison():
            qml.RY(0.1, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.Z(0))

        assert qml.math.allclose(*result, circuit_comparison())

    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [2, 3, 1, 0]])
    def test_multi_wire(self, wires):
        """Test that a QNode with a multi-wire custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=4))
        def circuit(x, wires):
            CustomOpMultiWire(x, wires=wires)
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        jaxpr = jax.make_jaxpr(circuit)(0.5, wires=wires)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)
        check_jaxpr_eqns(
            qfunc_jaxpr_eqns[0:7],
            [qml.CNOT, qml.DoubleExcitation, qml.CNOT, qml.RX, qml.RY, qml.RZ, qml.RX],
        )

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5, *wires)

        @qml.qnode(device=qml.device("default.qubit", wires=4))
        def circuit_comparison(x, wires):
            qml.CNOT([wires[0], wires[1]])
            qml.DoubleExcitation(x, wires)
            qml.CNOT([wires[0], wires[1]])
            qml.RX(0.1, wires=wires[0])
            qml.RY(x, wires=wires[1])
            qml.RZ(x, wires=wires[2])
            qml.RX(0.2, wires=wires[3])
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        comparison_result = circuit_comparison(0.5, wires)
        for res, comp in zip(result, comparison_result):
            assert qml.math.allclose(res, comp)

    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [2, 3, 1, 0]])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_const_hyperparams(self, wires, x):
        """Test that a QNode with a constant in the custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=4))
        def circuit(x, wires):
            CustomOpConstHyperparams(x, wires=wires)
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        jaxpr = jax.make_jaxpr(circuit)(x, wires=wires)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)
        check_jaxpr_eqns(
            qfunc_jaxpr_eqns[0:6],
            [qml.CNOT, qml.RX, qml.RX, qml.RX, qml.RY, qml.RZ],
        )

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, *wires)

        @qml.qnode(device=qml.device("default.qubit", wires=4))
        def circuit_comparison(x, wires):
            qml.CNOT([wires[0], wires[1]])
            qml.RX(x, wires=wires[2])
            qml.RX(0.1, wires=wires[0])
            qml.RX(0.1, wires=wires[3])
            qml.RY(0.1, wires=wires[0])
            qml.RZ(x, wires=wires[2])
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        comparison_result = circuit_comparison(x, wires)
        for res, comp in zip(result, comparison_result):
            assert qml.math.allclose(res, comp)

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_cond(self, x, wire):
        """Test that a QNode with a conditional custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpCond(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(x, wire=wire)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)

        true_branch_eqns, false_branch_eqns = get_eqns_cond_branches(qfunc_jaxpr_eqns[1])
        check_jaxpr_eqns(true_branch_eqns, [qml.RX])
        check_jaxpr_eqns(false_branch_eqns, [qml.RY])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            def true_fn(x, wire):
                qml.RX(x, wires=wire)

            def false_fn(x, wire):
                qml.RY(x, wires=wire)

            qml.cond(x > 0.5, true_fn, false_fn)(x, wire)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_cond_no_false_branch(self, x, wire):
        """Test that a QNode with a conditional custom operation that does not have a false branch is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpCondNoFalseBranch(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)

        true_branch_eqns, _ = get_eqns_cond_branches(qfunc_jaxpr_eqns[1], false_branch=False)
        check_jaxpr_eqns(true_branch_eqns, [qml.RX])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            def true_fn(x, wire):
                qml.RX(x, wires=wire)

            qml.cond(x > 0.5, true_fn)(x, wire)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    def test_qnode_for_loop(self, wire):
        """Test that a QNode with a for loop custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpForLoop(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)

        for_loop_eqns = get_eqns_for_loop(qfunc_jaxpr_eqns[0])
        for_loop_eqns = [eqn for eqn in for_loop_eqns if eqn.primitive != jax.lax.sin_p]

        check_jaxpr_eqns(for_loop_eqns, [qml.RX])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            @qml.for_loop(0, 3, 1)
            def loop_rx(i, phi):
                qml.RX(phi, wires=wire)
                return jax.numpy.sin(phi)

            # pylint: disable=unused-variable
            loop_rx(x)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(0.5, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    def test_qnode_while_loop(self, wire):
        """Test that a QNode with a while loop custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpWhileLoop(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)

        while_loop_eqns = get_eqns_while_loop(qfunc_jaxpr_eqns[0])
        check_jaxpr_eqns([while_loop_eqns[0]], [qml.RX])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):

            def while_f(i):
                return i < 3

            @qml.while_loop(while_f)
            def loop_fn(i):
                qml.RX(x, wires=wire)
                return i + 1

            _ = loop_fn(0)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(0.5, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_nested_cond(self, x, wire):
        """Test that a QNode with a nested conditional custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpNestedCond(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)
        qfunc_jaxpr_eqns = get_qnode_eqns(jaxpr)

        cond_eqns = get_eqns_cond_branches(qfunc_jaxpr_eqns[1])
        for_loop_eqns = get_eqns_for_loop(cond_eqns[0][0])
        while_loop_eqns = get_eqns_while_loop(cond_eqns[1][0])

        for_loop_eqns = [eqn for eqn in for_loop_eqns if eqn.primitive != jax.lax.sin_p]
        while_loop_eqns = [eqn for eqn in while_loop_eqns if eqn.primitive != jax.lax.add_p]

        check_jaxpr_eqns(for_loop_eqns, [qml.RX])
        check_jaxpr_eqns(while_loop_eqns, [qml.RZ])
        check_jaxpr_eqns([qfunc_jaxpr_eqns[2]], [qml.RX])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            def true_fn(x, wire):

                @qml.for_loop(0, 3, 1)
                def loop_rx(i, phi):
                    qml.RX(phi, wires=wire)
                    return jax.numpy.sin(phi)

                # pylint: disable=unused-variable
                loop_rx(x)

            def false_fn(x, wire):

                def while_f(i):
                    return i < 3

                @qml.while_loop(while_f)
                def loop_fn(i):
                    qml.RZ(x, wires=wire)
                    return i + 1

                _ = loop_fn(0)

            qml.cond(x > 0.5, true_fn, false_fn)(x, wire)
            qml.RX(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_autograph(self, x, wire):
        """Test that a QNode with a nested conditional custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpAutograph(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(circuit)(x, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[1].primitive == cond_prim
        assert (
            qfunc_jaxpr.eqns[1].params["jaxpr_branches"][0].eqns[0].primitive == qml.RX._primitive
        )
        assert (
            qfunc_jaxpr.eqns[1].params["jaxpr_branches"][1].eqns[0].primitive == qml.RY._primitive
        )
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):

            if x > 0.5:
                qml.RX(x, wires=wire)
            else:
                qml.RY(x, wires=wire)

            return qml.expval(qml.Z(wires=wire))

        # Autograph requires to capture the function first
        circuit_comparison = run_autograph(circuit_comparison)
        jaxpr_comparison = jax.make_jaxpr(circuit_comparison)(x, wire)
        result_comparison = jax.core.eval_jaxpr(
            jaxpr_comparison.jaxpr, jaxpr_comparison.consts, x, wire
        )

        assert qml.math.allclose(*result, *result_comparison)

    #################################
    ### Nested decomposition tests
    #################################

    @pytest.mark.parametrize(
        "max_expansion, expected_ops",
        [
            (0, [CustomOpNestedOp]),
            (1, [qml.RX, SimpleCustomOp]),
            (2, [qml.RX, qml.Hadamard, qml.Hadamard]),
            (None, [qml.RX, qml.Hadamard, qml.Hadamard]),
        ],
    )
    def test_qnode_nested_decomp_max_exp(self, max_expansion, expected_ops):
        """Test that a QNode with a nested decomposition custom operation is correctly decomposed."""

        @DecomposeInterpreter(max_expansion=max_expansion)
        def circuit(x, wire):
            CustomOpNestedOp(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        check_jaxpr_eqns(jaxpr_eqns[0 : len(expected_ops)], expected_ops)

    @pytest.mark.parametrize(
        "gate_set, expected_ops",
        [
            (
                [qml.RX, qml.Hadamard],
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                [qml.RX, qml.Hadamard, qml.Hadamard],
            ),
            (
                [qml.RX, qml.RY, qml.RZ, qml.CNOT],
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                # Hadamard -> RZ, RX, RZ
                [qml.RX, qml.RZ, qml.RX, qml.RZ, qml.RZ, qml.RX, qml.RZ],
            ),
        ],
    )
    def test_nested_decomp_gate_set(self, gate_set, expected_ops):
        """Test that a QNode with a nested decomposition custom operation is correctly decomposed using a custom gate set."""

        @DecomposeInterpreter(gate_set=gate_set)
        def circuit(x, wire):
            CustomOpNestedOp(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        check_jaxpr_eqns(jaxpr_eqns[0 : len(expected_ops)], expected_ops)

    @pytest.mark.parametrize(
        "max_expansion, expected_ops, expected_ops_for_loop, expected_ops_while_loop",
        [
            (1, [qml.Rot, CustomOpNestedOp], [CustomOpNestedOp], [SimpleCustomOp]),
            (
                2,
                [qml.Rot, qml.RX, SimpleCustomOp],
                [qml.RX, SimpleCustomOp],
                [qml.Hadamard, qml.Hadamard],
            ),
            (
                3,
                [qml.Rot, qml.RX, qml.Hadamard, qml.Hadamard],
                [qml.RX, qml.Hadamard, qml.Hadamard],
                [qml.Hadamard, qml.Hadamard],
            ),
            (
                None,
                [qml.Rot, qml.RX, qml.Hadamard, qml.Hadamard],
                [qml.RX, qml.Hadamard, qml.Hadamard],
                [qml.Hadamard, qml.Hadamard],
            ),
        ],
    )
    def test_nested_decomp_control_flow_max_exp(
        self, max_expansion, expected_ops, expected_ops_for_loop, expected_ops_while_loop
    ):
        """Test that a nested decomposition custom operation that contains control flow is correctly decomposed."""

        @DecomposeInterpreter(max_expansion=max_expansion)
        def circuit(x, wire):
            CustomOpNestedOpControlFlow(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        ops_before_cond = len(expected_ops)
        check_jaxpr_eqns(jaxpr_eqns[0:ops_before_cond], expected_ops)

        # The + 1 is for the operation that determines the branches of the cond primitive
        cond_eqns = get_eqns_cond_branches(jaxpr_eqns[ops_before_cond + 1])
        for_loop_eqns = get_eqns_for_loop(cond_eqns[0][0])
        while_loop_eqns = get_eqns_while_loop(cond_eqns[1][0])

        for_loop_eqns = [eqn for eqn in for_loop_eqns if eqn.primitive != jax.lax.sin_p]
        while_loop_eqns = [eqn for eqn in while_loop_eqns if eqn.primitive != jax.lax.add_p]

        check_jaxpr_eqns(for_loop_eqns, expected_ops_for_loop)
        check_jaxpr_eqns(while_loop_eqns, expected_ops_while_loop)

    @pytest.mark.parametrize(
        "gate_set, expected_ops, expected_ops_for_loop, expected_ops_while_loop",
        [
            (
                [qml.RX, qml.RY, qml.RZ, CustomOpNestedOp, SimpleCustomOp],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                [qml.RZ, qml.RY, qml.RZ, CustomOpNestedOp],
                # CustomOpNestedOp is in the for loop of the true branch
                [CustomOpNestedOp],
                # SimpleCustomOp is in the while loop of the false branch
                [SimpleCustomOp],
            ),
            (
                [qml.RX, qml.RY, qml.RZ, SimpleCustomOp],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                # CustomOpNestedOp -> RX, SimpleCustomOp
                [qml.RZ, qml.RY, qml.RZ, qml.RX, SimpleCustomOp],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                [qml.RX, SimpleCustomOp],
                # SimpleCustomOp is in the while loop of the false branch
                [SimpleCustomOp],
            ),
            (
                [qml.RX, qml.RY, qml.RZ, qml.Hadamard],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                [qml.RZ, qml.RY, qml.RZ, qml.RX, qml.Hadamard, qml.Hadamard],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                [qml.RX, qml.Hadamard, qml.Hadamard],
                # SimpleCustomOp is in the while loop of the false branch
                # SimpleCustomOp -> Hadamard, Hadamard
                [qml.Hadamard, qml.Hadamard],
            ),
            (
                [qml.RX, qml.RY, qml.RZ],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                # Hadamard -> RZ, RX, RZ
                [qml.RZ, qml.RY, qml.RZ, qml.RX, qml.RZ, qml.RX, qml.RZ, qml.RZ, qml.RX, qml.RZ],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                # Hadamard -> RZ, RX, RZ
                [qml.RX, qml.RZ, qml.RX, qml.RZ, qml.RZ, qml.RX, qml.RZ],
                # SimpleCustomOp is in the while loop of the false branch
                # SimpleCustomOp -> Hadamard, Hadamard
                # Hadamard -> RZ, RX, RZ
                [qml.RZ, qml.RX, qml.RZ, qml.RZ, qml.RX, qml.RZ],
            ),
        ],
    )
    def test_nested_decomp_control_flow_gate_set(
        self, gate_set, expected_ops, expected_ops_for_loop, expected_ops_while_loop
    ):
        """Test that a nested decomposition custom operation that contains control flow is correctly decomposed using a custom gate set."""

        @DecomposeInterpreter(gate_set=gate_set)
        def circuit(x, wire):
            CustomOpNestedOpControlFlow(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        ops_before_cond = len(expected_ops)
        check_jaxpr_eqns(jaxpr_eqns[0:ops_before_cond], expected_ops)

        # The + 1 is for the operation that determines the branches of the cond primitive
        cond_eqns = get_eqns_cond_branches(jaxpr_eqns[ops_before_cond + 1])
        for_loop_eqns = get_eqns_for_loop(cond_eqns[0][0])
        while_loop_eqns = get_eqns_while_loop(cond_eqns[1][0])

        for_loop_eqns = [eqn for eqn in for_loop_eqns if eqn.primitive != jax.lax.sin_p]
        while_loop_eqns = [eqn for eqn in while_loop_eqns if eqn.primitive != jax.lax.add_p]

        check_jaxpr_eqns(for_loop_eqns, expected_ops_for_loop)
        check_jaxpr_eqns(while_loop_eqns, expected_ops_while_loop)

    @pytest.mark.parametrize(
        "max_expansion, gate_set, expected_ops, expected_ops_for_loop, expected_ops_while_loop",
        [
            (
                1,
                [qml.RX, qml.RY, qml.RZ, qml.CNOT],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                [qml.Rot, CustomOpNestedOp],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                [CustomOpNestedOp],
                # SimpleCustomOp is in the while loop of the false branch
                [SimpleCustomOp],
            ),
            (
                2,
                [qml.RX, qml.RY, qml.RZ, CustomOpNestedOp],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ, CustomOpNestedOp is in the gate set
                [qml.RZ, qml.RY, qml.RZ, CustomOpNestedOp],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                # SimpleCustomOp -> Hadamard, Hadamard
                [CustomOpNestedOp],
                # SimpleCustomOp is in the while loop of the false branch
                # SimpleCustomOp -> Hadamard, Hadamard
                # Hadamard -> RZ, RX, RZ
                [qml.Hadamard, qml.Hadamard],
            ),
            (
                3,
                [qml.RX, qml.RY, qml.RZ, qml.Rot, SimpleCustomOp],
                # CustomOpNestedOpControlFlow -> Rot, CustomOpNestedOp (before cond)
                # Rot -> qml.RZ, qml.RY, qml.RZ
                # CustomOpNestedOp -> RX, SimpleCustomOp
                [qml.Rot, qml.RX, SimpleCustomOp],
                # CustomOpNestedOp is in the for loop of the true branch
                # CustomOpNestedOp -> RX, SimpleCustomOp
                [qml.RX, SimpleCustomOp],
                # SimpleCustomOp is in the while loop of the false branch
                [SimpleCustomOp],
            ),
        ],
    )
    def test_nested_decomp_control_flow_max_exp_gate_set(
        self, max_expansion, gate_set, expected_ops, expected_ops_for_loop, expected_ops_while_loop
    ):
        """Test that a nested decomposition custom operation that contains control flow is correctly decomposed using a gate set and max expansion."""

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        def circuit(x, wire):
            CustomOpNestedOpControlFlow(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        ops_before_cond = len(expected_ops)
        check_jaxpr_eqns(jaxpr_eqns[0:ops_before_cond], expected_ops)

        # The + 1 is for the operation that determines the branches of the cond primitive
        cond_eqns = get_eqns_cond_branches(jaxpr_eqns[ops_before_cond + 1])
        for_loop_eqns = get_eqns_for_loop(cond_eqns[0][0])
        while_loop_eqns = get_eqns_while_loop(cond_eqns[1][0])

        for_loop_eqns = [eqn for eqn in for_loop_eqns if eqn.primitive != jax.lax.sin_p]
        while_loop_eqns = [eqn for eqn in while_loop_eqns if eqn.primitive != jax.lax.add_p]

        check_jaxpr_eqns(for_loop_eqns, expected_ops_for_loop)
        check_jaxpr_eqns(while_loop_eqns, expected_ops_while_loop)

    @pytest.mark.parametrize(
        "max_expansion, expected_ops",
        [
            # No expansion is performed
            (0, [CustomOpNoPlxprDecomposition]),
            # the `compute_decomposition` of CustomOpNoPlxprDecomposition is called, because this method does not have a plxpr decomposition
            (1, [CustomOpNestedOpControlFlow]),
            # the `compute_decomposition` of CustomOpNestedOpControlFlow is called, because (even though this operator has a plxpr decomposition),
            # it was called in the `compute_decomposition` of CustomOpNoPlxprDecomposition. This is a necessary limitation of the current implementation.
            (2, [qml.S]),
            (None, [qml.S]),
        ],
    )
    def test_nested_decomp_no_plxpr_decomp_max_exp(self, max_expansion, expected_ops):
        """Test that a nested decomposition custom operation that contains an operator with no plxpr decomposition is correctly decomposed."""

        @DecomposeInterpreter(max_expansion=max_expansion)
        def circuit(x, wire):
            CustomOpNoPlxprDecomposition(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        check_jaxpr_eqns(jaxpr_eqns[0 : len(expected_ops)], expected_ops)

    @pytest.mark.parametrize(
        "gate_set, expected_ops",
        [
            ([CustomOpNoPlxprDecomposition], [CustomOpNoPlxprDecomposition]),
            ([CustomOpNestedOpControlFlow], [CustomOpNestedOpControlFlow]),
            # We expected the same decomposition of the single qml.S gate.
            # Notice that the `compute_decomposition` of CustomOpNoPlxprDecomposition is called and, as a consequence,
            # the `compute_decomposition` of CustomOpNestedOpControlFlow is called (instead of its plxpr decomposition).
            #
            # CustomOpNoPlxprDecomposition -> CustomOpNestedOpControlFlow -> qml.S -> qml.PhaseShift
            ([qml.RX, qml.RY, qml.RZ, qml.PhaseShift], [qml.PhaseShift]),
            # CustomOpNoPlxprDecomposition -> CustomOpNestedOpControlFlow -> qml.S -> ... -> qml.RZ
            ([qml.RX, qml.RY, qml.RZ, qml.CNOT], [qml.RZ]),
        ],
    )
    def test_nested_decomp_no_qfunc_decomposition_gate_set(self, gate_set, expected_ops):
        """Test that a nested decomposition custom operation that contains an operator with no plxpr decomposition is correctly decomposed using a custom gate set."""

        @DecomposeInterpreter(gate_set=gate_set)
        def circuit(x, wire):
            CustomOpNoPlxprDecomposition(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        check_jaxpr_eqns(jaxpr_eqns[0 : len(expected_ops)], expected_ops)

    @pytest.mark.parametrize(
        "max_expansion, gate_set, expected_ops",
        [
            (0, [CustomOpNoPlxprDecomposition], [CustomOpNoPlxprDecomposition]),
            (1, [CustomOpNoPlxprDecomposition], [CustomOpNoPlxprDecomposition]),
            (2, [CustomOpNoPlxprDecomposition], [CustomOpNoPlxprDecomposition]),
            (0, [CustomOpNestedOpControlFlow], [CustomOpNoPlxprDecomposition]),
            (1, [CustomOpNestedOpControlFlow], [CustomOpNestedOpControlFlow]),
            (2, [CustomOpNestedOpControlFlow], [CustomOpNestedOpControlFlow]),
            (0, [qml.RX, qml.RY, qml.RZ, qml.S], [CustomOpNoPlxprDecomposition]),
            (1, [qml.RX, qml.RY, qml.RZ, qml.S], [CustomOpNestedOpControlFlow]),
            (2, [qml.RX, qml.RY, qml.RZ, qml.S], [qml.S]),
            (2, [qml.RX, qml.RY, qml.RZ], [qml.S]),
            (None, [qml.RX, qml.RY, qml.RZ], [qml.RZ]),
        ],
    )
    def test_nested_decomp_no_qfunc_decomposition_max_exp_gate_set(
        self, max_expansion, gate_set, expected_ops
    ):
        """Test that a custom operation that contains an operator with no plxpr decomposition is correctly decomposed using a custom gate set and max_expansion."""

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        def circuit(x, wire):
            CustomOpNoPlxprDecomposition(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire=0)
        jaxpr_eqns = get_jaxpr_eqns(jaxpr)

        check_jaxpr_eqns(jaxpr_eqns[0 : len(expected_ops)], expected_ops)


class TestExpandPlxprTransformsDynamicDecompositions:
    """Unit tests for ``expand_plxpr_transforms`` with dynamic decompositions."""

    def test_expand_plxpr_transforms_simple(self):

        @partial(qml.transforms.decompose)
        def circuit():
            SimpleCustomOp(wires=0)
            return qml.probs(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qml.transforms.decompose._primitive

        transformed_f = expand_plxpr_transforms(circuit)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()

        assert transformed_jaxpr.eqns[0].primitive == qml.Hadamard._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert (
            transformed_jaxpr.eqns[2].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )

    def test_expand_plxpr_transforms_cond(self):
        @partial(qml.transforms.decompose)
        def circuit():
            CustomOpCond(0.5, wires=0)
            return qml.probs(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qml.transforms.decompose._primitive

        transformed_f = expand_plxpr_transforms(circuit)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()

        assert transformed_jaxpr.eqns[1].primitive == cond_prim
        assert (
            transformed_jaxpr.eqns[2].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )

    def test_expand_plxpr_transforms_for_loop(self):
        @partial(qml.transforms.decompose)
        def circuit():
            CustomOpForLoop(0.5, wires=0)
            return qml.probs(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qml.transforms.decompose._primitive

        transformed_f = expand_plxpr_transforms(circuit)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()

        assert transformed_jaxpr.eqns[0].primitive == for_loop_prim
        assert (
            transformed_jaxpr.eqns[1].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )

    def test_expand_plxpr_transforms_while_loop(self):
        @partial(qml.transforms.decompose)
        def circuit():
            CustomOpWhileLoop(0.5, wires=0)
            return qml.probs(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qml.transforms.decompose._primitive

        transformed_f = expand_plxpr_transforms(circuit)
        transformed_jaxpr = jax.make_jaxpr(transformed_f)()

        assert transformed_jaxpr.eqns[0].primitive == while_loop_prim
        assert (
            transformed_jaxpr.eqns[1].primitive == qml.measurements.ProbabilityMP._wires_primitive
        )
