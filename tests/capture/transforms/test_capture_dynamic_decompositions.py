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
# pylint:disable=protected-access,unused-argument, wrong-import-position, no-value-for-parameter, too-few-public-methods, wrong-import-order
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from functools import partial

from pennylane.capture import expand_plxpr_transforms
from pennylane.capture.primitives import cond_prim, for_loop_prim, qnode_prim, while_loop_prim
from pennylane.operation import Operation
from pennylane.transforms.decompose import DecomposeInterpreter

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class SimpleCustomOp(Operation):
    """Simple custom operation that contains a single gate in its decomposition"""

    num_wires = 1
    num_params = 0

    def _init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_plxpr_decomposition(wires):

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
    def compute_plxpr_decomposition(*args, **hyperparameters):

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
    def compute_plxpr_decomposition(*args, **hyperparameters):

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
    def compute_plxpr_decomposition(phi, wires):

        def true_fn(phi, wires):
            qml.RX(phi, wires=wires)

        def false_fn(phi, wires):
            qml.RY(phi, wires=wires)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi, wires)


class CustomOpForLoop(Operation):
    """Custom operation that contains a for loop in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_plxpr_decomposition(phi, wires):

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
    def compute_plxpr_decomposition(phi, wires):

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
    def compute_plxpr_decomposition(phi, wires):

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
                qml.RX(phi, wires)
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
    def compute_plxpr_decomposition(phi, wires):

        if phi > 0.5:
            qml.RX(phi, wires=wires)

        else:
            qml.RY(phi, wires=wires)


class TestDynamicDecomposeInterpreter:
    """Tests for the DynamicDecomposeInterpreter class"""

    def test_error_no_plxpr_decomposition(self):
        """Test that an error is raised if an operator does not have a plxpr decomposition."""

        with pytest.raises(qml.operation.DecompositionUndefinedError):
            qml.RX(0.1, 0).compute_plxpr_decomposition()

    def test_no_plxpr_decomposition(self):
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
        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    ############################
    ### QNode tests
    ############################

    @pytest.mark.parametrize("autograph", [True, False])
    def test_qnode_simple(self, autograph):
        """Test that a QNode with a custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=autograph)
        def circuit():
            qml.RY(0.1, wires=0)
            SimpleCustomOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = qml.capture.make_plxpr(circuit)()

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison():
            qml.RY(0.1, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.Z(0))

        assert qml.math.allclose(*result, circuit_comparison())

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [2, 3, 1, 0]])
    def test_multi_wire(self, wires, autograph):
        """Test that a QNode with a multi-wire custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=4), autograph=autograph)
        def circuit(x, wires):
            CustomOpMultiWire(x, wires=wires)
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        jaxpr = jax.make_jaxpr(circuit)(0.5, wires=wires)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.DoubleExcitation._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.CNOT._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[5].primitive == qml.RZ._primitive
        assert qfunc_jaxpr.eqns[6].primitive == qml.RX._primitive

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

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wires", [[0, 1, 2, 3], [2, 3, 1, 0]])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_const_hyperparams(self, wires, x, autograph):
        """Test that a QNode with a constant in the custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=4), autograph=autograph)
        def circuit(x, wires):
            CustomOpConstHyperparams(x, wires=wires)
            return qml.expval(qml.Z(0)), qml.probs(wires=1), qml.var(qml.Z(2)), qml.state()

        jaxpr = jax.make_jaxpr(circuit)(x, wires=wires)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[5].primitive == qml.RZ._primitive

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

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_cond(self, x, wire, autograph):
        """Test that a QNode with a conditional custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=autograph)
        def circuit(x, wire):
            CustomOpCond(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(x, wire=wire)

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
            def true_fn(x, wire):
                qml.RX(x, wires=wire)

            def false_fn(x, wire):
                qml.RY(x, wires=wire)

            qml.cond(x > 0.5, true_fn, false_fn)(x, wire)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wire", [0, 1])
    def test_qnode_for_loop(self, wire, autograph):
        """Test that a QNode with a for loop custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=autograph)
        def circuit(x, wire):
            CustomOpForLoop(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == for_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

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

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wire", [0, 1])
    def test_qnode_while_loop(self, wire, autograph):
        """Test that a QNode with a while loop custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=autograph)
        def circuit(x, wire):
            CustomOpWhileLoop(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(0.5, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == while_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=False)
        def circuit_comparison(x, wire):
            @qml.while_loop(lambda i: i < 3)
            def loop_fn(i):
                qml.RX(x, wires=wire)
                return i + 1

            _ = loop_fn(0)

            return qml.expval(qml.Z(wires=wire))

        assert qml.math.allclose(*result, circuit_comparison(0.5, wire))

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_nested_cond(self, x, wire, autograph):
        """Test that a QNode with a nested conditional custom operation is correctly decomposed."""

        @DecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=autograph)
        def circuit(x, wire):
            CustomOpNestedCond(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[1].primitive == cond_prim
        assert qfunc_jaxpr.eqns[1].params["jaxpr_branches"][0].eqns[0].primitive == for_loop_prim
        assert qfunc_jaxpr.eqns[1].params["jaxpr_branches"][1].eqns[0].primitive == while_loop_prim
        assert qfunc_jaxpr.eqns[2].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=False)
        def circuit_comparison(x, wire):
            def true_fn(x, wire):

                @qml.for_loop(0, 3, 1)
                def loop_rx(i, phi):
                    qml.RX(phi, wires=wire)
                    return jax.numpy.sin(phi)

                # pylint: disable=unused-variable
                loop_rx(x)

            def false_fn(x, wire):
                @qml.while_loop(lambda i: i < 3)
                def loop_fn(i):
                    qml.RX(x, wires=wire)
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
        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=True)
        def circuit(x, wire):
            CustomOpAutograph(x, wires=wire)
            return qml.expval(qml.Z(wires=wire))

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

        @qml.qnode(device=qml.device("default.qubit", wires=2), autograph=True)
        def circuit_comparison(x, wire):

            if x > 0.5:
                qml.RX(x, wires=wire)
            else:
                qml.RY(x, wires=wire)

            return qml.expval(qml.Z(wires=wire))

        # Autograph requires to capture the function first
        jaxpr_comparison = jax.make_jaxpr(circuit_comparison)(x, wire)
        result_comparison = jax.core.eval_jaxpr(
            jaxpr_comparison.jaxpr, jaxpr_comparison.consts, x, wire
        )

        assert qml.math.allclose(*result, *result_comparison)


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
        assert (
            transformed_jaxpr.eqns[1].primitive == qml.measurements.ProbabilityMP._wires_primitive
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

        assert transformed_jaxpr.eqns[0].primitive == cond_prim
        assert (
            transformed_jaxpr.eqns[1].primitive == qml.measurements.ProbabilityMP._wires_primitive
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
