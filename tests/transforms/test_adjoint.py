# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import numpy as np

import pennylane as qml
from pennylane.transforms.adjoint import adjoint


def test_adjoint_on_function():
    """Test that adjoint works when applied to a function"""

    dev = qml.device("default.qubit", wires=1)

    def my_op():
        qml.RX(0.123, wires=0)
        qml.RY(2.32, wires=0)
        qml.RZ(1.95, wires=0)

    @qml.qnode(dev)
    def my_circuit():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=0)
        my_op()
        adjoint(my_op)()
        qml.PauliY(wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([1.0j, 0.0]), atol=1e-6, rtol=1e-6)


def test_adjoint_directly_on_op():
    """Test that adjoint works when directly applyed to an op"""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(qml.RX)(np.pi / 4.0, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([0.92388, 0.382683j]), atol=1e-6, rtol=1e-6)


def test_barrier_adjoint():
    """Check that the adjoint for the Barrier is working"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(qml.Barrier)(wires=0)
        return qml.state()

    assert my_circuit()[0] == 1.0


def test_wirecut_adjoint():
    """Check that the adjoint for the WireCut is working"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(qml.WireCut)(wires=0)
        return qml.state()

    assert np.isclose(my_circuit()[0], 1.0)


def test_identity_adjoint():
    """Check that the adjoint for Identity is working"""
    dev = qml.device("default.qubit", wires=2, shots=100)

    @qml.qnode(dev)
    def circuit():
        identity()
        qml.adjoint(identity)()
        return qml.state()

    def identity():
        qml.PauliX(wires=0)
        qml.Identity(0)
        qml.CNOT(wires=[0, 1])

    assert circuit()[0] == 1.0

    queue = circuit.tape.queue

    assert queue[1].name == "Identity"
    assert queue[4].name == "Identity"


def test_nested_adjoint():
    """Test that adjoint works when nested with other adjoints"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(adjoint(qml.RX))(np.pi / 4.0, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([0.92388, -0.382683j]), atol=1e-6, rtol=1e-6)


def test_nested_adjoint_on_function():
    """Test that adjoint works when nested with other adjoints"""

    def my_op():
        qml.RX(0.123, wires=0)
        qml.RY(2.32, wires=0)
        qml.RZ(1.95, wires=0)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(my_op)()
        qml.Hadamard(wires=0)
        adjoint(adjoint(my_op))()
        return qml.state()

    np.testing.assert_allclose(
        my_circuit(), np.array([-0.995707, 0.068644 + 6.209710e-02j]), atol=1e-6, rtol=1e-6
    )


with qml.tape.JacobianTape() as tape:
    qml.PauliX(0)
    qml.Hadamard(1)

noncallable_objects = [
    qml.RX(0.2, wires=0),
    qml.AngleEmbedding(list(range(2)), wires=range(2)),
    [qml.Hadamard(1), qml.RX(-0.2, wires=1)],
    tape,
]


@pytest.mark.parametrize("obj", noncallable_objects)
def test_error_adjoint_on_noncallable(obj):
    """Test that an error is raised if qml.adjoint is applied to an object that
    is not callable, as it silently does not have any effect on those."""
    with pytest.raises(ValueError, match=f"{type(obj)} is not callable."):
        adjoint(obj)


class TestOutsideOfQueuing:
    """Test that operations and templates work with the adjoint transform when
    created outside of a queuing context"""

    non_param_ops = [(qml.S, 0), (qml.PauliZ, 3), (qml.CNOT, [32, 3])]

    @pytest.mark.parametrize("op,wires", non_param_ops)
    def test_single_op_non_param_adjoint(self, op, wires):
        """Test that the adjoint correctly inverts non-parametrized
        operations"""
        op_adjoint = adjoint(op)(wires=wires)
        expected = op(wires=wires).adjoint()

        assert type(op_adjoint) == type(expected)
        assert op_adjoint.wires == expected.wires

    param_ops = [(qml.RX, [0.123], 0), (qml.Rot, [0.1, 0.2, 0.3], [1]), (qml.CRY, [0.1], [1, 4])]

    @pytest.mark.parametrize("op,par,wires", param_ops)
    def test_single_op_param_adjoint(self, op, par, wires):
        """Test that the adjoint correctly inverts operations with a single
        parameter"""
        param_op_adjoint = adjoint(op)(*par, wires=wires)
        expected = op(*par, wires=wires).adjoint()

        assert type(param_op_adjoint) == type(expected)
        assert param_op_adjoint.parameters == expected.parameters
        assert param_op_adjoint.wires == expected.wires

    template_ops = [
        (qml.templates.AngleEmbedding, [np.ones((1))], [2, 3]),
        (qml.templates.StronglyEntanglingLayers, [np.ones((1, 2, 3))], [2, 3]),
    ]

    @pytest.mark.parametrize("template, par, wires", template_ops)
    def test_templates_adjoint(self, template, par, wires):
        """Test that the adjoint correctly inverts templates"""
        res = adjoint(template)(*par, wires=wires)
        result = res if hasattr(res, "__iter__") else [res]  # handle single operation case
        expected_ops = template(*par, wires=wires)

        expected_ops = expected_ops.expand().operations
        for o1, o2 in zip(result, reversed(expected_ops)):
            o2 = o2.adjoint()
            assert type(o1) == type(o2)
            assert o1.parameters == o2.parameters
            assert o1.wires == o2.wires

    def test_cv_template_adjoint(self):
        """Test that the adjoint correctly inverts CV templates"""
        template, par, wires = qml.templates.Interferometer, [[1], [0.3], [0.2, 0.3]], [2, 3]
        result = adjoint(template)(*par, wires=wires).expand().operations
        expected_ops = template(*par, wires=wires).expand().operations

        for o1, o2 in zip(result, reversed(expected_ops)):
            o2 = o2.adjoint()
            assert type(o1) == type(o2)
            assert o1.parameters == o2.parameters
            assert o1.wires == o2.wires


fn = lambda func, *args, **kwargs: adjoint(func)(*args, **kwargs)


class TestTemplateIntegration:
    """Test that templates work correctly with the adjoint transform"""

    def test_angle_embedding(self):
        """Test that the adjoint correctly inverts angle embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.AngleEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([0.2, 0.5, 0.8])
        res = circuit(weights)
        assert len(np.nonzero(res)) == 1

    def test_amplitude_embedding(self):
        """Test that the adjoint correctly inverts amplitude embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.AmplitudeEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([0.2, 0.5, 0.8, 0.6, 0.1, 0.6, 0.1, 0.5]) / np.sqrt(1.92)
        res = circuit(weights)
        assert len(np.nonzero(res)) == 1

    def test_basis_embedding(self):
        """Test that the adjoint correctly inverts basis embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.BasisEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([1, 0, 1])
        res = circuit(weights)
        expected = np.zeros([2**3])
        expected[0] = 1.0
        assert np.allclose(res, expected)

    def test_displacement_embedding(self):
        """Test that the adjoint correctly inverts displacement embedding"""
        dev = qml.device("default.gaussian", wires=3)
        template = qml.templates.DisplacementEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.expval(qml.NumberOperator(0))

        weights = np.array([0.6, 0.2, 0.1])
        res = circuit(weights)
        assert np.allclose(res, 0.0)

    def test_squeezing_embedding(self):
        """Test that the adjoint correctly inverts squeezing embedding"""
        dev = qml.device("default.gaussian", wires=3)
        template = qml.templates.SqueezingEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.expval(qml.NumberOperator(0))

        weights = np.array([0.6, 0.2, 0.1])
        res = circuit(weights)
        assert np.allclose(res, 0.0)

    def test_qaoa_embedding(self):
        """Test that the adjoint correctly inverts qaoa embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.QAOAEmbedding

        @qml.qnode(dev)
        def circuit(features, weights):
            template(features=features, weights=weights, wires=[0, 1, 2])
            fn(template, features=features, weights=weights, wires=[0, 1, 2])
            return qml.state()

        features = np.array([1.0, 2.0, 3.0])
        weights = np.random.random(template.shape(2, 3))

        res = circuit(features, weights)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_iqp_embedding(self):
        """Test that the adjoint correctly inverts iqp embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.IQPEmbedding

        @qml.qnode(dev)
        def circuit(features):
            template(features=features, wires=[0, 1, 2])
            fn(template, features=features, wires=[0, 1, 2])
            return qml.state()

        features = np.array([1.0, 2.0, 3.0])
        res = circuit(features)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "template",
        [
            qml.templates.BasicEntanglerLayers,
            qml.templates.StronglyEntanglingLayers,
            qml.templates.RandomLayers,
        ],
    )
    def test_layers(self, template):
        """Test that the adjoint correctly inverts layers"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(weights):
            template(weights=weights, wires=[0, 1, 2])
            fn(template, weights=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.random.random(template.shape(2, 3))
        res = circuit(weights)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "template",
        [
            qml.templates.ParticleConservingU1,
            qml.templates.ParticleConservingU2,
        ],
    )
    def test_particle_conserving(self, template):
        """Test that the adjoint correctly inverts particle conserving layers"""
        dev = qml.device("default.qubit", wires=3)
        init_state = np.array([0, 1, 1])

        @qml.qnode(dev)
        def circuit(weights):
            template(weights=weights, init_state=init_state, wires=[0, 1, 2])
            fn(template, weights=weights, init_state=init_state, wires=[0, 1, 2])
            return qml.state()

        weights = np.random.random(template.shape(2, 3))
        res = circuit(weights)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_simplified_two_design(self):
        """Test that the adjoint correctly inverts the simplified two design"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.SimplifiedTwoDesign

        @qml.qnode(dev)
        def circuit(data, weights):
            template(initial_layer_weights=data, weights=weights, wires=[0, 1, 2])
            fn(template, initial_layer_weights=data, weights=weights, wires=[0, 1, 2])
            return qml.state()

        weights = [np.random.random(s) for s in template.shape(2, 3)]
        res = circuit(weights[0], *weights[1:])
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_approx_time_evolution(self):
        """Test that the adjoint correctly inverts the approx time evolution"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.ApproxTimeEvolution

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliX(1)]
        H = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit(t):
            template(H, t, 1)
            fn(template, H, t, 1)
            return qml.state()

        res = circuit(0.5)
        expected = np.zeros([2**3])
        expected[0] = 1.0
        assert np.allclose(res, expected)

    def test_arbitrary_unitary(self):
        """Test that the adjoint correctly inverts the arbitrary unitary"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.ArbitraryUnitary

        @qml.qnode(dev)
        def circuit(weights):
            template(weights=weights, wires=[0, 1, 2])
            fn(template, weights=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.random.random(template.shape(3))
        res = circuit(weights)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_single_excitation(self):
        """Test that the adjoint correctly inverts the single excitation unitary"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.FermionicSingleExcitation

        @qml.qnode(dev)
        def circuit(weights):
            template(weight=weights, wires=[0, 1, 2])
            fn(template, weight=weights, wires=[0, 1, 2])
            return qml.state()

        res = circuit(0.6)
        expected = np.zeros([2**3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_double_excitation(self):
        """Test that the adjoint correctly inverts the double excitation unitary"""
        dev = qml.device("default.qubit", wires=4)
        template = qml.templates.FermionicDoubleExcitation

        @qml.qnode(dev)
        def circuit(weights):
            template(weight=weights, wires1=[0, 1], wires2=[2, 3])
            fn(template, weight=weights, wires1=[0, 1], wires2=[2, 3])
            return qml.state()

        res = circuit(0.6)
        expected = np.zeros([2**4])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_interferometer(self):
        """Test that the adjoint correctly inverts squeezing embedding"""
        dev = qml.device("default.gaussian", wires=3)
        template = qml.templates.Interferometer
        r = 1.5

        @qml.qnode(dev)
        def circuit(weights):
            qml.Squeezing(r, 0, wires=0)
            qml.Squeezing(r, 0, wires=1)
            qml.Squeezing(r, 0, wires=2)
            template(*weights, wires=[0, 1, 2])
            fn(template, *weights, wires=[0, 1, 2])
            return qml.expval(qml.NumberOperator(0))

        weights = [
            np.random.random([3 * (3 - 1) // 2]),
            np.random.random([3 * (3 - 1) // 2]),
            np.random.random([3]),
        ]
        res = circuit(weights)
        assert np.allclose(res, np.sinh(r) ** 2)

    def test_gate_fabric(self):
        """Test that the adjoint correctly inverts the gate fabric template"""
        dev = qml.device("default.qubit", wires=4)
        template = qml.templates.GateFabric

        @qml.qnode(dev)
        def circuit(weights):
            template(weights=weights, wires=[0, 1, 2, 3], init_state=[1, 1, 0, 0])
            fn(template, weights=weights, wires=[0, 1, 2, 3], init_state=[1, 1, 0, 0])
            return qml.state()

        res = circuit([[[0.6, 0.8]]])
        expected = np.zeros([2**4])
        expected[0] = 1.0

        assert np.allclose(res, expected)


def test_op_that_overwrites_expand():
    """Tests the adjoint method applied on an operation that overwrites the expand method.

    .. note::
        This is a discontinued practice, since all operators should define their decomposition
        in decomposition() or compute_decomposition(). Once the new standard is established
        everywhere, we can remove the "if isinstance(new_ops, QuantumTape)" check in
        the adjoint method.
    """
    dev = qml.device("default.qubit", wires=3)

    class MyOp(qml.operation.Operation):
        num_wires = 1

        def expand(self):
            with qml.tape.QuantumTape() as tape:
                qml.RX(0.1, wires=self.wires)
            return tape

    @qml.qnode(dev)
    def circuit():
        MyOp(wires=[0])
        qml.adjoint(MyOp)(wires=[0])
        return qml.state()

    res = circuit()
    assert len(np.nonzero(res)) == 1
