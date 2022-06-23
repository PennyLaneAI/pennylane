# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for mitigation transforms.
"""
import pytest
import numpy
from packaging import version

import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape
from pennylane.transforms import mitigate_with_zne, poly_extrapolate, fold_global

with QuantumTape() as tape:
    qml.BasisState([1], wires=0)
    qml.RX(0.9, wires=0)
    qml.RY(0.4, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=0)
    qml.RX(0.6, wires=1)
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

with QuantumTape() as tape_base:
    qml.RX(0.9, wires=0)
    qml.RY(0.4, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=0)
    qml.RX(0.6, wires=1)


def same_tape(tape1, tape2):
    """Raises an error if tapes are not identical"""
    assert all(o1.name == o2.name for o1, o2 in zip(tape1.operations, tape2.operations))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape1.operations, tape2.operations))
    assert all(
        np.allclose(o1.parameters, o2.parameters)
        for o1, o2 in zip(tape1.operations, tape2.operations)
    )
    assert len(tape1.measurements) == len(tape2.measurements)
    assert all(
        m1.return_type == m2.return_type for m1, m2 in zip(tape1.measurements, tape2.measurements)
    )
    assert all(o1.name == o2.name for o1, o2 in zip(tape1.observables, tape2.observables))
    assert all(o1.wires == o2.wires for o1, o2 in zip(tape1.observables, tape2.observables))


class TestMitigateWithZNE:
    """Tests for the mitigate_with_zne function"""

    folding = lambda *args, **kwargs: tape_base
    extrapolate = lambda *args, **kwargs: [3.141]

    def test_folding_call(self, mocker):
        """Tests that arguments are passed to the folding function as expected"""
        spy = mocker.spy(self, "folding")
        scale_factors = [1, 2, -4]
        folding_kwargs = {"Hello": "goodbye"}

        mitigate_with_zne(tape, scale_factors, self.folding, self.extrapolate, folding_kwargs)

        args = spy.call_args_list

        for i in range(3):
            same_tape(args[i][0][0], tape_base)
        assert [args[i][0][1] for i in range(3)] == scale_factors
        assert all(args[i][1] == folding_kwargs for i in range(3))

    def test_extrapolate_call(self, mocker):
        """Tests that arguments are passed to the extrapolate function as expected"""
        spy = mocker.spy(self, "extrapolate")
        scale_factors = [1, 2, -4]
        random_results = [0.1, 0.2, 0.3]
        extrapolate_kwargs = {"Hello": "goodbye"}

        tapes, fn = mitigate_with_zne(
            tape,
            scale_factors,
            self.folding,
            self.extrapolate,
            extrapolate_kwargs=extrapolate_kwargs,
        )
        res = fn(random_results)
        assert res == 3.141

        args = spy.call_args
        assert args[0][0] == scale_factors
        assert np.allclose(args[0][1], random_results)

        assert args[1] == extrapolate_kwargs

        for t in tapes:
            same_tape(t, tape)

    def test_reps_per_factor_not_1(self, mocker):
        """Tests if mitigation proceeds as expected when reps_per_factor is not 1 (default)"""
        scale_factors = [1, 2, -4]
        spy_fold = mocker.spy(self, "folding")
        spy_extrapolate = mocker.spy(self, "extrapolate")
        tapes, fn = mitigate_with_zne(
            tape, scale_factors, self.folding, self.extrapolate, reps_per_factor=2
        )
        random_results = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        args = spy_fold.call_args_list
        for i in range(6):
            same_tape(args[i][0][0], tape_base)
        assert [args[i][0][1] for i in range(6)] == [1, 1, 2, 2, -4, -4]

        fn(random_results)

        args = spy_extrapolate.call_args
        assert args[0][0] == scale_factors
        assert np.allclose(args[0][1], np.mean(np.reshape(random_results, (3, 2)), axis=1))


@pytest.fixture
def skip_if_no_mitiq_support():
    """Fixture to skip if minimum version of mitiq is not available"""
    try:
        import mitiq

        v = version.parse(mitiq.__version__)
        t = version.parse("0.11.0")
        if v.major < t.major and v.minor < t.minor:
            pytest.skip("Mitiq version too low")
    except ImportError:
        pytest.skip("Mitiq not available")


@pytest.fixture
def skip_if_no_pl_qiskit_support():
    """Fixture to skip if pennylane_qiskit is not available"""
    pytest.importorskip("pennylane_qiskit")


@pytest.mark.usefixtures("skip_if_no_pl_qiskit_support")
@pytest.mark.usefixtures("skip_if_no_mitiq_support")
class TestMitiqIntegration:
    """Tests if the mitigate_with_zne transform is compatible with using mitiq as a backend"""

    def test_multiple_returns(self):
        """Tests if the expected shape is returned when mitigating a circuit with two returns"""
        from mitiq.zne.scaling import fold_global
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_single_return(self):
        """Tests if the expected shape is returned when mitigating a circuit with a single return"""
        from mitiq.zne.scaling import fold_global
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_with_reps_per_factor(self):
        """Tests if the expected shape is returned when mitigating a circuit with a reps_per_factor
        set not equal to 1"""
        from mitiq.zne.scaling import fold_gates_at_random
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @qml.transforms.mitigate_with_zne(
            [1, 2, 3], fold_gates_at_random, RichardsonFactory.extrapolate, reps_per_factor=2
        )
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_noise_free)
        def ideal_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        res_mitigated = mitigated_circuit(w1, w2)
        res_ideal = ideal_circuit(w1, w2)

        assert res_mitigated.shape == res_ideal.shape
        assert not np.allclose(res_mitigated, res_ideal)

    def test_integration(self):
        """Test if the error of the mitigated result is less than the error of the unmitigated
        result for a circuit with known expectation values"""
        from mitiq.zne.scaling import fold_global
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            qml.adjoint(qml.SimplifiedTwoDesign)(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        exact_qnode = qml.QNode(circuit, dev_noise_free)
        noisy_qnode = qml.QNode(circuit, dev)

        @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
        @qml.qnode(dev)
        def mitigated_qnode(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            qml.adjoint(qml.SimplifiedTwoDesign)(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        exact_val = exact_qnode(w1, w2)
        noisy_val = noisy_qnode(w1, w2)
        mitigated_val = mitigated_qnode(w1, w2)

        mitigated_err = np.abs(exact_val - mitigated_val)
        noisy_err = np.abs(exact_val - noisy_val)

        assert np.allclose(exact_val, [1, 1])
        assert all(mitigated_err < noisy_err)

    @pytest.mark.xfail(
        reason="Using external tape transforms breaks differentiability",
    )
    def test_grad(self):
        """Tests if the gradient is calculated successfully."""
        from mitiq.zne.scaling import fold_global
        from mitiq.zne.inference import RichardsonFactory

        noise_strength = 0.05

        dev_noise_free = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev_noise_free)

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s, requires_grad=True) for s in shapes]

        @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
        @qml.qnode(dev)
        def mitigated_circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        g = qml.grad(mitigated_circuit)(w1, w2)
        for g_ in g:
            assert not np.allclose(g_, 0)


class TestFoldGlobal:
    """Testing ``qml.transforms.fold_global"""

    def test_constant_result(self):
        """Ensuring that the circuit always yields the same result upon folding."""

        dev = qml.device("default.qubit", wires=5)

        # Select template to use within circuit and generate parameters
        n_layers = 2
        n_wires = 3
        template = qml.SimplifiedTwoDesign
        weights_shape = template.shape(n_layers, n_wires)
        w1, w2 = [np.arange(np.prod(s)).reshape(s) for s in weights_shape]

        # This circuit itself produces the identity by construction
        with qml.tape.QuantumTape() as circuit:
            template(w1, w2, wires=range(n_wires))
            qml.adjoint(template(w1, w2, wires=range(n_wires)))
            qml.expval(qml.PauliZ(0))

        folded_qnodes = [fold_global(circuit, scale_factor=lambda_) for lambda_ in np.arange(1, 10)]
        res = [qml.execute([folded], dev, None) for folded in folded_qnodes]
        assert np.allclose(res, 1)

    @pytest.mark.jax
    @pytest.mark.autograd
    def test_polyfit(self):
        """Testing the poly_extrapolator function in autograd"""
        import jax.numpy as jnp

        x = jnp.arange(10, dtype="float32")
        y = 0.5 * x**2

        assert qml.math.allclose(poly_extrapolate(x, y, 2), 0.0, atol=1e-6)

        x = np.arange(10)
        y = 0.5 * x**2

        assert qml.math.allclose(poly_extrapolate(x, y, 2), 0.0)
