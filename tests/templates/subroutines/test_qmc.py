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
"""
Unit tests for the QuantumMonteCarlo subroutine template.
"""
import numpy as np
import pytest
from scipy.stats import norm

import pennylane as qml
from pennylane.templates.subroutines.qmc import (
    QuantumMonteCarlo,
    _make_V,
    _make_Z,
    func_to_unitary,
    make_Q,
    probs_to_unitary,
)
from pennylane.wires import Wires


class TestProbsToUnitary:
    """Tests for the probs_to_unitary function"""

    def test_invalid_distribution_sum_to_not_one(self):
        """Test if a ValueError is raised when a distribution that does not sum to one is input"""
        p = np.ones(4)
        with pytest.raises(ValueError, match="A valid probability distribution of non-negative"):
            probs_to_unitary(p)

    def test_invalid_distribution_negative(self):
        """Test if a ValueError is raised when a distribution with a negative value is input"""
        p = [2, 0, 0, -1]
        with pytest.raises(ValueError, match="A valid probability distribution of non-negative"):
            probs_to_unitary(p)

    ps = [
        [0.46085261032920616, 0.5391473896707938],
        [0.2111821738452515, 0.4235979103670337, 0.36521991578771484],
        [0.3167916924190049, 0.2651843704361695, 0.1871934980886578, 0.23083043905616774],
        [0.8123242419241959, 0.07990911578859018, 0.07983919018902215, 0.027927452098191852],
    ]

    @pytest.mark.parametrize("p", ps)
    def test_fixed_examples(self, p):
        """Test if the correct unitary is returned for fixed input examples. A correct unitary has
        its first column equal to the square root of the distribution and satisfies
        U @ U.T = U.T @ U = I."""
        unitary = probs_to_unitary(p)
        assert np.allclose(np.sqrt(p), unitary[:, 0])
        assert np.allclose(unitary @ unitary.T, np.eye(len(unitary)))
        assert np.allclose(unitary.T @ unitary, np.eye(len(unitary)))

    @pytest.mark.jax
    @pytest.mark.parametrize("p", ps)
    def test_fixed_examples_jax_jit(self, p):
        """Test if the correct unitary is returned for fixed input examples using JAX-JIT.
        A correct unitary has its first column equal to the square root of the distribution
        and satisfies U @ U.T = U.T @ U = I."""
        import jax
        from jax import numpy as jnp

        unitary = jax.jit(probs_to_unitary)(jnp.array(p))
        assert jnp.allclose(np.sqrt(p), unitary[:, 0])
        assert jnp.allclose(unitary @ unitary.T, np.eye(len(unitary)), atol=1e-7)
        assert jnp.allclose(unitary.T @ unitary, np.eye(len(unitary)), atol=1e-7)


class TestFuncToUnitary:
    """Tests for the func_to_unitary function"""

    @staticmethod
    def func(i):
        return np.sin(i) ** 2

    def test_not_bounded_func(self):
        """Test if a ValueError is raised if a function that evaluates outside of the [0, 1]
        interval is provided"""

        with pytest.raises(ValueError, match="func must be bounded within the interval"):
            func_to_unitary(np.sin, 8)

    def test_example(self):
        """Test for a fixed example if the returned unitary maps input states to the
        expected output state as well as if the unitary satisfies U @ U.T = U.T @ U = I."""
        M = 8

        r = func_to_unitary(self.func, M)

        for i in range(M):
            # The control qubit is the last qubit, so we have to look at every other term
            # using [::2].
            output_state = r[::2][i]
            output_0 = output_state[::2]
            output_1 = output_state[1::2]
            assert np.allclose(output_0[i], np.sqrt(1 - self.func(i)))
            assert np.allclose(output_1[i], np.sqrt(self.func(i)))

        assert np.allclose(r @ r.T, np.eye(2 * M))
        assert np.allclose(r.T @ r, np.eye(2 * M))

    @pytest.mark.jax
    def test_example_jax_jit(self):
        """Test for a fixed example using JAX-JIT if the returned unitary maps input states to the
        expected output state as well as if the unitary satisfies U @ U.T = U.T @ U = I."""
        import jax
        from jax import numpy as jnp

        M = 8

        def func(i):
            return jnp.sin(i) ** 2

        r = func_to_unitary(jax.jit(func), M)

        for i in range(M):
            # The control qubit is the last qubit, so we have to look at every other term
            # using [::2].
            output_state = r[::2][i]
            output_0 = output_state[::2]
            output_1 = output_state[1::2]
            assert np.allclose(output_0[i], np.sqrt(1 - func(i)))
            assert np.allclose(output_1[i], np.sqrt(func(i)))

        assert np.allclose(r @ r.T, np.eye(2 * M), atol=1e-7)
        assert np.allclose(r.T @ r, np.eye(2 * M), atol=1e-7)

    def test_example_with_pl(self):
        """Test for a fixed example if the returned unitary behaves as expected
        when used within a PennyLane circuit, i.e., so that the probability of the final control
        wire encodes the function."""
        wires = 3
        M = 2**wires
        r = func_to_unitary(self.func, M)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def apply_r(input_state):
            qml.StatePrep(input_state, wires=range(wires))
            qml.QubitUnitary(r, wires=range(wires + 1))
            return qml.probs(wires)

        for i, state in enumerate(np.eye(M)):
            p = apply_r(state)[1]
            assert np.allclose(p, self.func(i))


def test_V():
    """Test for the _make_V function"""
    dim = 4

    V_expected = -np.eye(dim)
    V_expected[1, 1] = V_expected[3, 3] = 1
    V = _make_V(dim)

    assert np.allclose(V, V_expected)


def test_Z():
    """Test for the _make_Z function"""
    dim = 4

    Z_expected = -np.eye(dim)
    Z_expected[0, 0] = 1
    Z = _make_Z(dim)

    assert np.allclose(Z, Z_expected)


def test_Q():
    """Test for the make_Q function using a fixed example"""

    A = np.array(
        [
            [0.85358423 - 0.32239299j, -0.12753659 + 0.38883306j],
            [0.39148136 - 0.11915985j, 0.34064316 - 0.84646648j],
        ]
    )
    R = np.array(
        [
            [
                0.45885289 + 0.03972856j,
                0.2798685 - 0.05981098j,
                0.64514642 - 0.51555038j,
                0.11015177 - 0.10877695j,
            ],
            [
                0.19407005 - 0.35483005j,
                0.29756077 + 0.80153453j,
                -0.19147104 + 0.0507968j,
                0.15553799 - 0.20493631j,
            ],
            [
                0.35083011 - 0.20807392j,
                -0.27602911 - 0.13934692j,
                0.11874165 + 0.34532609j,
                -0.45945242 - 0.62734969j,
            ],
            [
                -0.11379919 - 0.66706921j,
                -0.21120956 - 0.2165113j,
                0.30133006 + 0.23367271j,
                0.54593491 + 0.08446372j,
            ],
        ]
    )

    Q_expected = np.array(
        [
            [
                -0.46513201 - 1.38777878e-17j,
                -0.13035515 - 2.23341802e-01j,
                -0.74047856 + 7.08652160e-02j,
                -0.0990036 - 3.91977176e-01j,
            ],
            [
                0.13035515 - 2.23341802e-01j,
                0.46494302 + 0.00000000e00j,
                0.05507901 - 1.19182067e-01j,
                -0.80370146 - 2.31904873e-01j,
            ],
            [
                -0.74047856 - 7.08652160e-02j,
                -0.05507901 - 1.19182067e-01j,
                0.62233412 - 2.77555756e-17j,
                -0.0310774 - 2.02894077e-01j,
            ],
            [
                0.0990036 - 3.91977176e-01j,
                -0.80370146 + 2.31904873e-01j,
                0.0310774 - 2.02894077e-01j,
                -0.30774091 + 2.77555756e-17j,
            ],
        ]
    )

    Q = make_Q(A, R)

    assert np.allclose(Q, Q_expected)


class TestQuantumMonteCarlo:
    """Tests for the QuantumMonteCarlo template"""

    @staticmethod
    def func(i):
        return np.sin(i) ** 2

    @pytest.mark.jax
    def test_standard_validity(self):
        """Test standard validity criteria with assert_valid."""
        p = np.ones(4) / 4
        target_wires, estimation_wires = Wires(range(3)), Wires(range(3, 5))

        op = QuantumMonteCarlo(p, self.func, target_wires, estimation_wires)
        # Skip capture test because the _unflatten method of QMC is not compatible with capture
        qml.ops.functions.assert_valid(op, skip_differentiation=True, skip_capture=True)

    def test_non_flat(self):
        """Test if a ValueError is raised when a non-flat array is input"""
        p = np.ones((4, 1)) / 4
        with pytest.raises(ValueError, match="The probability distribution must be specified as a"):
            QuantumMonteCarlo(p, self.func, range(3), range(3, 5))

    def test_wrong_size_p(self):
        """Test if a ValueError is raised when a probability distribution is passed whose length
        cannot be mapped to qubits"""
        p = np.ones(5) / 5
        with pytest.raises(ValueError, match="The probability distribution must have a length"):
            QuantumMonteCarlo(p, self.func, range(3), range(3, 5))

    def test_unexpected_target_wires_number(self):
        """Test if a ValueError is raised when the number of target wires is incompatible with the
        expected number of target wires inferred from the length of the input probability
        distribution"""
        p = np.ones(4) / 4
        with pytest.raises(
            ValueError,
            match="The probability distribution of dimension 4 requires" " 3 target wires",
        ):
            QuantumMonteCarlo(p, self.func, range(4), range(4, 6))

    def test_expected_circuit(self):
        """Test if the circuit applied when using the QMC template is the same as the expected
        circuit for a fixed example"""
        p = np.ones(4) / 4
        target_wires, estimation_wires = Wires(range(3)), Wires(range(3, 5))

        op = QuantumMonteCarlo(p, self.func, target_wires, estimation_wires)
        tape = qml.tape.QuantumScript(op.decomposition())

        # Do expansion in two steps to avoid also decomposing the first QubitUnitary
        queue_before_qpe = tape.operations[:2]

        # Build a new tape from all operations following the two QubitUnitary ops and expand it
        queue_after_qpe = qml.tape.QuantumScript(tape.operations[2:]).expand().operations

        A = probs_to_unitary(p)
        R = func_to_unitary(self.func, 4)

        assert len(queue_before_qpe) == 2
        assert queue_before_qpe[0].name == "QubitUnitary"
        assert queue_before_qpe[1].name == "QubitUnitary"
        assert np.allclose(queue_before_qpe[0].matrix(), A)
        assert np.allclose(queue_before_qpe[1].matrix(), R)
        assert queue_before_qpe[0].wires == target_wires[:-1]
        assert queue_before_qpe[1].wires == target_wires

        Q = make_Q(A, R)

        with qml.queuing.AnnotatedQueue() as q_qpe_tape:
            qml.QuantumPhaseEstimation(Q, target_wires, estimation_wires)

        qpe_tape = qml.tape.QuantumScript.from_queue(q_qpe_tape)
        qpe_tape = qpe_tape.expand()

        assert len(queue_after_qpe) == len(qpe_tape.operations)
        assert all(o1.name == o2.name for o1, o2 in zip(queue_after_qpe, qpe_tape.operations))
        assert all(
            np.allclose(o1.matrix(), o2.matrix())
            for o1, o2 in zip(queue_after_qpe, qpe_tape.operations)
        )
        assert all(o1.wires == o2.wires for o1, o2 in zip(queue_after_qpe, qpe_tape.operations))

    def test_expected_value(self):
        """Test that the QuantumMonteCarlo template can correctly estimate the expectation value
        following the example in the usage details"""
        # pylint: disable=cell-var-from-loop
        m = 5
        M = 2**m

        xmax = np.pi
        xs = np.linspace(-xmax, xmax, M)

        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        def func(i):
            return np.cos(xs[i]) ** 2

        estimates = []

        for n in range(4, 11):
            N = 2**n

            target_wires = range(m + 1)
            estimation_wires = range(m + 1, n + m + 1)

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.QuantumMonteCarlo(
                    probs, func, target_wires=target_wires, estimation_wires=estimation_wires
                )
                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()[: int(N / 2)]) / N
            mu_estimated = (1 - np.cos(np.pi * phase_estimated)) / 2
            estimates.append(mu_estimated)

        exact = 0.432332358381693654

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - exact)
            err2 = np.abs(estimates[i + 1] - exact)
            assert err1 >= err2

        assert np.allclose(estimates[-1], exact, rtol=1e-3)

    @pytest.mark.jax
    def test_expected_value_jax_jit(self):
        """Test that the QuantumMonteCarlo template can correctly estimate the expectation value
        following the example in the usage details using JAX-JIT"""
        # pylint: disable=cell-var-from-loop
        import jax
        from jax import numpy as jnp

        m = 5
        M = 2**m

        xmax = jnp.pi
        xs = jnp.linspace(-xmax, xmax, M)

        probs = jnp.array([norm().pdf(x) for x in xs])
        probs /= jnp.sum(probs)

        def func(i):
            return jnp.cos(xs[i]) ** 2

        estimates = []

        for n in range(4, 11):
            N = 2**n

            target_wires = range(m + 1)
            estimation_wires = range(m + 1, n + m + 1)

            dev = qml.device("default.qubit")

            @jax.jit
            @qml.qnode(dev, interface="jax")
            def circuit():
                qml.QuantumMonteCarlo(
                    probs, func, target_wires=target_wires, estimation_wires=estimation_wires
                )
                return qml.probs(estimation_wires)

            phase_estimated = jnp.argmax(circuit()[: int(N / 2)]) / N
            mu_estimated = (1 - jnp.cos(np.pi * phase_estimated)) / 2
            estimates.append(mu_estimated)

        exact = 0.432332358381693654

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = jnp.abs(estimates[i] - exact)
            err2 = jnp.abs(estimates[i + 1] - exact)
            assert err1 >= err2

        assert jnp.allclose(estimates[-1], exact, rtol=1e-3)

    def test_expected_value_custom_wires(self):
        """Test that the QuantumMonteCarlo template can correctly estimate the expectation value
        following the example in the usage details when the wires have custom labels"""
        m = 5
        M = 2**m

        xmax = np.pi
        xs = np.linspace(-xmax, xmax, M)

        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        def func(i):
            return np.cos(xs[i]) ** 2

        n = 10
        N = 2**n

        target_wires = [0, "a", -1.1, -10, "bbb", 1000]
        estimation_wires = ["bob", -3, 42, "penny", "lane", 247, "straw", "berry", 5.5, 6.6]

        dev = qml.device("default.qubit", wires=target_wires + estimation_wires)

        @qml.qnode(dev)
        def circuit():
            qml.QuantumMonteCarlo(
                probs, func, target_wires=target_wires, estimation_wires=estimation_wires
            )
            return qml.probs(estimation_wires)

        phase_estimated = np.argmax(circuit()[: int(N / 2)]) / N
        mu_estimated = (1 - np.cos(np.pi * phase_estimated)) / 2

        exact = 0.432332358381693654
        assert np.allclose(mu_estimated, exact, rtol=1e-3)

    def test_id(self):
        """Tests that the id attribute can be set."""
        xs = np.linspace(-np.pi, np.pi, 2**5)
        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        def func(i):
            return np.cos(xs[i]) ** 2

        target_wires = [0, "a", -1.1, -10, "bbb", 1000]
        estimation_wires = ["bob", -3, 42, "penny", "lane", 247, "straw", "berry", 5.5, 6.6]

        template = qml.QuantumMonteCarlo(
            probs, func, target_wires=target_wires, estimation_wires=estimation_wires, id="a"
        )

        assert template.id == "a"
