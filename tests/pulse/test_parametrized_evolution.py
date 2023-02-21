# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the ParametrizedEvolution class
"""
from functools import reduce

import numpy as np

# pylint: disable=unused-argument, too-few-public-methods
import pytest

import pennylane as qml
from pennylane.operation import AnyWires
from pennylane.ops import QubitUnitary
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian
from pennylane.tape import QuantumTape


class MyOp(qml.RX):  # pylint: disable=too-few-public-methods
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


def time_independent_hamiltonian():
    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params  # constant

    def f2(params, t):
        return params  # constant

    coeffs = [f1, f2, 4, 9]

    return ParametrizedHamiltonian(coeffs, ops)


def time_dependent_hamiltonian():
    import jax.numpy as jnp

    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params * t

    def f2(params, t):
        return params * jnp.cos(t)

    coeffs = [f1, f2, 4, 9]
    return ParametrizedHamiltonian(coeffs, ops)


def test_error_raised_if_jax_not_installed():
    """Test that an error is raised if an ``Evolve`` operator is instantiated without jax installed"""
    try:
        import jax  # pylint: disable=unused-import

        pytest.skip()
    except ImportError:
        with pytest.raises(ImportError, match="Module jax is required"):
            ParametrizedEvolution(H=ParametrizedHamiltonian([1], [qml.PauliX(0)]))


@pytest.mark.jax
class TestInitialization:
    """Unit tests for the ParametrizedEvolution class."""

    def test_init(self):
        """Test the initialization."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=2)

        assert ev.H is H
        assert qml.math.allequal(ev.t, [0, 2])

        assert ev.wires == H.wires
        assert ev.num_wires == AnyWires
        assert ev.name == "ParametrizedEvolution"
        assert ev.id is None
        assert ev.queue_idx is None

        assert qml.math.allequal(ev.data, [1, 2])
        assert qml.math.allequal(ev.parameters, [1, 2])
        assert ev.num_params == 2

    def test_odeint_kwargs(self):
        """Test the initialization with odeint kwargs."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=2, mxstep=10, hmax=1, atol=1e-3, rtol=1e-6)

        assert ev.odeint_kwargs == {"mxstep": 10, "hmax": 1, "atol": 1e-3, "rtol": 1e-6}

    def test_update_attributes(self):
        """Test that the ``ParametrizedEvolution`` attributes can be updated using the ``__call__`` method."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, mxstep=10)

        assert ev.parameters == []
        assert ev.num_params == 0
        assert ev.t is None
        assert ev.odeint_kwargs == {"mxstep": 10}
        params = [1, 2, 3]
        t = 6
        new_ev = ev(params, t, atol=1e-6, rtol=1e-4)

        assert new_ev is not ev
        assert qml.math.allequal(new_ev.parameters, params)
        assert new_ev.num_params == 3
        assert qml.math.allequal(new_ev.t, [0, 6])
        assert new_ev.odeint_kwargs == {"mxstep": 10, "atol": 1e-6, "rtol": 1e-4}

    def test_update_attributes_inside_queuing_context(self):
        """Make sure that updating a ``ParametrizedEvolution`` inside a queuing context, the initial
        operator is removed from the queue."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)

        with QuantumTape() as tape:
            op = qml.evolve(H)
            op2 = op(params=[1, 2, 3], t=6)

        assert len(tape) == 1
        assert tape[0] is op2

    def test_list_of_times(self):
        """Test the initialization."""
        import jax.numpy as jnp

        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        t = np.arange(0, 10, 0.01)
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=t)

        assert isinstance(ev.t, jnp.ndarray)
        assert qml.math.allclose(ev.t, t)

    def test_has_matrix_true(self):
        """Test that a parametrized evolution has ``has_matrix=True`` only when `t` and `params` are
        defined."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H)
        assert ev.has_matrix is False
        new_ev = ev(params=[1, 2, 3], t=3)
        assert new_ev.has_matrix is True

    def test_evolve_with_operator_without_matrix_raises_error(self):
        """Test that an error is raised when an ``ParametrizedEvolution`` operator is initialized with a
        ``ParametrizedHamiltonian`` that contains an operator without a matrix defined."""
        ops = [qml.PauliX(0), MyOp(phi=0, wires=0)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        with pytest.raises(
            ValueError,
            match="All operators inside the parametrized hamiltonian must have a matrix defined",
        ):
            _ = ParametrizedEvolution(H=H, params=[1, 2], t=2)


@pytest.mark.jax
class TestMatrix:
    """Test matrix method."""

    # pylint: disable=unused-argument
    def test_time_independent_hamiltonian(self):
        """Test matrix method for a time independent hamiltonian."""
        H = time_independent_hamiltonian()
        t = np.arange(0, 4, 0.001)
        params = [1, 2]
        ev = ParametrizedEvolution(H=H, params=params, t=t, hmax=1, mxstep=1e4)
        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=max(t))) * max(t))
        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-3)

    @pytest.mark.slow
    def test_time_dependent_hamiltonian(self):
        """Test matrix method for a time dependent hamiltonian. This test approximates the
        time-ordered exponential with a product of exponentials using small time steps.
        For more information, see https://en.wikipedia.org/wiki/Ordered_exponential."""
        import jax
        import jax.numpy as jnp

        H = time_dependent_hamiltonian()

        t = jnp.arange(0, jnp.pi / 4, 0.001)
        params = [1, 2]
        ev = ParametrizedEvolution(H=H, params=params, t=t, atol=1e-6, rtol=1e-6)

        def generator(params):
            for ti in t:
                yield jax.scipy.linalg.expm(-1j * 0.001 * qml.matrix(H(params, t=ti)))

        true_mat = reduce(lambda x, y: y @ x, generator(params))

        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-2)


@pytest.mark.jax
class TestIntegration:
    """Integration tests for the ParametrizedEvolution class."""

    # pylint: disable=unused-argument
    def test_time_independent_hamiltonian(self):
        """Test the execution of a time independent hamiltonian."""
        import jax
        import jax.numpy as jnp

        H = time_independent_hamiltonian()

        dev = qml.device("default.qubit", wires=2)

        t = 4

        @qml.qnode(dev)
        def circuit(params):
            ParametrizedEvolution(H=H, params=params, t=t)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        @jax.jit
        @qml.qnode(dev)
        def jitted_circuit(params):
            ParametrizedEvolution(H=H, params=params, t=t)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        @qml.qnode(dev)
        def true_circuit(params):
            true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
            QubitUnitary(U=true_mat, wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        params = jnp.array([1.0, 2.0])

        assert qml.math.allclose(circuit(params), true_circuit(params), atol=1e-3)
        assert qml.math.allclose(jitted_circuit(params), true_circuit(params), atol=1e-3)
        assert qml.math.allclose(
            jax.grad(circuit)(params), jax.grad(true_circuit)(params), atol=1e-3
        )
        assert qml.math.allclose(
            jax.grad(jitted_circuit)(params), jax.grad(true_circuit)(params), atol=1e-3
        )

    @pytest.mark.slow
    def test_time_dependent_hamiltonian(self):
        """Test the execution of a time dependent hamiltonian. This test approximates the
        time-ordered exponential with a product of exponentials using small time steps.
        For more information, see https://en.wikipedia.org/wiki/Ordered_exponential."""
        import jax
        import jax.numpy as jnp

        H = time_dependent_hamiltonian()

        dev = qml.device("default.qubit", wires=2)
        t = 2

        def generator(params):
            time_step = 1e-3
            times = jnp.arange(0, t, step=time_step)
            for ti in times:
                yield jax.scipy.linalg.expm(-1j * time_step * qml.matrix(H(params, t=ti)))

        @qml.qnode(dev)
        def circuit(params):
            ParametrizedEvolution(H=H, params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        @jax.jit
        @qml.qnode(dev)
        def jitted_circuit(params):
            ParametrizedEvolution(H=H, params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        @qml.qnode(dev)
        def true_circuit(params):
            true_mat = reduce(lambda x, y: y @ x, generator(params))
            QubitUnitary(U=true_mat, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        params = jnp.array([1.0, 2.0])

        assert qml.math.allclose(circuit(params), true_circuit(params), atol=5e-3)
        assert qml.math.allclose(jitted_circuit(params), true_circuit(params), atol=5e-3)
        assert qml.math.allclose(
            jax.grad(circuit)(params), jax.grad(true_circuit)(params), atol=5e-3
        )
        assert qml.math.allclose(
            jax.grad(jitted_circuit)(params), jax.grad(true_circuit)(params), atol=5e-3
        )

    def test_two_commuting_parametrized_hamiltonians(self):
        """Test that the evolution of two parametrized hamiltonians that commute with each other
        is equal to evolve the two hamiltonians simultaneously. This test uses 8 wires for the device
        to test the case where 2 * n < N (the matrix is evolved instead of the state)."""
        import jax
        import jax.numpy as jnp

        def f1(p, t):
            return p * t

        def f2(p, t):
            return jnp.sin(t) * (p - 1)

        coeffs = [1, f1, f2]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliX(2)]
        H1 = qml.dot(coeffs, ops)

        def f3(p, t):
            return jnp.cos(t) * (p + 1)

        coeffs = [7, f3]
        ops = [qml.PauliX(0), qml.PauliX(2)]
        H2 = qml.dot(coeffs, ops)

        dev = qml.device("default.qubit", wires=8)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit1(params):
            qml.evolve(H1)(params[0], t=2)
            qml.evolve(H2)(params[1], t=2)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit2(params):
            qml.evolve(H1 + H2)(params, t=2)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        params1 = [(1.0, 2.0), (3.0,)]
        params2 = [1.0, 2.0, 3.0]

        assert qml.math.allclose(circuit1(params1), circuit2(params2), atol=5e-4)
        assert qml.math.allclose(
            qml.math.concatenate(jax.grad(circuit1)(params1)),
            jax.grad(circuit2)(params2),
            atol=5e-4,
        )

    def test_mixed_device(
        self,
    ):
        """Test mixed device integration matches that of default qubit"""
        import jax
        import jax.numpy as jnp

        mixed = qml.device("default.mixed", wires=range(3))
        default = qml.device("default.qubit", wires=range(3))

        coeff = [qml.pulse.pwc(5.0), qml.pulse.pwc(5.0)]
        ops = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(1) @ qml.PauliY(2)]
        H_pulse = qml.dot(coeff, ops)

        @qml.qnode(default, interface="jax")
        def qnode_def(x):
            qml.pulse.ParametrizedEvolution(H_pulse, x, 5.0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(mixed, interface="jax")
        def qnode_mix(x):
            qml.pulse.ParametrizedEvolution(H_pulse, x, 5.0)
            return qml.expval(qml.PauliZ(0))

        x = [jnp.arange(3, dtype=float)]*2
        res_def = qnode_def(x)
        grad_def = jax.grad(qnode_def)(x)

        res_mix = qnode_mix(x)
        grad_mix = jax.grad(qnode_mix)(x)

        assert qml.math.isclose(res_def, res_mix, atol=1e-4)
        assert qml.math.allclose(grad_def, grad_mix, atol=1e-4)
