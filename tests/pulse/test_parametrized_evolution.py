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
# pylint: disable=unused-argument,too-few-public-methods,import-outside-toplevel
from functools import reduce
import numpy as np

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
    """Create a time-independent Hamiltonian on two qubits."""
    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params  # constant

    def f2(params, t):
        return params  # constant

    coeffs = [f1, f2, 4, 9]

    return ParametrizedHamiltonian(coeffs, ops)


def time_dependent_hamiltonian():
    """Create a time-dependent two-qubit Hamiltonian that takes two scalar parameters."""
    import jax.numpy as jnp

    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params * t

    def f2(params, t):
        return params * jnp.cos(t)

    coeffs = [f1, f2, 4, 9]
    return ParametrizedHamiltonian(coeffs, ops)


@pytest.mark.jax
class TestInitialization:
    """Unit tests for the ParametrizedEvolution class."""

    def test_init(self):
        """Test the initialization."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=2, dense=True)

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
        assert ev.dense is True

    def test_set_dense(self):
        """Test that flag dense is set correctly"""
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        coeffs = [1, 2, 3]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=None, t=2)
        assert ev.dense is False

        ev2 = ParametrizedEvolution(H=H, params=None, t=2, dense=True)
        assert ev2.dense is True

        ev3 = ParametrizedEvolution(H=H, params=None, t=2, dense=False)
        assert ev3.dense is False

    @pytest.mark.parametrize("ret_intmdt, comp", ([False, False], [True, False], [True, True]))
    def test_return_intermediate_and_complementary(self, ret_intmdt, comp):
        """Test that the keyword arguments return_intermediate and complementary are taken into
        account correctly at initialization and when calling. This includes testing
        inheritance when calling without explicitly providing these kwargs."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        t = [0.1, 0.2, 0.9]  # avoid warning because of simple time argument+return_intermediate
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(
            H=H, params=coeffs, t=t, return_intermediate=ret_intmdt, complementary=comp
        )

        assert ev.hyperparameters["return_intermediate"] is ret_intmdt
        assert ev.hyperparameters["complementary"] is comp

        new_ev = ev(coeffs, t=t)

        assert new_ev.hyperparameters["return_intermediate"] is ret_intmdt
        assert new_ev.hyperparameters["complementary"] is comp

        for new_ret_intmdt, new_comp in ([False, False], [True, False], [True, True]):
            new_ev = ev(coeffs, t=t, return_intermediate=new_ret_intmdt, complementary=new_comp)
            assert new_ev.hyperparameters["return_intermediate"] is new_ret_intmdt
            assert new_ev.hyperparameters["complementary"] is new_comp

    @pytest.mark.parametrize("len_t", [3, 8])
    def test_batch_size_with_return_intermediate(self, len_t):
        """Test that the batch size is correctly set for intermediate time values."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        t = np.linspace(0, 1, len_t)
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=coeffs, t=t)
        assert ev.batch_size is None
        ev = ParametrizedEvolution(H=H, params=coeffs, t=t, return_intermediate=True)
        assert ev.batch_size == len_t
        ev = ParametrizedEvolution(
            H=H, params=coeffs, t=t, return_intermediate=True, complementary=True
        )
        assert ev.batch_size == len_t

    def test_warns_with_complementary_without_ret_intermediate(self):
        """Test that a warning is raised if the keyword argument complementary is activated
        without return_intermediate being activated."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        with pytest.warns(UserWarning, match="The keyword argument complementary"):
            ev = ParametrizedEvolution(
                H=H, params=[1, 2], t=2, return_intermediate=False, complementary=True
            )

        assert ev.hyperparameters["return_intermediate"] is False
        assert ev.hyperparameters["complementary"] is True

    def test_odeint_kwargs(self):
        """Test the initialization with odeint kwargs."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=2, mxstep=10, hmax=1, atol=1e-3, rtol=1e-6)

        assert ev.odeint_kwargs == {"mxstep": 10, "hmax": 1, "atol": 1e-3, "rtol": 1e-6}

    def test_update_attributes(self):
        """Test that the ``ParametrizedEvolution`` attributes can be updated
        using the ``__call__`` method."""
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

    @pytest.mark.parametrize("time_interface", ["jax", "python", "numpy"])
    def test_list_of_times(self, time_interface):
        """Test the initialization."""
        import jax.numpy as jnp

        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        t = {
            "jax": jnp.arange(0, 10, 0.01),
            "python": list(np.arange(0, 10, 0.01)),
            "numpy": np.arange(0, 10, 0.01),
        }[time_interface]
        ev = ParametrizedEvolution(H=H, params=[1, 2], t=t)
        exp_time_type = {"jax": jnp.ndarray, "python": qml.numpy.ndarray, "numpy": np.ndarray}

        assert isinstance(ev.t, exp_time_type[time_interface])
        assert qml.math.allclose(ev.t, t)

    def test_has_matrix(self):
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

    def test_hash_with_data(self):
        """Test that the hash of a ParametrizedEvolution takes all attributes into account."""

        H_0 = 0.2 * qml.PauliZ(0) + qml.pulse.constant * (qml.PauliX(0) @ qml.PauliY(1))
        H_1 = 0.2 * qml.PauliX(0) + qml.pulse.constant * (qml.PauliX(0) @ qml.PauliY(1))

        params_0 = [np.array(0.4)]
        params_1 = [np.array(0.43)]

        t_0 = (0.3, 0.4)
        t_1 = (0.3, 0.5)

        atol_0 = 1e-8
        atol_1 = 1e-7

        compare_to = ParametrizedEvolution(H_0, params_0, t_0, False, False, atol=atol_0)
        equal = ParametrizedEvolution(H_0, params_0, t_0, False, False, atol=atol_0)
        diff_H = ParametrizedEvolution(H_1, params_0, t_0, False, False, atol=atol_0)
        diff_params = ParametrizedEvolution(H_0, params_1, t_0, False, False, atol=atol_0)
        diff_t = ParametrizedEvolution(H_0, params_0, t_1, False, False, atol=atol_0)
        diff_atol = ParametrizedEvolution(H_0, params_0, t_0, False, False, atol=atol_1)
        diff_ret_intmdt = ParametrizedEvolution(H_0, params_0, t_0, True, False, atol=atol_0)
        diff_complementary = ParametrizedEvolution(H_0, params_0, t_0, False, True, atol=atol_0)

        assert compare_to.hash == equal.hash
        assert compare_to.hash != diff_H.hash
        assert compare_to.hash != diff_params.hash
        assert compare_to.hash != diff_t.hash
        assert compare_to.hash != diff_atol.hash
        assert compare_to.hash != diff_ret_intmdt.hash
        assert compare_to.hash != diff_complementary.hash


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

    @pytest.mark.parametrize("comp", [False, True])
    @pytest.mark.parametrize("len_t", [2, 6])
    def test_return_intermediate_and_complementary(self, comp, len_t):
        """Test that intermediate time evolution matrices are returned."""
        import jax
        from jax import numpy as jnp

        jax.config.update("jax_enable_x64", True)

        H = time_independent_hamiltonian()
        t = np.linspace(0.4, 0.7, len_t)
        params = [1, 2]
        ev = ParametrizedEvolution(
            H=H, params=params, t=t, return_intermediate=True, complementary=comp, rtol=1e-10
        )
        matrices = ev.matrix()
        assert isinstance(matrices, jnp.ndarray)
        assert matrices.shape == (len_t, 4, 4)

        H_mat = qml.matrix(H(params, t=t[-1]))
        if comp:
            true_matrices = [qml.math.expm(-1j * H_mat * (t[-1] - _t)) for _t in t]
        else:
            true_matrices = [qml.math.expm(-1j * H_mat * (_t - t[0])) for _t in t]
        assert qml.math.allclose(matrices, true_matrices, atol=1e-6, rtol=0.0)


@pytest.mark.jax
class TestIntegration:
    """Integration tests for the ParametrizedEvolution class."""

    @pytest.mark.parametrize("time", [0.3, 1, [0, 2], [0.4, 2], (3, 3.1)])
    @pytest.mark.parametrize("time_interface", ["python", "numpy", "jax"])
    @pytest.mark.parametrize("use_jit", [False, True])
    def test_time_input_formats(self, time, time_interface, use_jit):
        import jax
        import jax.numpy as jnp

        if time_interface == "jax":
            time = jnp.array(time)
        elif time_interface == "numpy":
            time = np.array(time)
        H = qml.pulse.ParametrizedHamiltonian([2], [qml.PauliX(0)])

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(t):
            qml.evolve(H)([], t)
            return qml.expval(qml.PauliZ(0))

        if use_jit:
            circuit = jax.jit(circuit)

        res = circuit(time)
        duration = time if qml.math.ndim(time) == 0 else time[1] - time[0]
        assert qml.math.isclose(res, qml.math.cos(4 * duration))

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

    def test_mixed_device(self):
        """Test mixed device integration matches that of default qubit"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        mixed = qml.device("default.mixed", wires=range(3))
        default = qml.device("default.qubit", wires=range(3))

        coeff = [qml.pulse.pwc(5.0), qml.pulse.pwc(5.0)]
        ops = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(1) @ qml.PauliY(2)]
        H_pulse = qml.dot(coeff, ops)

        def circuit(x):
            qml.pulse.ParametrizedEvolution(H_pulse, x, 5.0)
            return qml.expval(qml.PauliZ(0))

        qnode_def = qml.QNode(circuit, default, interface="jax")
        qnode_mix = qml.QNode(circuit, mixed, interface="jax")

        x = [jnp.arange(3, dtype=float)] * 2
        res_def = qnode_def(x)
        grad_def = jax.grad(qnode_def)(x)

        res_mix = qnode_mix(x)
        grad_mix = jax.grad(qnode_mix)(x)

        assert qml.math.isclose(res_def, res_mix, atol=1e-4)
        assert qml.math.allclose(grad_def, grad_mix, atol=1e-4)

    def test_jitted_unitary_differentiation_sparse(
        self,
    ):
        """Test that the unitary can be differentiated with and without jitting using sparse matrices"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        def U(params):
            H = jnp.polyval * qml.PauliZ(0)
            Um = qml.evolve(H)(params, t=10.0, dense=False)
            return qml.matrix(Um)

        params = jnp.array([[0.5]], dtype=complex)
        jac = jax.jacobian(U, holomorphic=True)(params)
        jac_jit = jax.jacobian(jax.jit(U), holomorphic=True)(params)

        assert qml.math.allclose(jac, jac_jit)

    def test_jitted_unitary_differentiation_dense(
        self,
    ):
        """Test that the unitary can be differentiated with and without jitting using dense matrices"""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        def U(params):
            H = jnp.polyval * qml.PauliZ(0)
            Um = qml.evolve(H)(params, t=10.0, dense=True)
            return qml.matrix(Um)

        params = jnp.array([[0.5]], dtype=complex)
        jac = jax.jacobian(U, holomorphic=True)(params)
        jac_jit = jax.jacobian(jax.jit(U), holomorphic=True)(params)

        assert qml.math.allclose(jac, jac_jit)
