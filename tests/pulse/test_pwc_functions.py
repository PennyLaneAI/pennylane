# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the convenience functions used in pulsed programming.
"""
# pylint: disable=import-outside-toplevel
import inspect
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse.parametrized_hamiltonian import ParametrizedHamiltonian


# error expected to be raised locally - test will pass in CI, where it will be run without jax installed
def test_error_raised_if_jax_not_installed():
    """Test that an error is raised if a convenience function is called without jax installed"""
    try:
        import jax  # pylint: disable=unused-import

        pytest.skip()
    except ImportError:
        with pytest.raises(ImportError, match="Module jax is required"):
            qml.pulse.pwc(10)
        with pytest.raises(ImportError, match="Module jax is required"):
            qml.pulse.pwc_from_function(10, 10)


@pytest.mark.jax
class TestPWC:
    """Unit tests for the pwc function"""

    def test_pwc_returns_callable(self):
        """Tests that the pwc function returns a callable with arguments (params, t)"""
        c = qml.pulse.pwc(10)
        argspec = inspect.getfullargspec(c)

        assert callable(c)
        assert argspec.args == ["params", "t"]

    def test_t_out_of_bounds_returns_0(self):
        """Tests that requesting a value for the pwc function outside the defined window returns 0"""
        f_pwc = qml.pulse.pwc(timespan=(1, 3))
        constants = np.linspace(0, 12, 13)

        assert f_pwc(constants, 1.5) != 0
        assert f_pwc(constants, 0) == 0
        assert f_pwc(constants, 4) == 0

    def test_bins_match_params_array(self):
        """Test the pwc function contains bins matching the array of constants passed as params"""
        f_pwc = qml.pulse.pwc(timespan=(1, 3))
        constants = np.linspace(0, 12, 13)

        y = [float(f_pwc(constants, i)) for i in np.linspace(1, 3, 100)]
        assert set(y) == set(constants)

    def test_t_input_types(self):
        """Tests that both shapes for input t work when creating a pwc function"""
        constants = np.linspace(0, 12, 13)

        # should be identical
        f1 = qml.pulse.pwc(10)
        f2 = qml.pulse.pwc((0, 10))

        assert np.all([f1(constants, t) == f2(constants, t) for t in np.linspace(-2, 12, 200)])
        assert f1(constants, 10) == 0
        assert f2(constants, 10) == 0
        assert f1(constants, 0) == constants[0]
        assert f2(constants, 0) == constants[0]

        # should set t1=1 instead of default t1=0
        f3 = qml.pulse.pwc((1, 3))
        assert f3(constants, 3) == 0
        assert f3(constants, 1) == constants[0]
        assert f3(constants, 0) == 0

    def test_function_call_is_jittable(self):
        """Test that jax.jit can be used on the callable produced by pwc_from_function"""
        import jax

        f = qml.pulse.pwc(10)
        assert jax.jit(f)([1.2, 2.3], 2) != 0
        assert jax.jit(f)([1.2, 2.3], 13) == 0


@pytest.mark.jax
class TestPWC_from_function:
    """Unit tests for the pwc_from_function decorator"""

    def test_pwc_from_function_returns_callable(self):
        """Tests that the pwc function returns a callable with arguments (fn), which if
        passed in turn returns a callable with arguments (params, t)"""

        def f(x):
            return x**2

        c1 = qml.pulse.pwc_from_function(10, 10)
        c2 = c1(f)

        argspec1 = inspect.getfullargspec(c1)
        argspec2 = inspect.getfullargspec(c2)

        assert callable(c1)
        assert callable(c2)
        assert argspec1.args == ["fn"]
        assert argspec2.args == ["params", "t"]

    def test_use_as_decorator_returns_callable(self):
        """Test that decorating a function with pwc_from_function returns a callable with arguments (params, t)"""

        @qml.pulse.pwc_from_function(9, 10)
        def f(param, t):
            return t**2 + param

        argspec = inspect.getfullargspec(f)

        assert callable(f)
        assert argspec.args == ["params", "t"]

    def test_expected_values_are_returned(self):
        """Test that output values for the pwc version functions and the inital function match each other
        when t is one of the input time_bins"""

        def f_initial(param, t):
            return t**2 + param

        f_pwc = qml.pulse.pwc_from_function(timespan=9, num_bins=10)(f_initial)

        @qml.pulse.pwc_from_function(timespan=9, num_bins=10)
        def f_decorated(param, t):
            return t**2 + param

        time_bins = np.linspace(0, 9, 10)
        for i in time_bins:
            # pwc functions should match initial function between times i-1 and i,
            # i.e. points just before time i should always match
            assert f_initial(2, i) == f_pwc(2, i * 0.99)
            assert f_initial(2, i) == f_decorated(
                2, i * 0.99
            )  # 0.99*i because for edge point, f_pwc(i) is 0

    @pytest.mark.parametrize("num_bins", [10, 15, 21])
    def test_num_bins_is_correct(self, num_bins):
        """Test the pwc function has been divided into the expected number of bins"""

        def f_initial(param, t):
            return t + param

        f_pwc = qml.pulse.pwc_from_function(timespan=9, num_bins=num_bins)(f_initial)

        # check that there are only a limited number of unique output values for the pwc function
        y = [float(f_pwc(2, i)) for i in np.linspace(0, 9, 1000)]

        assert len(set(y)) == num_bins + 1  # all bins plus 0 at the edges

    def test_t_out_of_bounds_returns_0(self):
        """Tests that requesting a value for the pwc function outside the defined window returns 0"""

        def f_initial(param, t):
            return t + param

        f_pwc = qml.pulse.pwc_from_function(timespan=(1, 3), num_bins=10)(f_initial)

        assert f_pwc(3, 1.5) != 0
        assert f_pwc(3, 0) == 0
        assert f_pwc(3, 4) == 0

    def test_t_input_types(self):
        """Tests that both shapes for input t work when creating a pwc function"""

        def f(params, t):
            return params[1] * t**2 + params[0]

        params = [1.2, 2.3]

        # should be identical, t1=0, t2=10
        f1 = qml.pulse.pwc_from_function(10, 12)(f)
        f2 = qml.pulse.pwc_from_function((0, 10), 12)(f)

        assert np.all([f1(params, t) == f2(params, t) for t in np.linspace(-2, 12, 200)])
        assert f1(params, 10) == 0
        assert f2(params, 10) == 0
        assert f1(params, 0) == params[0]
        assert f2(params, 0) == params[0]

        # should set t1=1 instead of default t1=0
        f3 = qml.pulse.pwc_from_function((1, 3), 12)(f)
        assert f3(params, 3) == 0
        assert f3(params, 1) == f(params, 1)
        assert f3(params, 0) == 0

    def test_function_call_is_jittable(self):
        """Test that jax.jit can be used on the callable produced by pwc_from_function"""
        import jax

        @qml.pulse.pwc_from_function((1, 3), 12)
        def f(params, t):
            return params[1] * t**2 + params[0]

        assert jax.jit(f)([1.2, 2.3], 0) == 0
        assert jax.jit(f)([1.2, 2.3], 2) != 0
        assert jax.jit(f)([1.2, 2.3], 4) == 0


@pytest.mark.jax
class TestIntegration:
    """Test integration of pwc functions with the pulse module."""

    def integral_pwc(  # pylint:disable = too-many-arguments
        self, t1, t2, num_bins, integration_bounds, fn, params, pwc_from_function=False
    ):
        """Helper function that integrates a pwc function."""
        from jax import numpy as jnp

        # constants set by array if pwc, constants must be found if pwc_from_function
        constants = jnp.array(params)
        if pwc_from_function:
            time_bins = np.linspace(t1, t2, num_bins)
            constants = jnp.array(list(fn(params, time_bins)) + [0])

        # get start and end point as indicies, without casting to int
        start = num_bins / (t2 - t1) * (integration_bounds[0] - t1)
        end = num_bins / (t2 - t1) * (integration_bounds[1] - t1)

        # get indices of bins that are completely within the integration window
        complete_indices = np.linspace(
            int(start) + 1, int(end) - 1, int(end) - int(start) - 1, dtype=int
        )
        relevant_indices = np.array([i for i in complete_indices if -1 < i < num_bins])
        # find area contributed by bins that are completely within the integration window
        bin_width = (t2 - t1) / num_bins
        main_area = np.sum(constants[relevant_indices] * bin_width)

        # if start index is not out of range, add contribution from partial bin at start
        if start >= 0:
            width = bin_width * 1 - (start - int(start))
            main_area += constants[int(start)] * width

        # if end index is not out of range, add contribution from partial bin at end
        if end < num_bins:
            width = bin_width * (end - int(end))
            main_area += constants[int(end)] * width

        return main_area

    def test_parametrized_hamiltonian_with_pwc(self):
        """Test that a pwc function can be used to create a ParametrizedHamiltonian"""

        f1 = qml.pulse.pwc((1, 6))
        f2 = qml.pulse.pwc((0.5, 3))
        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

        constants = np.linspace(0, 9, 10)

        assert isinstance(H, ParametrizedHamiltonian)

        # at t=7 and t=0.2, both terms are 0
        assert qml.math.allequal(qml.matrix(H(params=[constants, constants], t=7)), 0)
        assert qml.math.allequal(qml.matrix(H(params=[constants, constants], t=0.2)), 0)

        # at t=4, only term 1 is non-zero
        true_mat = qml.matrix(f1(constants, 4) * qml.PauliX(0), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params=[constants, constants], t=4)), true_mat)

        # at t=0.7, only term 2 is non-zero
        true_mat = qml.matrix(f2(constants, 0.7) * qml.PauliY(1), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params=[constants, constants], t=0.7)), true_mat)

        # at t=1.5, both are non-zero and output is as expected
        true_mat = qml.matrix(f1(constants, 1.5) * qml.PauliX(0), wire_order=[0, 1]) + qml.matrix(
            f2(constants, 1.5) * qml.PauliY(1), wire_order=[0, 1]
        )
        assert qml.math.allequal(qml.matrix(H(params=[constants, constants], t=1.5)), true_mat)

    @pytest.mark.slow
    def test_qnode_pwc(self):
        """Test that the evolution of a parametrized hamiltonian defined with a pwc function be executed on a QNode."""
        import jax

        f1 = qml.pulse.pwc((1, 6))
        f2 = qml.pulse.pwc((0.5, 3))
        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

        t = (0, 4)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev, interface="jax")
        def true_circuit(params):
            # ops X0 and Y1 are commuting - time evolution of f1*X0 + f2*X1 is exp(-i*F1*X0)exp(-i*F2*Y1)
            # Where Fj = integral of fj(p,t)dt over evolution time t
            coeff1 = partial(self.integral_pwc, 1, 6, 10, (0, 4), f1)
            coeff2 = partial(self.integral_pwc, 0.5, 3, 10, (0, 4), f2)
            qml.prod(
                qml.exp(qml.PauliX(0), -1j * coeff1(params[0])),
                qml.exp(qml.PauliY(1), -1j * coeff2(params[1])),
            )
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        constants = np.linspace(0, 9, 10)
        params = [constants, constants]

        assert qml.math.allclose(circuit(params), true_circuit(params), atol=5e-3)
        assert qml.math.allclose(
            jax.grad(circuit)(params), jax.grad(true_circuit)(params), atol=5e-3
        )

    def test_qnode_pwc_jit(self):
        """Test that the evolution of a parametrized hamiltonian defined with a pwc function can executed on
        a QNode using jax-jit, and the results don't differ from execution without jitting."""
        import jax

        f1 = qml.pulse.pwc((1, 6))
        f2 = qml.pulse.pwc((0.5, 3))
        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

        t = (0, 4)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def jitted_circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        constants = np.linspace(0, 9, 10)
        params = [constants, constants]

        assert qml.math.allclose(jitted_circuit(params), circuit(params), atol=5e-3)
        assert qml.math.allclose(
            jax.grad(jitted_circuit)(params), jax.grad(circuit)(params), atol=5e-3
        )

    def test_parametrized_hamiltonian_with_pwc_from_function(self):
        """Test that a function decorated by pwc_from_function can be used to create a ParametrizedHamiltonian"""

        @qml.pulse.pwc_from_function((2, 5), 20)
        def f1(params, t):
            return params + t

        @qml.pulse.pwc_from_function((3, 7), 10)
        def f2(params, t):
            return params[0] + params[1] * t**2

        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)
        params = [1.2, [2.3, 3.4]]

        assert isinstance(H, ParametrizedHamiltonian)

        # at t=8 and t=1, both terms are 0
        assert qml.math.allequal(qml.matrix(H(params, t=8)), 0)
        assert qml.math.allequal(qml.matrix(H(params, t=1)), 0)

        # at t=2.5, only term 1 is non-zero
        true_mat = qml.matrix(f1(params[0], 2.5) * qml.PauliX(0), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params, t=2.5)), true_mat)

        # # at t=6, only term 2 is non-zero
        true_mat = qml.matrix(f2(params[1], 6) * qml.PauliY(1), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params, t=6)), true_mat)
        #
        # # at t=4, both are non-zero and output is as expected
        true_mat = qml.matrix(f1(params[0], 4) * qml.PauliX(0), wire_order=[0, 1]) + qml.matrix(
            f2(params[1], 4) * qml.PauliY(1), wire_order=[0, 1]
        )
        assert qml.math.allequal(qml.matrix(H(params, t=4)), true_mat)

    def test_qnode_pwc_from_function(self):
        """Test that the evolution of a ParametrizedHamiltonian defined with a function decorated by pwc_from_function
        can be executed on a QNode."""
        import jax

        @qml.pulse.pwc_from_function((2, 5), 20)
        def f1(params, t):
            return params + t

        @qml.pulse.pwc_from_function((3, 7), 10)
        def f2(params, t):
            return params[0] + params[1] * t**2

        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

        t = (1, 4)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev, interface="jax")
        def true_circuit(params):
            # ops X0 and Y1 are commuting - time evolution of f1*X0 + f2*X1 is exp(-i*F1*X0)exp(-i*F2*Y1)
            # Where Fj = integral of fj(p,t)dt over evolution time t
            coeff1 = partial(self.integral_pwc, 2, 5, 20, (1, 4), f1)
            coeff2 = partial(self.integral_pwc, 3, 7, 10, (1, 4), f2)
            qml.prod(
                qml.exp(qml.PauliX(0), -1j * coeff1(params[0], pwc_from_function=True)),
                qml.exp(qml.PauliY(1), -1j * coeff2(params[1], pwc_from_function=True)),
            )
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = [1.2, [2.3, 3.4]]

        circuit_grad_flattened, _ = jax.flatten_util.ravel_pytree(jax.grad(circuit)(params))
        true_grad_flattened, _ = jax.flatten_util.ravel_pytree(jax.grad(true_circuit)(params))

        assert qml.math.allclose(circuit(params), true_circuit(params), atol=5e-2)
        assert qml.math.allclose(circuit_grad_flattened, true_grad_flattened, atol=5e-2)

    def test_qnode_pwc_from_function_jit(self):
        """Test that the evolution of a ParametrizedHamiltonian defined with a function decorated by pwc_from_function
        can be executed on a QNode using jax-jit, and the results don't differ from an execution without jitting.
        """
        import jax

        @qml.pulse.pwc_from_function((2, 5), 20)
        def f1(params, t):
            return params + t

        @qml.pulse.pwc_from_function((3, 7), 10)
        def f2(params, t):
            return params[0] + params[1] * t**2

        H = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

        t = (1, 4)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def jitted_circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = [1.2, [2.3, 3.4]]

        circuit_grad_flattened, _ = jax.flatten_util.ravel_pytree(jax.grad(circuit)(params))
        jitted_grad_flattened, _ = jax.flatten_util.ravel_pytree(jax.grad(jitted_circuit)(params))

        assert qml.math.allclose(jitted_circuit(params), circuit(params), atol=5e-2)
        assert qml.math.allclose(circuit_grad_flattened, jitted_grad_flattened, atol=5e-2)
