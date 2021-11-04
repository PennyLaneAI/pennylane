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
Tests for the Fourier reconstruction transform.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.reconstruct import (
    _reconstruct_equ,
    _reconstruct_gen,
    _prepare_jobs,
    reconstruct,
)

dev = qml.device("default.qubit", wires=1)

class Lambda:
    def __init__(self, fun):
        self.fun = fun
    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

@qml.qnode(dev)
def dummy_qnode(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

def fun_close(fun1, fun2, zero=None, tol=1e-5):
    X = np.linspace(-np.pi, np.pi, 100)
    if zero is not None:
        X = qml.math.convert_like(X, zero)
        
    for x in X:
        if not np.isclose(fun1(x), fun2(x), atol=tol, rtol=0):
            print(fun1(x), fun2(x), " at ", x, type(x))
            return False
    return True

class TestErrors:
    """Test that errors are raised e.g. for invalid inputs."""

    def test_nums_frequency_and_spectra_missing(self):
        """Tests that an error is raised if neither information about the number
        of frequencies nor about the spectrum is given to ``reconstruct``."""
        with pytest.raises(ValueError, match="Either nums_frequency or spectra must be given."):
            reconstruct(dummy_qnode)

    @pytest.mark.parametrize("num_frequency", [-3, -9.2, 0.999])
    def test_nums_frequency_and_spectra_missing(self, num_frequency):
        """Tests that an error is raised if ``_reconstruct_equ`` receives a
        negative or non-integer ``num_frequency`` ."""
        with pytest.raises(ValueError, match="num_frequency must be a non-negative integer"):
            _reconstruct_equ(dummy_qnode, num_frequency=num_frequency)

class TestReconstructEqu:
    """Tests the one-dimensional reconstruction subroutine based on equidistantly
    shifted evaluations for equidistant frequencies."""
    
    c_funs = [
        lambda x: 13.71 * np.sin(x) - np.cos(2*x)/30,
        lambda x: -0.49 * np.sin(3.2 * x),
        lambda x: 0.1 * np.cos(-2.1 * x) + 2.9 * np.sin(4.2*x-1.2),
        lambda x: 4.01,
        lambda x: np.sum([i**2*0.1*np.sin(i*3.921*x-2.7/i) for i in range(1, 10)])
    ]

    nums_frequency = [2, 1, 2, 0, 9]
    base_frequencies = [1, 3.2, 2.1, 1, 3.921]
    expected_grads = [
        lambda x: 13.71 * np.cos(x) + 2*np.sin(2*x)/30,
        lambda x: -0.49 * np.cos(3.2 * x) * 3.2,
        lambda x: (-2.1) * (-0.1) * np.sin(-2.1 * x) + 4.2 * 2.9 * np.cos(4.2*x-1.2),
        lambda x: 0.,
        lambda x: np.sum([i*3.921*i**2*0.1*np.cos(i*3.921*x-2.7/i) for i in range(1, 10)])
    ]


    @pytest.mark.parametrize("fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies))
    def test_with_classical_fun(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly (via rescaling to integer frequencies)."""
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x/f0)
        _rec = qml.fourier.reconstruct._reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0*x)
        assert spy.call_count==num_frequency*2+1
        assert fun_close(fun, rec)

        # Repeat, using precomputed fun_at_zero
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        fun_at_zero = _fun(0.)
        _rec = qml.fourier.reconstruct._reconstruct_equ(_fun, num_frequency, fun_at_zero=fun_at_zero)
        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0*x)
        assert spy.call_count==num_frequency*2+1
        assert fun_close(fun, rec)


    @pytest.mark.parametrize("fun, num_frequency, f0, expected_grad", zip(c_funs, nums_frequency, base_frequencies, expected_grads))
    def test_differentiability_autograd(self, fun, num_frequency, f0, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable in the framework of their input variable."""
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x/f0)
        _rec = qml.fourier.reconstruct._reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0*x)
        grad = qml.grad(rec)
        print(grad(pnp.array(0.1, requires_grad=True)))
        assert fun_close(expected_grad, grad, zero=pnp.array(0.0, requires_grad=True))

    @pytest.mark.parametrize("fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies))
    def test_with_classical_fun_num_freq_too_small(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed wrongly if num_frequency is too small."""
        if num_frequency==0:
            pytest.skip("Can't reduce the number of frequencies below 0.")
        num_frequency -= 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x/f0)
        _rec = qml.fourier.reconstruct._reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0*x)
        assert spy.call_count==num_frequency*2+1
        assert not fun_close(fun, rec)

    @pytest.mark.parametrize("fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies))
    def test_with_classical_fun_num_freq_too_large(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly if num_frequency is too large."""
        num_frequency += 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x/f0)
        _rec = qml.fourier.reconstruct._reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0*x)
        assert spy.call_count==num_frequency*2+1
        assert fun_close(fun, rec)

    def get_circuit_equ(self, scales):
        """Generate a circuit with Pauli gates with ``f*x`` as argument where
        the ``f`` s are stored in ``scales`` ."""
        @qml.qnode(dev)
        def circuit_equ(x):
            for f in scales:
                qml.RX(f*x, wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit_equ

    all_scales = [
        [1],
        [0],
        [1, 1],
        [1, 2],
        [1, 5],
        [1, 2, 10],
    ]

    @pytest.mark.parametrize("scales", all_scales)
    def test_with_qnode(self, scales, mocker):
        """Test that integer-frequency qnodes are reconstructed correctly."""
        circuit_equ = self.get_circuit_equ(scales)
        Fun = Lambda(circuit_equ)
        spy = mocker.spy(Fun, "fun")
        num_frequency = sum(scales)
        rec = qml.fourier.reconstruct._reconstruct_equ(Fun, num_frequency)
        assert spy.call_count==num_frequency*2+1
        assert fun_close(circuit_equ, rec)

        # Repeat, using precomputed fun_at_zero
        Fun = Lambda(circuit_equ)
        spy = mocker.spy(Fun, "fun")
        fun_at_zero = circuit_equ(0.)
        rec = qml.fourier.reconstruct._reconstruct_equ(Fun, num_frequency, fun_at_zero=fun_at_zero)
        assert fun_close(circuit_equ, rec)

