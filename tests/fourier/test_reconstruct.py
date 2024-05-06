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
from functools import reduce

# pylint: disable=too-many-arguments,too-few-public-methods, unnecessary-lambda-assignment, consider-using-dict-items
from inspect import signature
from itertools import chain

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.reconstruct import (
    _prepare_jobs,
    _reconstruct_equ,
    _reconstruct_gen,
    reconstruct,
)
from pennylane.fourier.utils import join_spectra

dev_0 = qml.device("default.qubit", wires=1)


class Lambda:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


@qml.qnode(dev_0)
def dummy_qnode(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


def get_RX_circuit(scales):
    """Generate a circuit with Pauli-X rotation gates with ``f*x``
    as argument where the ``f`` s are stored in ``scales`` ."""

    @qml.qnode(dev_0)
    def circuit(x):
        for f in scales:
            qml.RX(f * x, wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


def fun_close(fun1, fun2, zero=None, tol=1e-5, samples=10):
    X = np.linspace(-np.pi, np.pi, samples)
    if zero is not None:
        X = qml.math.convert_like(X, zero)

    for x in X:
        if not np.isclose(fun1(x), fun2(x), atol=tol, rtol=0):
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
    def test_num_frequency_invalid(self, num_frequency):
        """Tests that an error is raised if ``_reconstruct_equ`` receives a
        negative or non-integer ``num_frequency`` ."""
        with pytest.raises(ValueError, match="num_frequency must be a non-negative integer"):
            _reconstruct_equ(dummy_qnode, num_frequency=num_frequency)

    @pytest.mark.parametrize(
        "spectra, shifts",
        [
            ({"x": {(): [0.0, 1.0]}}, {"x": {(): [0.3]}}),
            ({"x": {(): [0.0, 1.0, 2.0]}}, {"x": {(): list(range(20))}}),
        ],
    )
    def test_wrong_number_of_shifts(self, spectra, shifts):
        """Tests that an error is raised if the number of provided shifts does not match."""
        with pytest.raises(ValueError, match="The number of provided shifts"):
            reconstruct(dummy_qnode, spectra=spectra, shifts=shifts)


class TestWarnings:
    """Test that warnings are raised e.g. for an ill-conditioned
    Fourier transform during the reconstruction."""

    def test_ill_conditioned(self):
        """Test that a warning is raised for an ill-conditioned matrix in the Fourier trafo."""
        shifts = [-np.pi / 2 - 1e-9, -np.pi / 2, 0, np.pi / 2, np.pi / 2 + 1e-9]
        with pytest.warns(UserWarning, match="condition number of the Fourier"):
            _reconstruct_gen(dummy_qnode, spectrum=[1.0, 2.0], shifts=shifts)


class TestReconstructEqu:
    """Tests the one-dimensional reconstruction subroutine based on equidistantly
    shifted evaluations for equidistant frequencies."""

    c_funs = [
        lambda x: 13.71 * qml.math.sin(x) - qml.math.cos(2 * x) / 30,
        lambda x: -0.49 * qml.math.sin(3.2 * x),
        lambda x: 0.1 * qml.math.cos(-2.1 * x) + 2.9 * qml.math.sin(4.2 * x - 1.2),
        lambda x: qml.math.ones_like(x) * 4.01,
        lambda x: qml.math.sum(
            [i**2 * 0.1 * qml.math.sin(i * 3.921 * x - 2.7 / i) for i in range(1, 10)]
        ),
    ]

    nums_frequency = [2, 1, 2, 0, 9]
    base_frequencies = [1.0, 3.2, 2.1, 1.0, 3.921]
    expected_grads = [
        lambda x: 13.71 * qml.math.cos(x) + 2 * qml.math.sin(2 * x) / 30,
        lambda x: -0.49 * qml.math.cos(3.2 * x) * 3.2,
        lambda x: (-2.1) * (-0.1) * qml.math.sin(-2.1 * x)
        + 4.2 * 2.9 * qml.math.cos(4.2 * x - 1.2),
        lambda x: 0.0,
        lambda x: qml.math.sum(
            [i * 3.921 * i**2 * 0.1 * qml.math.cos(i * 3.921 * x - 2.7 / i) for i in range(1, 10)]
        ),
    ]

    @pytest.mark.parametrize(
        "fun, num_frequency, base_f", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun(self, fun, num_frequency, base_f, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly (via rescaling to integer frequencies)."""
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        assert spy.call_count == num_frequency * 2 + 1
        assert fun_close(fun, rec)

        # Repeat, using precomputed f0
        f0 = _fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        _rec = _reconstruct_equ(_fun, num_frequency, f0=f0)
        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        assert spy.call_count == num_frequency * 2
        assert fun_close(fun, rec)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "fun, num_frequency, base_f, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_autograd(self, fun, num_frequency, base_f, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Autograd input variables."""
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency, interface="autograd")

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        grad = qml.grad(rec)

        assert fun_close(fun, rec, zero=pnp.array(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=pnp.array(0.0, requires_grad=True))

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "fun, num_frequency, base_f, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_jax(self, fun, num_frequency, base_f, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for JAX input variables."""
        import jax

        jax.config.update("jax_enable_x64", True)

        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency, interface="jax")

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        grad = jax.grad(rec)
        assert fun_close(fun, rec, zero=jax.numpy.array(0.0))
        assert fun_close(expected_grad, grad, zero=jax.numpy.array(0.0))

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "fun, num_frequency, base_f, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_tensorflow(self, fun, num_frequency, base_f, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for TensorFlow input variables."""
        import tensorflow as tf

        # Convert fun to have integer frequencies
        base_f = tf.constant(base_f, dtype=tf.float64)
        _fun = lambda x: fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency, interface="tensorflow")

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)

        def grad(arg):
            arg = tf.Variable(arg)
            with tf.GradientTape() as tape:
                out = rec(arg)
            return tape.gradient(out, arg)

        assert fun_close(fun, rec, zero=tf.Variable(0.0, dtype=tf.float64))
        assert fun_close(expected_grad, grad, zero=tf.Variable(0.0, dtype=tf.float64))

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "fun, num_frequency, base_f, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_torch(self, fun, num_frequency, base_f, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Torch input variables."""
        import torch

        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency, interface="torch")

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        grad = lambda x: torch.autograd.functional.jacobian(rec, x)
        assert fun_close(fun, rec, zero=torch.tensor(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=torch.tensor(0.0, requires_grad=True))

    @pytest.mark.parametrize(
        "fun, num_frequency, base_f", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun_num_freq_too_small(self, fun, num_frequency, base_f, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed wrongly if num_frequency is too small."""
        if num_frequency == 0:
            pytest.skip("Can't reduce the number of frequencies below 0.")
        num_frequency -= 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        assert spy.call_count == num_frequency * 2 + 1
        assert not fun_close(fun, rec)

    @pytest.mark.parametrize(
        "fun, num_frequency, base_f", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun_num_freq_too_large(self, fun, num_frequency, base_f, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly if num_frequency is too large."""
        num_frequency += 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / base_f)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(base_f * x)
        assert spy.call_count == num_frequency * 2 + 1
        assert fun_close(fun, rec)

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
        circuit = get_RX_circuit(scales)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        num_frequency = sum(scales)
        rec = _reconstruct_equ(Fun, num_frequency)
        assert spy.call_count == num_frequency * 2 + 1
        assert fun_close(circuit, rec)

        # Repeat, using precomputed f0
        f0 = circuit(0.0)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_equ(Fun, num_frequency, f0=f0)
        assert spy.call_count == num_frequency * 2
        assert fun_close(circuit, rec)


class TestReconstructGen:
    """Tests the one-dimensional reconstruction subroutine based on arbitrary
    shifted evaluations for arbitrary frequencies."""

    c_funs = [
        lambda x: -3.27 * np.sin(0.1712 * x) - np.cos(20.812 * x) / 23,
        lambda x: -0.49 * np.sin(3.2 * x),
        lambda x: 0.1 * np.cos(-0.1 * x) + 2.9 * np.sin(0.3 * x - 1.2),
        lambda x: np.sum([np.sin(i * x) for i in range(1, 10)]),
        lambda x: np.sum(
            [i**0.9 * 0.2 * np.sin(i**1.2 * 3.921 * x - 5.1 / i) for i in np.arange(1, 10)]
        ),
    ]

    spectra = [
        [0.1712, 20.812],
        [3.2],
        [-0.3, -0.1, 0.0, 0.1, 0.3],
        list(np.arange(1, 10)),
        [3.921 * i**1.2 for i in np.arange(1, 10)],
    ]

    expected_grads = [
        lambda x: -3.27 * np.cos(0.1712 * x) * 0.1712 + np.sin(20.812 * x) / 23 * 20.812,
        lambda x: -0.49 * np.cos(3.2 * x) * 3.2,
        lambda x: (-0.1) ** 2 * np.sin(-0.1 * x) + 0.3 * 2.9 * np.cos(0.3 * x - 1.2),
        lambda x: np.sum([i * np.cos(i * x) for i in range(1, 10)]),
        lambda x: np.sum(
            [i**2.1 * 3.921 * 0.2 * np.cos(i**1.2 * 3.921 * x - 5.1 / i) for i in range(1, 10)]
        ),
    ]

    all_shifts = [
        [-np.pi / 3, -np.pi / 20, 0.0, np.pi / 20, np.pi / 3],
        [-0.15, -0.05, 0.05],
        [-2 * np.pi, -np.pi, -0.1, np.pi, 2 * np.pi],
        np.arange(-9, 10) * np.pi / 19,
        np.arange(-9, 10) * np.pi / 19,
    ]

    @pytest.mark.parametrize("fun, spectrum", zip(c_funs, spectra))
    def test_with_classical_fun(self, fun, spectrum, mocker):
        """Test that arbitrary-frequency classical functions are
        reconstructed correctly."""
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2 + 1
        assert fun_close(fun, rec)

        # Repeat, using precomputed f0
        f0 = fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, f0=f0)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2
        assert fun_close(fun, rec)

    @pytest.mark.parametrize("fun, spectrum, shifts", zip(c_funs, spectra, all_shifts))
    def test_with_classical_fun_with_shifts(self, fun, spectrum, shifts, mocker, recwarn):
        """Test that arbitrary-frequency classical functions are
        reconstructed correctly."""
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, shifts=shifts)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2 + 1
        assert fun_close(fun, rec)

        # Repeat, using precomputed f0
        f0 = fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, shifts=shifts, f0=f0)
        if 0.0 in shifts:
            assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2
        else:
            assert len(recwarn) == 1
            assert recwarn[0].category == UserWarning
            assert recwarn[0].message.args[0].startswith("The provided value")
        assert fun_close(fun, rec)

    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_autograd(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Autograd input variables."""
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum, interface="autograd")
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=pnp.array(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=pnp.array(0.0, requires_grad=True))

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_jax(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for JAX input variables."""
        import jax

        jax.config.update("jax_enable_x64", True)

        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum, interface="jax")
        grad = jax.grad(rec)
        assert fun_close(fun, rec, zero=jax.numpy.array(0.0))
        assert fun_close(expected_grad, grad, zero=jax.numpy.array(0.0))

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_tensorflow(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for TensorFlow input variables."""
        import tensorflow as tf

        spectrum = tf.constant(spectrum, dtype=tf.float64)
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum, interface="tensorflow")

        def grad(arg):
            arg = tf.Variable(arg)
            with tf.GradientTape() as tape:
                out = rec(arg)
            return tape.gradient(out, arg)

        assert fun_close(fun, rec, zero=tf.Variable(0.0))
        assert fun_close(expected_grad, grad, zero=tf.Variable(0.0))

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_torch(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Torch input variables."""
        import torch

        spectrum = torch.tensor(spectrum, dtype=torch.float64)
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum, interface="torch")
        grad = lambda x: torch.autograd.functional.jacobian(rec, x)
        assert fun_close(fun, rec, zero=torch.tensor(np.float64(0.0), requires_grad=True))
        assert fun_close(
            expected_grad, grad, zero=torch.tensor(np.float64(0.0), requires_grad=True)
        )

    @pytest.mark.parametrize("fun, spectrum", zip(c_funs, spectra))
    def test_with_classical_fun_spectrum_incomplete(self, fun, spectrum, mocker):
        """Test that arbitrary-frequency classical functions are reconstructed wrongly
        if spectrum does not contain all frequencies."""
        if len(spectrum) <= 1:
            pytest.skip("Can't skip a frequency if len(spectrum)<=1.")
        spectrum = spectrum[:-1]
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum)

        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2 + 1
        assert not fun_close(fun, rec)

    @pytest.mark.parametrize("fun, spectrum", zip(c_funs, spectra))
    def test_with_classical_fun_spectrum_overcomplete(self, fun, spectrum, mocker):
        """Test that arbitrary-frequency classical functions are reconstructed correctly
        if spectrum contains additional frequencies."""
        spectrum = spectrum + [0.4812759, 1.2281]
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(Fun, spectrum)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2 + 1
        assert fun_close(fun, rec)

    all_scales = [
        [1.3],
        [1.02, 1.59],
        [0.08, 20.2],
        [1.2, 5.0001],
        [1.0, 2.0, 3.0],
    ]

    @pytest.mark.parametrize("scales", all_scales)
    def test_with_qnode(self, scales, mocker):
        """Test that arbitrary-frequency qnodes are reconstructed correctly."""
        circuit = get_RX_circuit(scales)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        if len(scales) == 1:
            spectrum = sorted({0.0, scales[0]})
        else:
            _spectra = [{0.0, s} for s in scales]
            spectrum = sorted(reduce(join_spectra, _spectra, {0.0}))

        rec = _reconstruct_gen(Fun, spectrum)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2 + 1
        assert fun_close(circuit, rec)

        # Repeat, using precomputed f0
        f0 = circuit(0.0)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, f0=f0)
        assert spy.call_count == len([f for f in spectrum if f > 0.0]) * 2
        assert fun_close(circuit, rec)


all_ids = [
    None,
    {"x": [0, 1], "y": {1, 5}},
    {"z": (0, 9)},
    ["z", "y", "x", "x"],
    {"x", "z", "y"},
    ("y",),
    "x",
]

all_spectra = [
    {
        "x": {0: [0.0], 1: [4.2, 0.0, 0.2]},
        "y": {3: [0.3, 0.0, 0.2], 1: [0.0, 1.1, 5.2], 5: [0.0, 1.2]},
        "z": {i: [0.0, i * 8.7] for i in range(20)},
    },
]

all_shifts = [
    {
        "x": {0: [-1.3], 1: [1.0, -0.4, 4.2, 2.3, -1.5]},
        "y": {
            3: [0.3 * i + 0.05 for i in range(-2, 3)],
            1: [-1, -0.5, -0.1, 0.1, 0.9],
            5: [-1, -0.5, -0.2],
        },
        "z": {i: [-np.pi / 2, 0.0, np.pi / 2] for i in range(20)},
    },
]

all_nums_frequency = [
    {
        "x": {0: 1, 1: 4},
        "y": {3: 1, 1: 1, 5: 9},
        "z": {i: 2 * i for i in range(20)},
    },
]


class TestPrepareJobs:
    """Tests the subroutine that determines the 1D reconstruction
    jobs to be carried out for a call to ``reconstruct`` ."""

    def nested_dict_ids_match(self, ndict, ids):
        if ids.keys() != ndict.keys():
            return False
        for id_, ids_ in ids.items():
            if list(ids_) != list(ndict[id_].keys()):
                return False
        return True

    def ids_match(self, ids_in, ids_out):
        if isinstance(ids_in, dict):
            return ids_in == ids_out
        return all(id_ in ids_in for id_ in ids_out)

    @pytest.mark.parametrize("ids", all_ids)
    @pytest.mark.parametrize("shifts", all_shifts)
    def test_missing_spectra_and_nums_frequency(self, ids, shifts, tol):
        """Test that an error is raised if both, spectra
        and nums_frequency are missing."""
        with pytest.raises(ValueError, match="Either nums_frequency or spectra"):
            _prepare_jobs(ids, nums_frequency=None, spectra=None, shifts=shifts, atol=tol)

    @pytest.mark.parametrize("ids", all_ids)
    @pytest.mark.parametrize("spectra", all_spectra)
    def test_with_spectra(self, ids, spectra, tol):
        """Test the prepared jobs when using spectra and shifts."""
        ids_, recon_fn, jobs, need_f0 = _prepare_jobs(
            ids,
            nums_frequency=None,
            spectra=spectra,
            shifts=None,
            atol=tol,
        )
        if ids is None:
            assert self.nested_dict_ids_match(spectra, ids_)
        else:
            assert self.ids_match(ids, ids_)

        # Check function to use for 1D reconstructions
        assert recon_fn == _reconstruct_gen

        # Check reconstruction jobs to be run
        assert self.nested_dict_ids_match(jobs, ids_)

        # Check all job details
        for _id, _jobs in jobs.items():
            for idx, job in _jobs.items():
                if len(spectra[_id][idx]) == 1:
                    assert job is None
                    continue
                assert list(job.keys()) == ["shifts", "spectrum"]
                assert job["shifts"] is None
                assert job["spectrum"] == spectra[_id][idx]
        assert need_f0

    @pytest.mark.parametrize("ids", all_ids)
    @pytest.mark.parametrize("spectra", all_spectra)
    @pytest.mark.parametrize("shifts", all_shifts)
    def test_with_spectra_and_shifts(self, ids, spectra, shifts, tol):
        """Test the prepared jobs when using spectra and shifts."""
        ids_, recon_fn, jobs, need_f0 = _prepare_jobs(
            ids,
            nums_frequency=None,
            spectra=spectra,
            shifts=shifts,
            atol=tol,
        )
        if ids is None:
            assert self.nested_dict_ids_match(spectra, ids_)
        else:
            assert self.ids_match(ids, ids_)

        # Check function to use for 1D reconstructions
        assert recon_fn == _reconstruct_gen

        # Check reconstruction jobs to be run
        assert self.nested_dict_ids_match(jobs, ids_)

        # Check all job details
        for _id, _jobs in jobs.items():
            for idx, job in _jobs.items():
                if len(spectra[_id][idx]) == 1:
                    assert job is None
                    continue
                assert list(job.keys()) == ["shifts", "spectrum"]
                assert job["shifts"] == shifts[_id][idx]
                assert job["spectrum"] == spectra[_id][idx]
        # sometimes need fun at zero if general reconstruction is performed
        _all_shifts = chain.from_iterable(
            [
                sum(
                    [
                        __shifts
                        for par_idx, __shifts in _shifts.items()
                        if id_ in ids_ and par_idx in ids_[id_]
                    ],
                    [],
                )
                for id_, _shifts in shifts.items()
            ],
        )
        assert need_f0 == any(np.isclose(_shift, 0.0, atol=tol, rtol=0) for _shift in _all_shifts)

    @pytest.mark.parametrize("ids", all_ids)
    @pytest.mark.parametrize("nums_frequency", all_nums_frequency)
    def test_with_nums_frequency(self, ids, nums_frequency, tol):
        """Test the prepared jobs when using nums_frequency."""

        ids_, recon_fn, jobs, need_f0 = _prepare_jobs(
            ids,
            nums_frequency,
            None,
            None,
            atol=tol,
        )

        # Check ids
        if ids is None:
            assert self.nested_dict_ids_match(nums_frequency, ids_)
        else:
            assert self.ids_match(ids, ids_)

        # Check function to use for 1D reconstructions
        assert recon_fn == _reconstruct_equ

        # Check reconstruction jobs to be run
        assert self.nested_dict_ids_match(jobs, ids_)

        for _id, _jobs in jobs.items():
            for idx, job in _jobs.items():
                if nums_frequency[_id][idx] == 0:
                    assert job is None
                    continue
                assert list(job.keys()) == ["num_frequency"]
                assert job["num_frequency"] == nums_frequency[_id][idx]
        # always need fun at zero if equidistant reconstruction is performed
        assert need_f0


dev_1 = qml.device("default.qubit", wires=2)


def qnode_0(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


def qnode_1(X):
    qml.RX(X[0], wires=0)
    qml.RX(X[0], wires=1)
    qml.RX(X[1], wires=0)
    qml.RX(X[1], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def qnode_2(X, y):
    qml.RX(X[0], wires=0)
    qml.RX(X[2], wires=1)
    qml.RY(y, wires=0)
    qml.RX(0.5 * X[0], wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def qnode_3(X, Y):
    for i in range(3):
        qml.RX(X[i], wires=0)
        qml.RY(Y[i], wires=0)
        qml.RX(X[i], wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# pylint: disable=unused-argument
def qnode_4(x):
    return qml.expval(qml.PauliX(0))


def qnode_5(Z, y):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(Z[0, 1], wires=0)
    qml.RZ(Z[2, 4], wires=1)
    qml.RY(y, wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


_x = 0.1
_y = 2.3

_X = pnp.array([i**1.2 - 2.0 / i for i in range(1, 6)])
_Y = pnp.array([i**0.9 - 1.0 / i for i in range(1, 6)])
_Z = pnp.array(
    [
        [0.3, 9.1, -0.2, 0.6, 1.2],
        [0.9, -0.1, 1.6, 2.3, -1.5],
        [0.3, 0.1, -0.9, 0.6, 1.8],
    ]
)

test_cases_qnodes = [
    (qnode_0, (_x,), "x", None, {"x": {(): [0.0, 1.0]}}, None, 3),
    (
        qnode_0,
        (_x,),
        {"x": [()]},
        None,
        {"x": {(): [0.0, 1.0]}},
        {"x": {(): [-np.pi / 3, 0.0, np.pi / 3]}},
        3,
    ),
    (qnode_0, (_x,), "x", {"x": {(): 1}}, None, None, 3),
    (qnode_1, (_X,), {"X"}, None, {"X": {(0,): [0.0, 1.0, 2.0], (1,): [0.0, 2.0]}}, None, 7),
    (
        qnode_1,
        (_X,),
        "X",
        None,
        {"X": {(0,): [0.0, 2.0]}},
        {"X": {(0,): [-np.pi / 2, -0.1, np.pi / 5]}},
        3,
    ),
    (qnode_1, (_X,), ["X"], {"X": {(0,): 2, (1,): 2}}, None, None, 9),
    (
        qnode_2,
        (_X, _y),
        ["X", "y"],
        None,
        {"X": {(0,): [0.0, 0.5, 1.0, 1.5], (2,): [0.0, 1.0]}, "y": {(): [0.0, 1.0]}},
        None,
        11,
    ),
    (
        qnode_3,
        (_X, _Y),
        {"X": [(0,), (3,)], "Y": ((4,), (1,))},
        {"X": {(i,): 2 for i in range(5)}, "Y": {(i,): 1 for i in range(5)}},
        None,
        None,
        13,
    ),
    (qnode_4, (_x,), ["x"], {"x": {(): 0}}, None, None, 1),
    (qnode_5, (_Z, _y), ["Z"], {"Z": {(0, 1): 1, (2, 4): 1, (1, 3): 0}}, None, None, 5),
]


# pylint: disable=cell-var-from-loop
class TestReconstruct:
    """Tests the integration of ``_reconstruct_equ`` and ``_reconstruct_gen`` via
    the full ``reconstruct`` function as well as the differentiability of the
    reconstructed function with respect to their single scalar argument."""

    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts, exp_calls",
        test_cases_qnodes,
    )
    def test_with_qnode(self, qnode, params, ids, nums_frequency, spectra, shifts, exp_calls):
        """Run a full reconstruction on a QNode."""
        qnode = qml.QNode(qnode, dev_1, interface="autograd")

        with qml.Tracker(qnode.device) as tracker:
            recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)

        assert tracker.totals["executions"] == exp_calls
        arg_names = list(signature(qnode.func).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                x0 = params[outer_key_num]
                if not pnp.isscalar(x0):
                    x0 = x0[inner_key]
                    shift_vec = qml.math.zeros_like(params[outer_key_num])
                    shift_vec[inner_key] = 1.0
                shift_vec = 1.0 if pnp.isscalar(params[outer_key_num]) else shift_vec
                mask = (
                    0.0
                    if pnp.isscalar(params[outer_key_num])
                    else pnp.ones(qml.math.shape(params[outer_key_num])) - shift_vec
                )
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] * mask + x * shift_vec,
                    *params[outer_key_num + 1 :],
                )
                assert np.isclose(rec(x0), qnode(*params))
                assert np.isclose(rec(x0 + 0.1), univariate(x0 + 0.1))
                assert fun_close(rec, univariate, 10)

    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts, exp_calls",
        test_cases_qnodes,
    )
    def test_differentiability_autograd(
        self, qnode, params, ids, nums_frequency, spectra, shifts, exp_calls
    ):
        """Tests the reconstruction and differentiability with autograd."""
        qnode = qml.QNode(qnode, dev_1, interface="autograd")
        with qml.Tracker(qnode.device) as tracker:
            recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)
        assert tracker.totals["executions"] == exp_calls
        arg_names = list(signature(qnode.func).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                x0 = params[outer_key_num]
                if not pnp.isscalar(x0):
                    x0 = x0[inner_key]
                    shift_vec = qml.math.zeros_like(params[outer_key_num])
                    shift_vec[inner_key] = 1.0
                shift_vec = 1.0 if pnp.isscalar(params[outer_key_num]) else shift_vec
                mask = (
                    0.0
                    if pnp.isscalar(params[outer_key_num])
                    else pnp.ones(qml.math.shape(params[outer_key_num])) - shift_vec
                )
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] * mask + x * shift_vec,
                    *params[outer_key_num + 1 :],
                )
                exp_qnode_grad = qml.grad(qnode, argnum=outer_key_num)
                exp_grad = qml.grad(univariate)
                grad = qml.grad(rec)
                if nums_frequency is None:
                    # Gradient evaluation at reconstruction point not supported for
                    # Dirichlet reconstruction
                    assert np.isclose(
                        grad(pnp.array(x0, requires_grad=True)),
                        exp_qnode_grad(*params)[inner_key],
                    )
                assert np.isclose(
                    grad(pnp.array(x0 + 0.1, requires_grad=True)),
                    exp_grad(pnp.array(x0 + 0.1, requires_grad=True)),
                )
                assert fun_close(grad, exp_grad, pnp.array(10, requires_grad=True))

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts, exp_calls",
        test_cases_qnodes,
    )
    def test_differentiability_jax(
        self, qnode, params, ids, nums_frequency, spectra, shifts, exp_calls
    ):
        """Tests the reconstruction and differentiability with JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)

        params = tuple(jax.numpy.array(par) for par in params)
        qnode = qml.QNode(qnode, dev_1, interface="jax")
        with qml.Tracker(qnode.device) as tracker:
            recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)
        assert tracker.totals["executions"] == exp_calls
        arg_names = list(signature(qnode.func).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                x0 = params[outer_key_num]
                if not pnp.isscalar(x0):
                    x0 = x0[inner_key]
                    shift_vec = qml.math.zeros_like(params[outer_key_num])
                    shift_vec = qml.math.scatter_element_add(shift_vec, inner_key, 1.0)
                shift_vec = 1.0 if pnp.isscalar(params[outer_key_num]) else shift_vec
                mask = (
                    0.0
                    if pnp.isscalar(params[outer_key_num])
                    else pnp.ones(qml.math.shape(params[outer_key_num])) - shift_vec
                )
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] * mask + x * shift_vec,
                    *params[outer_key_num + 1 :],
                )
                exp_qnode_grad = jax.grad(qnode, argnums=outer_key_num)
                exp_grad = jax.grad(univariate)
                grad = jax.grad(rec)
                assert np.isclose(grad(x0), exp_qnode_grad(*params)[inner_key])
                assert np.isclose(grad(x0 + 0.1), exp_grad(x0 + 0.1))
                assert fun_close(grad, exp_grad, samples=3)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts, exp_calls",
        test_cases_qnodes,
    )
    def test_differentiability_tensorflow(
        self, qnode, params, ids, nums_frequency, spectra, shifts, exp_calls
    ):
        """Tests the reconstruction and differentiability with TensorFlow."""
        if qnode is qnode_4:
            pytest.skip("Gradients are empty in TensorFlow for independent functions.")
        import tensorflow as tf

        qnode = qml.QNode(qnode, dev_1, interface="tf")
        params = tuple(tf.Variable(par, dtype=tf.float64) for par in params)
        if spectra is not None:
            spectra = {
                outer_key: {
                    inner_key: tf.constant(val, dtype=tf.float64)
                    for inner_key, val in outer_val.items()
                }
                for outer_key, outer_val in spectra.items()
            }
        if shifts is not None:
            shifts = {
                outer_key: {
                    inner_key: tf.constant(val, dtype=tf.float64)
                    for inner_key, val in outer_val.items()
                }
                for outer_key, outer_val in shifts.items()
            }
        with qml.Tracker(qnode.device) as tracker:
            recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)
        assert tracker.totals["executions"] == exp_calls
        arg_names = list(signature(qnode.func).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                if outer_key == "Z" and inner_key == (1, 3):
                    # This is a constant function dependence, which can
                    # not be properly resolved by this test.
                    continue
                x0 = params[outer_key_num]
                if not len(qml.math.shape(x0)) == 0:
                    x0 = x0[inner_key]
                    shift_vec = qml.math.zeros_like(params[outer_key_num])
                    shift_vec = qml.math.scatter_element_add(shift_vec, inner_key, 1.0)
                    mask = pnp.ones(qml.math.shape(params[outer_key_num])) - shift_vec
                else:
                    shift_vec = 1.0
                    mask = 0.0
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] * mask + x * shift_vec,
                    *params[outer_key_num + 1 :],
                )
                with tf.GradientTape() as tape:
                    out = qnode(*params)
                exp_qnode_grad = tape.gradient(out, params[outer_key_num])

                def exp_grad(x):
                    x = tf.Variable(x, dtype=tf.float64)
                    with tf.GradientTape() as tape:
                        out = univariate(x)
                    return tape.gradient(out, x)

                def grad(x):
                    x = tf.Variable(x, dtype=tf.float64)
                    with tf.GradientTape() as tape:
                        out = rec(x)
                    return tape.gradient(out, x)

                if nums_frequency is None:
                    # Gradient evaluation at reconstruction point not supported for
                    # Dirichlet reconstruction
                    assert np.isclose(grad(x0), exp_qnode_grad[inner_key])
                assert np.isclose(grad(x0 + 0.1), exp_grad(x0 + 0.1))
                assert fun_close(grad, exp_grad, 10)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts, exp_calls",
        test_cases_qnodes,
    )
    def test_differentiability_torch(
        self, qnode, params, ids, nums_frequency, spectra, shifts, exp_calls
    ):
        """Tests the reconstruction and differentiability with Torch."""
        import torch

        qnode = qml.QNode(qnode, dev_1, interface="torch")
        params = tuple(torch.tensor(par, requires_grad=True, dtype=torch.float64) for par in params)
        if spectra is not None:
            spectra = {
                outer_key: {
                    inner_key: torch.tensor(val, dtype=torch.float64)
                    for inner_key, val in outer_val.items()
                }
                for outer_key, outer_val in spectra.items()
            }
        if shifts is not None:
            shifts = {
                outer_key: {
                    inner_key: torch.tensor(val, dtype=torch.float64)
                    for inner_key, val in outer_val.items()
                }
                for outer_key, outer_val in shifts.items()
            }
        with qml.Tracker(qnode.device) as tracker:
            recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)
        assert tracker.totals["executions"] == exp_calls
        arg_names = list(signature(qnode.func).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                x0 = params[outer_key_num]
                if not len(qml.math.shape(x0)) == 0:
                    x0 = x0[inner_key]
                    shift_vec = qml.math.zeros_like(params[outer_key_num])
                    shift_vec = qml.math.scatter_element_add(shift_vec, inner_key, 1.0)
                    mask = torch.ones(qml.math.shape(params[outer_key_num])) - shift_vec
                else:
                    shift_vec = 1.0
                    mask = 0.0
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] * mask + x * shift_vec,
                    *params[outer_key_num + 1 :],
                )
                exp_qnode_grad = torch.autograd.functional.jacobian(qnode, params)[outer_key_num]

                exp_grad = lambda x: torch.autograd.functional.jacobian(univariate, x)
                grad = lambda x: torch.autograd.functional.jacobian(rec, x)

                assert np.isclose(grad(x0), exp_qnode_grad[inner_key])
                assert np.isclose(grad(x0 + 0.1), exp_grad(x0 + 0.1))
                assert fun_close(
                    grad, exp_grad, zero=torch.tensor(0.0, requires_grad=True), samples=10
                )
