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
from inspect import signature
from itertools import chain, combinations
from functools import reduce
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.reconstruct import (
    _reconstruct_equ,
    _reconstruct_gen,
    _prepare_jobs,
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


def fun_close(fun1, fun2, zero=None, tol=1e-5):
    X = np.linspace(-np.pi, np.pi, 100)
    if zero is not None:
        X = qml.math.cast_like(X, zero)

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
    def test_num_frequency_invalid(self, num_frequency):
        """Tests that an error is raised if ``_reconstruct_equ`` receives a
        negative or non-integer ``num_frequency`` ."""
        with pytest.raises(ValueError, match="num_frequency must be a non-negative integer"):
            _reconstruct_equ(dummy_qnode, num_frequency=num_frequency)


class TestWarnings:
    """Test that warnings are raised e.g. for an ill-conditioned
    Fourier transform during the reconstruction."""

    def test_ill_conditioned(self):
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
        lambda x: 4.01,
        lambda x: qml.math.sum([i ** 2 * 0.1 * qml.math.sin(i * 3.921 * x - 2.7 / i) for i in range(1, 10)]),
    ]

    nums_frequency = [2, 1, 2, 0, 9]
    base_frequencies = [1, 3.2, 2.1, 1, 3.921]
    expected_grads = [
        lambda x: 13.71 * qml.math.cos(x) + 2 * qml.math.sin(2 * x) / 30,
        lambda x: -0.49 * qml.math.cos(3.2 * x) * 3.2,
        lambda x: (-2.1) * (-0.1) * qml.math.sin(-2.1 * x) + 4.2 * 2.9 * qml.math.cos(4.2 * x - 1.2),
        lambda x: 0.0,
        lambda x: qml.math.sum(
            [i * 3.921 * i ** 2 * 0.1 * qml.math.cos(i * 3.921 * x - 2.7 / i) for i in range(1, 10)]
        ),
    ]

    @pytest.mark.parametrize(
        "fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly (via rescaling to integer frequencies)."""
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        assert spy.call_count == num_frequency * 2 + 1
        assert fun_close(fun, rec)

        # Repeat, using precomputed fun_at_zero
        fun_at_zero = _fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        _rec = _reconstruct_equ(_fun, num_frequency, fun_at_zero=fun_at_zero)
        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        assert spy.call_count == num_frequency * 2
        assert fun_close(fun, rec)

    @pytest.mark.parametrize(
        "fun, num_frequency, f0, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_autograd(self, fun, num_frequency, f0, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Autograd input variables."""
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        grad = qml.grad(rec)

        assert fun_close(fun, rec, zero=pnp.array(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=pnp.array(0.0, requires_grad=True))

    @pytest.mark.parametrize(
        "fun, num_frequency, f0, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_jax(self, fun, num_frequency, f0, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for JAX input variables."""
        jax = pytest.importorskip("jax")
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=jax.numpy.array(0.0))
        assert fun_close(expected_grad, grad, zero=jax.numpy.array(0.0))

    @pytest.mark.parametrize(
        "fun, num_frequency, f0, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_tensorflow(self, fun, num_frequency, f0, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for TensorFlow input variables."""
        tf = pytest.importorskip("tensorflow")
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=tf.Variable(0.0))
        assert fun_close(expected_grad, grad, zero=tf.Variable(0.0))

    @pytest.mark.parametrize(
        "fun, num_frequency, f0, expected_grad",
        zip(c_funs, nums_frequency, base_frequencies, expected_grads),
    )
    def test_differentiability_torch(self, fun, num_frequency, f0, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Torch input variables."""
        torch = pytest.importorskip("torch")
        # Convert fun to have integer frequencies
        _fun = lambda x: fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=torch.tensor(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=torch.tensor(0.0, requires_grad=True))

    @pytest.mark.parametrize(
        "fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun_num_freq_too_small(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed wrongly if num_frequency is too small."""
        if num_frequency == 0:
            pytest.skip("Can't reduce the number of frequencies below 0.")
        num_frequency -= 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
        assert spy.call_count == num_frequency * 2 + 1
        assert not fun_close(fun, rec)

    @pytest.mark.parametrize(
        "fun, num_frequency, f0", zip(c_funs, nums_frequency, base_frequencies)
    )
    def test_with_classical_fun_num_freq_too_large(self, fun, num_frequency, f0, mocker):
        """Test that equidistant-frequency classical functions are
        reconstructed correctly if num_frequency is too large."""
        num_frequency += 1
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        # Convert fun to have integer frequencies
        _fun = lambda x: Fun(x / f0)
        _rec = _reconstruct_equ(_fun, num_frequency)

        # Convert reconstruction to have original frequencies
        rec = lambda x: _rec(f0 * x)
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

        # Repeat, using precomputed fun_at_zero
        fun_at_zero = circuit(0.0)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_equ(Fun, num_frequency, fun_at_zero=fun_at_zero)
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
            [i ** 0.9 * 0.2 * np.sin(i ** 1.2 * 3.921 * x - 5.1 / i) for i in range(1, 10)]
        ),
    ]

    spectra = [
        [0.1712, 20.812],
        [3.2],
        [-0.3, -0.1, 0.0, 0.1, 0.3],
        list(range(1, 10)),
        [3.921 * i ** 1.2 for i in range(1, 10)],
    ]

    expected_grads = [
        lambda x: -3.27 * np.cos(0.1712 * x) * 0.1712 + np.sin(20.812 * x) / 23 * 20.812,
        lambda x: -0.49 * np.cos(3.2 * x) * 3.2,
        lambda x: (-0.1) ** 2 * np.sin(-0.1 * x) + 0.3 * 2.9 * np.cos(0.3 * x - 1.2),
        lambda x: np.sum([i * np.cos(i * x) for i in range(1, 10)]),
        lambda x: np.sum(
            [i ** 2.1 * 3.921 * 0.2 * np.cos(i ** 1.2 * 3.921 * x - 5.1 / i) for i in range(1, 10)]
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

        # Repeat, using precomputed fun_at_zero
        fun_at_zero = fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, fun_at_zero=fun_at_zero)
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

        # Repeat, using precomputed fun_at_zero
        fun_at_zero = fun(0.0)
        Fun = Lambda(fun)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, shifts=shifts, fun_at_zero=fun_at_zero)
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
        rec = _reconstruct_gen(fun, spectrum)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=pnp.array(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=pnp.array(0.0, requires_grad=True))

    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_jax(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for JAX input variables."""
        jax = pytest.importorskip("jax")
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=jax.numpy.array(0.0))
        assert fun_close(expected_grad, grad, zero=jax.numpy.array(0.0))

    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_tensorflow(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for TensorFlow input variables."""
        tf = pytest.importorskip("tensorflow")
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=tf.Variable(0.0))
        assert fun_close(expected_grad, grad, zero=tf.Variable(0.0))

    @pytest.mark.parametrize(
        "fun, spectrum, expected_grad",
        zip(c_funs, spectra, expected_grads),
    )
    def test_differentiability_torch(self, fun, spectrum, expected_grad):
        """Test that the reconstruction of equidistant-frequency classical
        functions are differentiable for Torch input variables."""
        torch = pytest.importorskip("torch")
        # Convert fun to have integer frequencies
        rec = _reconstruct_gen(fun, spectrum)
        grad = qml.grad(rec)
        assert fun_close(fun, rec, zero=torch.tensor(0.0, requires_grad=True))
        assert fun_close(expected_grad, grad, zero=torch.tensor(0.0, requires_grad=True))

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

        # Repeat, using precomputed fun_at_zero
        fun_at_zero = circuit(0.0)
        Fun = Lambda(circuit)
        spy = mocker.spy(Fun, "fun")
        rec = _reconstruct_gen(Fun, spectrum, fun_at_zero=fun_at_zero)
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
    None,
    {
        "x": {0: [1.3], 1: [4.2, 0.2]},
        "y": {3: [0.3, 0.2], 1: [1.1, 5.2], 5: [1.2]},
        "z": {i: [i * 3.2, i * 8.7] for i in range(20)},
    },
]

all_shifts = [
    None,
    {
        "x": {0: [1.3], 1: [4.2]},
        "y": {3: [0.3, 0.2], 1: [1.1, 5.2], 5: [1.2]},
        "z": {i: [i * 3.2 + 1.0] for i in range(20)},
    },
]

all_nums_frequency = [
    None,
    {
        "x": {0: 1, 1: 4},
        "y": {3: 1, 1: 1, 5: 9},
        "z": {i: 2 * i for i in range(20)},
    },
]


class TestPrepareJobs:
    """Tests the subroutine that determines the 1D reconstruction
    jobs to be carried out for a call to ``reconstruct`` ."""

    @pytest.mark.parametrize("ids", all_ids)
    @pytest.mark.parametrize("spectra", all_spectra)
    @pytest.mark.parametrize("shifts", all_shifts)
    @pytest.mark.parametrize("nums_frequency", all_nums_frequency)
    def test_prejobs(self, ids, spectra, shifts, nums_frequency, tol):
        """Test ``_prepare_jobs`` with only ``spectra`` given."""
        if nums_frequency is None and spectra is None:
            with pytest.raises(ValueError, match="Either nums_frequency or spectra"):
                _prepare_jobs(ids, spectra, shifts, nums_frequency, atol=tol)
            return

        ids_, recon_fn, jobs, need_fun_at_zero = _prepare_jobs(
            ids,
            spectra,
            shifts,
            nums_frequency,
            atol=tol,
        )

        # Check ids
        if ids is None:
            # Test automatic generation of ids
            if nums_frequency is None:
                assert ids_.keys() == spectra.keys()
                for _id, _ids in ids_.items():
                    assert _ids == spectra[_id].keys()
            else:
                assert ids_.keys() == nums_frequency.keys()
                for _id, _ids in ids_.items():
                    assert _ids == nums_frequency[_id].keys()

        else:
            assert all(id_ in ids for id_ in ids_.keys())
            if type(ids) == dict:
                assert all(ids_[id_] == ids[id_] for id_ in ids_.keys())

        # Check function to use for 1D reconstructions
        assert recon_fn == _reconstruct_gen if nums_frequency is None else _reconstruct_equ
        # Check reconstruction jobs to be run
        print(jobs)
        assert jobs.keys() == ids_.keys()
        for _id, _jobs in jobs.items():
            _jobs.keys() == ids_[_id]
        # Check all job details
        if nums_frequency is None:
            # for spectra given
            for _id, _jobs in jobs.items():
                for idx, job in _jobs.items():
                    if len(spectra[_id][idx]) == 1:
                        assert job is None
                        continue
                    assert list(job.keys()) == ["shifts", "spectrum"]
                    if shifts is not None:
                        assert job["shifts"] == shifts[_id][idx]
                    else:
                        assert job["shifts"] is None
                    job["spectrum"] == spectra[_id][idx]
            if shifts is not None:
                # sometimes need fun at zero if general reconstruction is performed
                _all_shifts = chain.from_iterable(
                    [
                        sum([__shifts for __shifts in _shifts.values()], start=[])
                        for _shifts in shifts.values()
                    ],
                )
                assert need_fun_at_zero == any(
                    np.isclose(_shift, 0.0, atol=tol, rtol=0) for _shift in _all_shifts
                )
            else:
                assert need_fun_at_zero

        else:
            # for nums_frequency given
            for _id, _jobs in jobs.items():
                for idx, job in _jobs.items():
                    if nums_frequency[_id][idx] == 0:
                        assert job is None
                        continue
                    assert list(job.keys()) == ["num_frequency"]
                    assert job["num_frequency"] == nums_frequency[_id][idx]
            # always need fun at zero if equidistant reconstruction is performed
            assert need_fun_at_zero


dev_1 = qml.device("default.qubit", wires=2)


@qml.qnode(dev_1)
def qnode_0(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_1)
def qnode_1(X):
    qml.RX(X[0], wires=0)
    qml.RX(X[0], wires=1)
    qml.RX(X[1], wires=0)
    qml.RX(X[1], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@qml.qnode(dev_1)
def qnode_2(X, y):
    qml.RX(X[0], wires=0)
    qml.RX(X[2], wires=1)
    qml.RY(y, wires=0)
    qml.RX(0.5 * X[0], wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@qml.qnode(dev_1)
def qnode_3(X, Y):
    for i in range(5):
        qml.RX(X[i], wires=0)
        qml.RY(Y[i], wires=0)
        qml.RX(X[i], wires=0)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


x = 0.1
y = 2.3

X = pnp.array([i ** 1.2 - 2.0 / i for i in range(1, 6)])
Y = pnp.array([i ** 0.9 - 1.0 / i for i in range(1, 6)])


class TestReconstruct:
    """Tests the integration of ``_reconstruct_equ`` and ``_reconstruct_gen`` via
    the full ``reconstruct`` function."""

    @pytest.mark.parametrize(
        "qnode, params, ids, nums_frequency, spectra, shifts",
        [
            (qnode_0, (x,), "x", None, {"x": {0: [0.0, 1.0]}}, None),
            (
                qnode_0,
                (x,),
                {"x": [0]},
                None,
                {"x": {0: [0.0, 1.0]}},
                {"x": {0: [-np.pi / 3, 0.0, np.pi / 3]}},
            ),
            (qnode_0, (x,), "x", {"x": {0: 1}}, None, None),
            (qnode_1, (X,), {"X"}, None, {"X": {0: [0.0, 1.0, 2.0], 1: [0.0, 2.0]}}, None),
            (
                qnode_1,
                (X,),
                "X",
                None,
                {"X": {0: [0.0, 2.0]}},
                {"X": {0: [-np.pi / 2, -0.1, np.pi / 5]}},
            ),
            (qnode_1, (X,), ["X"], {"X": {0: 2, 1: 2}}, None, None),
            (
                qnode_2,
                (X, y),
                ["X", "y"],
                None,
                {"X": {0: [0.0, 0.5, 1.0, 1.5], 2: [0.0, 1.0]}, "y": {0: [0.0, 1.0]}},
                None,
            ),
            (
                qnode_3,
                (X, Y),
                {"X": [0, 3], "Y": (4, 1)},
                {"X": {i: 2 for i in range(5)}, "Y": {i: 1 for i in range(5)}},
                None,
                None,
            ),
        ],
    )
    def test_with_qnode(self, qnode, params, ids, nums_frequency, spectra, shifts):
        """Run a full reconstruction on a QNode."""
        recons = reconstruct(qnode, ids, nums_frequency, spectra, shifts)(*params)
        arg_names = list(signature(qnode).parameters.keys())
        for outer_key in recons:
            outer_key_num = arg_names.index(outer_key)
            for inner_key, rec in recons[outer_key].items():
                shift_vec = (
                    1.0
                    if pnp.isscalar(params[outer_key_num])
                    else np.eye(len(params[outer_key_num]))[inner_key]
                )
                univariate = lambda x: qnode(
                    *params[:outer_key_num],
                    params[outer_key_num] + x * shift_vec,
                    *params[outer_key_num + 1 :]
                )
                fun_close(rec, univariate)
