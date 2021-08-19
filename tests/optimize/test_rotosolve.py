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
Unit tests for the ``RotosolveOptimizer``.
"""
import pytest
from scipy.optimize import shgo

import pennylane as qml
from pennylane import numpy as np
from pennylane.utils import _flatten, unflatten

from pennylane.optimize import RotosolveOptimizer


def expand_num_freq(num_freq, param):
    if np.isscalar(num_freq):
        num_freq = [num_freq] * len(param)
    expanded = []
    for _num_freq, par in zip(num_freq, param):
        if np.isscalar(_num_freq) and np.isscalar(par):
            expanded.append(_num_freq)
        elif np.isscalar(_num_freq):
            expanded.append(np.ones_like(par) * _num_freq)
        elif np.isscalar(par):
            raise ValueError(f"{num_freq}\n{param}\n{_num_freq}\n{par}")
        elif len(_num_freq) == len(par):
            expanded.append(_num_freq)
        else:
            raise ValueError()
    return expanded


def successive_params(par1, par2):
    """Return a list of parameter configurations, successively walking from
    par1 to par2 coordinate-wise."""
    par1_flat = np.fromiter(_flatten(par1), dtype=float)
    par2_flat = np.fromiter(_flatten(par2), dtype=float)
    walking_param = []
    for i in range(len(par1_flat) + 1):
        walking_param.append(unflatten(np.append(par2_flat[:i], par1_flat[i:]), par1))
    return walking_param


@pytest.mark.parametrize(
    "fun, param, num_freq",
    zip(
        [
            lambda x: np.sin(x),
            lambda x: np.sin(x),
            lambda x, y: np.sin(x) * np.sin(y),
            lambda x, y: np.sin(x) * np.sin(y[0]) * np.sin(y[1]),
        ],
        [[0.5], [0.5], [0.5, 0.2], [0.5, [0.2, 0.4]]],
        [[], [1, 1], [1], [1, 1, [1, 2]]],
    ),
)
def test_wrong_len_num_freqs(fun, param, num_freq):
    """Test that an error is raised for a different number of
    numbers of frequencies than number of function arguments."""

    opt = RotosolveOptimizer()

    with pytest.raises(ValueError, match="The length of the provided numbers of frequencies"):
        opt.step(fun, *param, num_freqs=num_freq)


@pytest.mark.parametrize(
    "fun, param, num_freq",
    zip(
        [
            lambda x: np.sin(x),
            lambda x: np.sin(x),
            lambda x, y: np.sin(x) * np.sin(y),
            lambda x, y: np.sin(x) * np.sin(y[0]) * np.sin(y[1]),
        ],
        [[0.5], [0.5], [0.5, 0.2], [0.5, [0.2, 0.4]]],
        [[[1, 1]], [[]], [[1], [1, 1]], [[1], [1]]],
    ),
)
def test_wrong_num_of_num_freqs_per_parameter(fun, param, num_freq):
    """Test that an error is raised for a different number of
    numbers of frequencies than number of function arguments."""

    opt = RotosolveOptimizer()

    with pytest.raises(ValueError, match="The number of the frequency counts"):
        opt.step(fun, *param, num_freqs=num_freq)


@pytest.mark.parametrize(
    "fun, param, num_freq",
    zip(
        [
            lambda x: np.sin(x),
            lambda x: np.sin(x),
            lambda x, y: np.sin(x) * np.sin(y),
            lambda x, y: np.sin(x) * np.sin(y[0]) * np.sin(y[1]),
        ],
        [[0.5], [0.5], [0.5, 0.2], [0.5, [0.2, 0.4]]],
        [[0.1], [1.0], [1, 1.0], [1, [1, 2 + 1j]]],
    ),
)
def test_wrong_typed_num_freqs(fun, param, num_freq):
    """Test that an error is raised for a non-integer entry in the numbers of frequencies."""

    opt = RotosolveOptimizer()

    with pytest.raises(ValueError, match="The numbers of frequencies are expected to be integers."):
        opt.step(fun, *param, num_freqs=num_freq)


classical_functions = [
    lambda x: np.sin(x + 0.124) * 2.5123,
    lambda x: -np.cos(x[0] + 0.12) * 0.872 + np.sin(x[1] - 2.01) - np.cos(x[2] - 1.35) * 0.111,
    lambda x, y: -np.cos(x + 0.12) * 0.872 + np.sin(y[0] - 2.01) - np.cos(y[1] - 1.35) * 0.111,
    lambda x, y: (
        -np.cos(x + 0.12) * 0.872
        + np.sin(2 * y[0] - 2.01)
        + np.sin(y[0] - 2.01 / 2 - np.pi / 4) * 0.1
        - np.cos(y[1] - 1.35 / 2) * 0.2
        - np.cos(2 * y[1] - 1.35) * 0.111
    ),
    lambda x, y, z: -np.cos(x + 0.12) * 0.872 + np.sin(y - 2.01) - np.cos(z - 1.35) * 0.111,
    lambda x, y, z: -np.cos(x + 0.06)
    - np.cos(2 * x + 0.12) * 0.872
    + np.sin(y - 2.01 / 3 - np.pi / 3)
    + np.sin(3 * y - 2.01)
    - np.cos(z - 1.35) * 0.111,
]
classical_minima = [
    (-np.pi / 2 - 0.124,),
    ([-0.12, -np.pi / 2 + 2.01, 1.35],),
    (-0.12, [-np.pi / 2 + 2.01, 1.35]),
    (-0.12, [(-np.pi / 2 + 2.01) / 2, 1.35 / 2]),
    (-0.12, -np.pi / 2 + 2.01, 1.35),
    (-0.12 / 2, (-np.pi / 2 + 2.01) / 3, 1.35),
]
classical_params = [
    (0.24,),
    ([0.2, -0.3, 0.1],),
    (0.3, [0.8, 0.1]),
    (0.2, [0.3, 0.5]),
    (0.1, 0.2, 0.5),
    (0.9, 0.7, 0.2),
]
classical_num_freqs = [[1], [[1, 1, 1]], [1, [1, 1]], [1, 2], 1, [2, 3, 1]]


def custom_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` that does not return y_min."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x, None


optimizers = [None, "brute", "shgo", custom_optimizer]
optimizer_kwargs = [
    {"Ns": 93, "num_steps": 3},
    None,
    {"bounds": ((-1.0, 1.0),), "n": 512},
    {"bounds": ((-1.1, 1.4),)},
]


@pytest.mark.parametrize(
    "fun, x_min, param, num_freq",
    list(zip(classical_functions, classical_minima, classical_params, classical_num_freqs)),
)
@pytest.mark.parametrize(
    "optimizer, optimizer_kwargs",
    list(zip(optimizers, optimizer_kwargs)),
)
class TestWithClassicalFunctions:
    def test_number_of_function_calls(
        self, fun, x_min, param, num_freq, optimizer, optimizer_kwargs
    ):
        """Tests that per parameter 2R+1 function calls are used for an update step."""
        global num_calls
        num_calls = 0

        def _fun(*args, **kwargs):
            global num_calls
            num_calls += 1
            return fun(*args, **kwargs)

        opt = RotosolveOptimizer()
        new_param = opt.step(
            _fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        expected_num_calls = np.sum(
            np.fromiter(_flatten(expand_num_freq(num_freq, param)), dtype=int) * 2 + 1
        )
        assert num_calls == expected_num_calls

    def test_single_step_convergence(
        self, fun, x_min, param, num_freq, optimizer, optimizer_kwargs
    ):
        """Tests convergence for easy classical functions in a single Rotosolve step.
        Includes testing of the parameter output shape and the old cost when using step_and_cost."""
        opt = RotosolveOptimizer()

        new_param_step = opt.step(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param_step,)

        assert len(x_min) == len(new_param_step)
        assert np.allclose(
            np.fromiter(_flatten(x_min), dtype=float),
            np.fromiter(_flatten(new_param_step), dtype=float),
            atol=1e-5,
        )

        new_param_step_and_cost, old_cost = opt.step_and_cost(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert len(x_min) == len(new_param_step_and_cost)
        assert np.allclose(
            np.fromiter(_flatten(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten(new_param_step), dtype=float),
            atol=1e-5,
        )
        assert np.isclose(old_cost, fun(*param))

    def test_full_output(self, fun, x_min, param, num_freq, optimizer, optimizer_kwargs):
        """Tests the ``full_output`` feature of Rotosolve, delivering intermediate cost
        function values at the univariate optimization substeps."""
        opt = RotosolveOptimizer()

        _, y_output_step = opt.step(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            full_output=True,
        )
        new_param, old_cost, y_output_step_and_cost = opt.step_and_cost(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            full_output=True,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param,)
        expected_intermediate_x = successive_params(param, new_param)
        expected_y_output = [fun(*par) for par in expected_intermediate_x[1:]]

        assert np.allclose(y_output_step, expected_y_output)
        assert np.allclose(y_output_step_and_cost, expected_y_output)
        assert np.isclose(old_cost, fun(*expected_intermediate_x[0]))


@pytest.mark.parametrize(
    "fun, x_min, param, num_freq",
    list(zip(classical_functions, classical_minima, classical_params, classical_num_freqs)),
)
def test_multiple_steps(fun, x_min, param, num_freq):
    """Tests that repeated steps execute as expected."""
    opt = RotosolveOptimizer()

    optimizer = "brute"
    optimizer_kwargs = None
    for _ in range(3):
        param = opt.step(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(x_min) == 1:
            param = (param,)

    assert (np.isscalar(x_min) and np.isscalar(param)) or len(x_min) == len(param)
    assert np.allclose(
        np.fromiter(_flatten(x_min), dtype=float),
        np.fromiter(_flatten(param), dtype=float),
        atol=1e-5,
    )


classical_functions_deact = [
    lambda x: np.sin(x + 0.124) * 2.5123,
    lambda x, y: -np.cos(x + 0.12) * 0.872 + np.sin(y[0] - 2.01) - np.cos(y[1] - 1.35) * 0.111,
    lambda x, y, z: -np.cos(x + 0.12) * 0.872 + np.sin(y - 2.01) - np.cos(z - 1.35) * 0.111,
]
classical_minima_deact = [
    (0.24,),
    (-0.12, [0.8, 0.1]),
    (-0.12, 0.2, 1.35),
]
classical_params_deact = [
    (np.array(0.24, requires_grad=False),),
    (0.3, np.array([0.8, 0.1], requires_grad=False)),
    (0.1, np.array(0.2, requires_grad=False), 0.5),
]
classical_num_freqs_deact = [[], [1], 1]


@pytest.mark.parametrize(
    "fun, x_min, param, num_freq",
    list(
        zip(
            classical_functions_deact,
            classical_minima_deact,
            classical_params_deact,
            classical_num_freqs_deact,
        )
    ),
)
class TestDeactivatedTrainingWithClassicalFunctions:
    def test_single_step(self, fun, x_min, param, num_freq):
        """Tests convergence for easy classical functions in a single Rotosolve step
        with some arguments deactivated for training.
        Includes testing of the parameter output shape and the old cost when using step_and_cost."""
        opt = RotosolveOptimizer()

        new_param_step = opt.step(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer="brute",
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param_step,)

        assert len(x_min) == len(new_param_step)
        assert np.allclose(
            np.fromiter(_flatten(x_min), dtype=float),
            np.fromiter(_flatten(new_param_step), dtype=float),
            atol=1e-5,
        )

        new_param_step_and_cost, old_cost = opt.step_and_cost(
            fun,
            *param,
            num_freqs=num_freq,
            optimizer="brute",
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert len(x_min) == len(new_param_step_and_cost)
        assert np.allclose(
            np.fromiter(_flatten(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten(new_param_step), dtype=float),
            atol=1e-5,
        )
        assert np.isclose(old_cost, fun(*param))


num_wires = 3
dev = qml.device("default.qubit", wires=num_wires)


@qml.qnode(dev)
def scalar_qnode(x):
    for w in dev.wires:
        qml.RX(x, wires=w)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))


@qml.qnode(dev)
def array_qnode(x, y, z):
    for _x, w in zip(x, dev.wires):
        qml.RX(_x, wires=w)

    for i in range(num_wires):
        qml.CRY(y, wires=[i, (i + 1) % num_wires])

    qml.RZ(z[0], wires=0)
    qml.RZ(z[1], wires=1)
    qml.RZ(z[1], wires=2)  # z[1] is used twice on purpose

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))


@qml.qnode(dev)
def _postprocessing_qnode(x, y, z):
    for w in dev.wires:
        qml.RX(x, wires=w)
    for w in dev.wires:
        qml.RY(y, wires=w)
    for w in dev.wires:
        qml.RZ(z, wires=w)
    return [qml.expval(qml.PauliZ(w)) for w in dev.wires]


def postprocessing_qnode(x, y, z):
    return np.sum(_postprocessing_qnode(x, y, z))


qnodes = [scalar_qnode, array_qnode, postprocessing_qnode]
qnode_params = [
    (0.2,),
    (np.array([0.1, -0.3, 2.9]), 1.3, [0.2, 0.1]),
    (1.2, -2.3, -0.2),
]
qnode_num_freqs = [
    [num_wires],
    [1, 2 * num_wires, [1, 2]],
    num_wires,
]


@pytest.mark.parametrize(
    "qnode, param, num_freq",
    list(zip(qnodes, qnode_params, qnode_num_freqs)),
)
@pytest.mark.parametrize(
    "optimizer, optimizer_kwargs",
    list(zip(optimizers, optimizer_kwargs)),
)
class TestWithQNodes:
    def test_single_step(self, qnode, param, num_freq, optimizer, optimizer_kwargs):
        opt = RotosolveOptimizer()

        repack_param = len(param) == 1
        new_param_step = opt.step(
            qnode,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        if repack_param:
            new_param_step = (new_param_step,)

        assert (np.isscalar(new_param_step) and np.isscalar(param)) or len(new_param_step) == len(
            param
        )
        new_param_step_and_cost, old_cost = opt.step_and_cost(
            qnode,
            *param,
            num_freqs=num_freq,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        if repack_param:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert np.allclose(
            np.fromiter(_flatten(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten(new_param_step), dtype=float),
        )
        assert np.isclose(qnode(*param), old_cost)

    def test_multiple_steps(self, qnode, param, num_freq, optimizer, optimizer_kwargs):
        opt = RotosolveOptimizer()

        repack_param = len(param) == 1
        initial_cost = qnode(*param)

        for _ in range(3):
            param = opt.step(
                qnode,
                *param,
                num_freqs=num_freq,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
            )
            # The following accounts for the unpacking functionality for length-1 param
            if repack_param:
                param = (param,)

        assert qnode(*param) < initial_cost
