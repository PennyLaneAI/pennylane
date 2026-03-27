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
# pylint: disable=too-many-arguments,too-few-public-methods
import functools

import pytest
from scipy.optimize import shgo

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import RotosolveOptimizer
from pennylane.optimize.qng import _flatten_np, _unflatten_np


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
    par1_flat = np.fromiter(_flatten_np(par1), dtype=float)
    par2_flat = np.fromiter(_flatten_np(par2), dtype=float)
    walking_param = []
    for i in range(len(par1_flat) + 1):
        walking_param.append(_unflatten_np(np.append(par2_flat[:i], par1_flat[i:]), par1))
    return walking_param


def test_error_missing_frequency_info():
    """Test that an error is raised if neither nums_frequency nor spectra is given."""

    opt = RotosolveOptimizer()
    fun = lambda x: x
    x = np.array(0.5, requires_grad=True)

    with pytest.raises(ValueError, match="Neither the number of frequencies nor the"):
        opt.step(fun, x)


def test_no_error_missing_frequency_info_untrainable():
    """Test that no error is raised if neither nums_frequency nor spectra
    is given for a parameter not marked as trainable."""

    opt = RotosolveOptimizer()
    fun = lambda x, y: x
    x = np.array(0.5, requires_grad=True)
    y = np.array(0.1, requires_grad=False)
    nums_frequency = {"x": {(): 1}}

    opt.step(fun, x, y, nums_frequency=nums_frequency)


def test_error_missing_frequency_info_single_par():
    """Test that an error is raised if neither nums_frequency nor spectra is given
    for one of the function arguments."""

    opt = RotosolveOptimizer()

    def sum_named_arg(x):
        return qml.math.sum(x)

    x = np.arange(4, requires_grad=True)
    nums_frequency = {"x": {(0,): 1, (1,): 1}}
    spectra = {"x": {(0,): [0.0, 1.0], (2,): [0.0, 1.0]}}

    # For the first three entries either nums_frequency or spectra is provided
    with pytest.raises(ValueError, match=r"was provided for the entry \(3,\)"):
        opt.step(sum_named_arg, x, nums_frequency=nums_frequency, spectra=spectra)


def test_error_no_trainable_args():
    """Test that an error is raised if none of the arguments is trainable."""

    opt = RotosolveOptimizer()
    fun = lambda x, y, z: 1.0

    x = np.array(1.0, requires_grad=False)
    y = np.array(2.0, requires_grad=False)
    z = np.array(3.0, requires_grad=False)
    args = (x, y, z)

    with pytest.raises(ValueError, match="Found no parameters to optimize."):
        opt.step(fun, *args, nums_frequency=None, spectra=None)


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
    lambda x, y, z: (
        -np.cos(x + 0.06)
        - np.cos(2 * x + 0.12) * 0.872
        + np.sin(y - 2.01 / 3 - np.pi / 3)
        + np.sin(3 * y - 2.01)
        - np.cos(z - 1.35) * 0.111
    ),
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

classical_nums_frequency = [
    {"x": {(): 1}},
    {"x": {(0,): 1, (1,): 1, (2,): 1}},
    {"x": {(): 1}, "y": {(0,): 1, (1,): 1}},
    {"x": {(): 1}, "y": {(0,): 2, (1,): 2}},
    {"x": {(): 1}, "y": {(): 1}, "z": {(): 1}},
    {"x": {(): 2}, "y": {(): 3}, "z": {(): 1}},
]

classical_expected_num_calls = [3, 9, 9, 13, 9, 15]


def custom_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` that does not return y_min."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x[0], None


substep_optimizers = ["brute", "shgo", custom_optimizer]
all_substep_kwargs = [
    {"Ns": 93, "num_steps": 3},
    {"bounds": ((-1.0, 1.0),), "n": 512},
    {"bounds": ((-1.1, 1.4),)},
]


@pytest.mark.parametrize(
    "fun, x_min, param, nums_freq, exp_num_calls",
    list(
        zip(
            classical_functions,
            classical_minima,
            classical_params,
            classical_nums_frequency,
            classical_expected_num_calls,
        )
    ),
)
@pytest.mark.parametrize(
    "substep_optimizer, substep_kwargs",
    list(zip(substep_optimizers, all_substep_kwargs)),
)
class TestWithClassicalFunction:
    # pylint: disable=unused-argument
    def test_number_of_function_calls(
        self, fun, x_min, param, nums_freq, exp_num_calls, substep_optimizer, substep_kwargs
    ):
        """Tests that per parameter 2R+1 function calls are used for an update step."""
        # pylint: disable=too-many-arguments
        num_calls = 0

        @functools.wraps(fun)
        def _fun(*args, **kwargs):
            nonlocal num_calls
            num_calls += 1
            return fun(*args, **kwargs)

        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        # Make only the first argument trainable
        param = (np.array(param[0], requires_grad=True),) + param[1:]
        # Only one argument is marked as trainable -> Expect only the executions for that arg
        opt.step(_fun, *param, nums_frequency=nums_freq)
        exp_num_calls_single_trainable = sum(2 * num + 1 for num in nums_freq["x"].values())
        assert num_calls == exp_num_calls_single_trainable
        num_calls = 0

        # Parameters are now marked as trainable -> Expect full number of executions
        param = tuple(np.array(p, requires_grad=True) for p in param)
        opt.step(_fun, *param, nums_frequency=nums_freq)
        assert num_calls == exp_num_calls

    def test_single_step_convergence(
        self, fun, x_min, param, nums_freq, exp_num_calls, substep_optimizer, substep_kwargs
    ):
        """Tests convergence for easy classical functions in a single Rotosolve step.
        Includes testing of the parameter output shape and the old cost when using step_and_cost."""
        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        # Make only the first argument trainable
        param = (np.array(param[0], requires_grad=True),) + param[1:]
        # Only one argument is marked as trainable -> All other arguments have to stay fixed
        new_param_step = opt.step(
            fun,
            *param,
            nums_frequency=nums_freq,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param_step,)

        assert all(np.allclose(p, new_p) for p, new_p in zip(param[1:], new_param_step[1:]))

        # With trainable parameters, training should happen
        param = tuple(np.array(p, requires_grad=True) for p in param)
        new_param_step = opt.step(
            fun,
            *param,
            nums_frequency=nums_freq,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param_step,)

        assert len(x_min) == len(new_param_step)
        assert np.allclose(
            np.fromiter(_flatten_np(x_min), dtype=float),
            np.fromiter(_flatten_np(new_param_step), dtype=float),
            atol=1e-5,
        )

        # Now with step_and_cost and trainable params
        # pylint:disable=unbalanced-tuple-unpacking
        new_param_step_and_cost, old_cost = opt.step_and_cost(
            fun,
            *param,
            nums_frequency=nums_freq,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert len(x_min) == len(new_param_step_and_cost)
        assert np.allclose(
            np.fromiter(_flatten_np(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten_np(new_param_step), dtype=float),
            atol=1e-5,
        )
        assert np.isclose(old_cost, fun(*param))

    def test_full_output(
        self, fun, x_min, param, nums_freq, exp_num_calls, substep_optimizer, substep_kwargs
    ):
        """Tests the ``full_output`` feature of Rotosolve, delivering intermediate cost
        function values at the univariate optimization substeps."""
        param = tuple(np.array(p, requires_grad=True) for p in param)
        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        _, y_output_step = opt.step(
            fun,
            *param,
            nums_frequency=nums_freq,
            full_output=True,
        )
        new_param, old_cost, y_output_step_and_cost = opt.step_and_cost(
            fun,
            *param,
            nums_frequency=nums_freq,
            full_output=True,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param = (new_param,)
        expected_intermediate_x = successive_params(param, new_param)
        expected_y_output = [fun(*par) for par in expected_intermediate_x[1:]]

        assert np.allclose(y_output_step, expected_y_output)
        assert np.allclose(y_output_step_and_cost, expected_y_output)
        assert np.isclose(old_cost, fun(*expected_intermediate_x[0]))


@pytest.mark.parametrize(
    "fun, x_min, param, num_freq",
    list(zip(classical_functions, classical_minima, classical_params, classical_nums_frequency)),
)
def test_multiple_steps(fun, x_min, param, num_freq):
    """Tests that repeated steps execute as expected."""
    param = tuple(np.array(p, requires_grad=True) for p in param)
    substep_optimizer = "brute"
    substep_kwargs = None
    opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

    for _ in range(3):
        param = opt.step(
            fun,
            *param,
            nums_frequency=num_freq,
        )
        # The following accounts for the unpacking functionality for length-one param
        if len(x_min) == 1:
            param = (param,)

    assert (np.isscalar(x_min) and np.isscalar(param)) or len(x_min) == len(param)
    assert np.allclose(
        np.fromiter(_flatten_np(x_min), dtype=float),
        np.fromiter(_flatten_np(param), dtype=float),
        atol=1e-5,
    )


classical_functions_deact = [
    lambda x, y: -np.cos(x + 0.12) * 0.872 + np.sin(y[0] - 2.01) - np.cos(y[1] - 1.35) * 0.111,
    lambda x, y, z: -np.cos(x + 0.12) * 0.872 + np.sin(y - 2.01) - np.cos(z - 1.35) * 0.111,
]
classical_minima_deact = [
    (-0.12, [0.8, 0.1]),
    (-0.12, 0.2, 1.35),
]
classical_params_deact = [
    (np.array(0.3, requires_grad=True), np.array([0.8, 0.1], requires_grad=False)),
    (
        np.array(0.1, requires_grad=True),
        np.array(0.2, requires_grad=False),
        np.array(0.5, requires_grad=True),
    ),
]
classical_nums_frequency_deact = [
    {"x": {(): 1}, "y": {(0,): 1, (1,): 1}},
    {"x": {(): 1}, "y": {(): 1}, "z": {(): 1}},
]


@pytest.mark.parametrize(
    "fun, x_min, param, num_freq",
    list(
        zip(
            classical_functions_deact,
            classical_minima_deact,
            classical_params_deact,
            classical_nums_frequency_deact,
        )
    ),
)
class TestDeactivatedTrainingWithClassicalFunctions:
    def test_single_step(self, fun, x_min, param, num_freq):
        """Tests convergence for easy classical functions in a single Rotosolve step
        with some arguments deactivated for training.
        Includes testing of the parameter output shape and the old cost when using step_and_cost."""
        substep_optimizer = "brute"
        substep_kwargs = None
        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        new_param_step = opt.step(
            fun,
            *param,
            nums_frequency=num_freq,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step = (new_param_step,)

        assert len(x_min) == len(new_param_step)
        assert np.allclose(
            np.fromiter(_flatten_np(x_min), dtype=float),
            np.fromiter(_flatten_np(new_param_step), dtype=float),
            atol=1e-5,
        )

        # pylint:disable=unbalanced-tuple-unpacking
        new_param_step_and_cost, old_cost = opt.step_and_cost(
            fun,
            *param,
            nums_frequency=num_freq,
        )
        # The following accounts for the unpacking functionality for length-1 param
        if len(param) == 1:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert len(x_min) == len(new_param_step_and_cost)
        assert np.allclose(
            np.fromiter(_flatten_np(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten_np(new_param_step), dtype=float),
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
qnode_nums_frequency = [
    {"x": {(): num_wires}},
    {"x": {(0,): 1, (1,): 1, (2,): 1}, "y": {(): 2 * num_wires}, "z": {(0,): 1, (1,): 2}},
    None,
]
qnode_spectra = [
    None,
    None,
    {
        "x": {(): list(range(num_wires + 1))},
        "y": {(): list(range(num_wires + 1))},
        "z": {(): list(range(num_wires + 1))},
    },
]


@pytest.mark.parametrize(
    "qnode, param, nums_frequency, spectra",
    list(zip(qnodes, qnode_params, qnode_nums_frequency, qnode_spectra)),
)
@pytest.mark.parametrize(
    "substep_optimizer, substep_kwargs",
    list(zip(substep_optimizers, all_substep_kwargs)),
)
class TestWithQNodes:
    def test_single_step(
        self, qnode, param, nums_frequency, spectra, substep_optimizer, substep_kwargs
    ):
        """Test executing a single step of the RotosolveOptimizer on a QNode."""
        param = tuple(np.array(p, requires_grad=True) for p in param)
        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        repack_param = len(param) == 1
        new_param_step = opt.step(
            qnode,
            *param,
            nums_frequency=nums_frequency,
            spectra=spectra,
        )
        if repack_param:
            new_param_step = (new_param_step,)

        assert (np.isscalar(new_param_step) and np.isscalar(param)) or len(new_param_step) == len(
            param
        )
        # pylint:disable=unbalanced-tuple-unpacking
        new_param_step_and_cost, old_cost = opt.step_and_cost(
            qnode,
            *param,
            nums_frequency=nums_frequency,
            spectra=spectra,
        )
        if repack_param:
            new_param_step_and_cost = (new_param_step_and_cost,)

        assert np.allclose(
            np.fromiter(_flatten_np(new_param_step_and_cost), dtype=float),
            np.fromiter(_flatten_np(new_param_step), dtype=float),
        )
        assert np.isclose(qnode(*param), old_cost)

    def test_multiple_steps(
        self, qnode, param, nums_frequency, spectra, substep_optimizer, substep_kwargs
    ):
        """Test executing multiple steps of the RotosolveOptimizer on a QNode."""
        param = tuple(np.array(p, requires_grad=True) for p in param)
        # For the following 1D substep_optimizer, the bounds need to be expanded for these QNodes
        if substep_optimizer in ["shgo", custom_optimizer]:
            substep_kwargs["bounds"] = ((-2.0, 2.0),)
        opt = RotosolveOptimizer(substep_optimizer, substep_kwargs)

        repack_param = len(param) == 1
        initial_cost = qnode(*param)

        for _ in range(3):
            param = opt.step(
                qnode,
                *param,
                nums_frequency=nums_frequency,
                spectra=spectra,
            )
            # The following accounts for the unpacking functionality for length-1 param
            if repack_param:
                param = (param,)

        assert qnode(*param) < initial_cost
