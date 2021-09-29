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
This file contains the _is_independent function that checks 
a function to be independent of its arguments for the interfaces

* Autograd
* JAX
* TensorFlow
* PyTorch
"""
import warnings

import numpy as np

from autograd.tracer import isbox, new_box, trace_stack
from autograd.core import VJPNode


def _autograd_is_independent(func, *args, **kwargs):
    """Test whether a function is independent of its arguments using autograd boxes."""
    # pylint: disable=protected-access
    node = VJPNode.new_root()
    with trace_stack.new_trace() as t:
        start_box = new_box(args, t, node)
        end_box = func(*start_box, **kwargs)

        if type(end_box) in [tuple, list]:
            if any(isbox(_end) and _end._trace == start_box._trace for _end in end_box):
                return False
        elif isinstance(end_box, np.ndarray):
            if end_box.ndim == 0:
                end_box = [end_box.item()]
            if any(isbox(_end) and _end._trace == start_box._trace for _end in end_box):
                return False
        else:
            if isbox(end_box) and end_box._trace == start_box._trace:
                return False
        return True


def _jax_is_independent(func, *args, **kwargs):
    """Test whether a function is independent of its arguments using JAX vjps."""
    import jax  # pylint: disable=import-outside-toplevel

    mapped_func = lambda *_args: func(*_args, **kwargs)
    _vjp = jax.vjp(mapped_func, *args)[1]
    if _vjp.args[0].args != ((),):
        return False
    if _vjp.args[0].func.args[0][0][0] is not None:
        return False

    return True


def _torch_is_constant(func, test_kwargs, *args, **kwargs):
    """Test whether a function is constant over ``num_pos`` random positions."""
    import torch  # pylint: disable=import-outside-toplevel

    test_kwargs = test_kwargs or {}
    num_pos = test_kwargs.get("num_pos", 5)
    seed = test_kwargs.get("seed", 9123)
    atol = test_kwargs.get("atol", 1e-8)
    rtol = test_kwargs.get("rtol", 0)

    torch.random.manual_seed(seed)
    original_output = func(*args, **kwargs)
    is_tuple_valued = isinstance(original_output, tuple)
    for _ in range(num_pos):
        rnd_args = tuple(torch.rand(np.shape(arg)) * 2 * np.pi - np.pi for arg in args)
        new_output = func(*rnd_args, **kwargs)
        if is_tuple_valued:
            if not all(
                np.allclose(new, orig, atol=atol, rtol=rtol)
                for new, orig in zip(new_output, original_output)
            ):
                return False
        else:
            if not np.allclose(new_output, original_output, atol=atol, rtol=rtol):
                return False

    return True


def _tf_is_independent(func, *args, **kwargs):
    """Test whether a function is independent of its arguments using
    a tensorflow GradientTape."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    with tf.GradientTape(persistent=True) as tape:
        out = func(*args, **kwargs)

    if isinstance(out, tuple):
        jac = [tape.jacobian(_out, args) for _out in out]
        return all(all(__jac is None for __jac in _jac) for _jac in jac)
    jac = tape.jacobian(out, args)
    return all(_jac is None for _jac in jac)


def _is_independent(func, interface, args, kwargs=None, torch_kwargs=None):
    """Test whether a function is independent of its input arguments."""
    kwargs = kwargs or {}
    if interface == "autograd":
        return _autograd_is_independent(func, *args, **kwargs)

    if interface == "jax":
        return _jax_is_independent(func, *args, **kwargs)

    if interface == "torch":
        warnings.warn(
            "The function _is_constant is not safe to use for the PyTorch interface. "
            "Make sure that sampling positions and evaluating the function at these positions "
            "is a sufficient test, or change the interface."
        )
        return _torch_is_constant(func, torch_kwargs, *args, **kwargs)

    if interface == "tf":
        return _tf_is_independent(func, *args, **kwargs)

    raise ValueError(f"Unknown interface: {interface}")
