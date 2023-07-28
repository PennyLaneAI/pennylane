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

import jax
from jax import numpy as jnp

import pennylane as qml

dtype = jnp.float64


def _numeric_type_to_dtype(numeric_type):
    """Auxiliary function for converting from Python numeric types to JAX
    dtypes based on the precision defined for the interface."""

    single_precision = dtype is jnp.float32
    if numeric_type is int:
        return jnp.int32 if single_precision else jnp.int64

    if numeric_type is float:
        return jnp.float32 if single_precision else jnp.float64

    # numeric_type is complex
    return jnp.complex64 if single_precision else jnp.complex128


def _create_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""

    shape = tape.shape(device)
    if len(tape.measurements) == 1:
        tape_dtype = _numeric_type_to_dtype(tape.numeric_type)
        return jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

    tape_dtype = tuple(_numeric_type_to_dtype(elem) for elem in tape.numeric_type)
    return tuple(jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(shape, tape_dtype))


def _jac_shape_dtype_struct(tape: "qml.tape.QuantumScript", device: "qml.Device"):
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliX(0)), qml.probs(0)])
    >>> dev = qml.devices.experimental.DefaultQubit2()
    >>> _jac_shape_dtype_struct(tape, dev)
    (ShapeDtypeStruct(shape=(), dtype=float64),
    ShapeDtypeStruct(shape=(2,), dtype=float64))
    >>> tapes, fn = qml.gradients.param_shift(tape)
    >>> fn(dev.execute(tapes))
    (array(0.), array([-0.42073549,  0.42073549]))
    """
    shape_and_dtype = _create_shape_dtype_struct(tape, device)
    if len(tape.trainable_params) == 1:
        return shape_and_dtype
    if len(tape.measurements) == 1:
        return tuple(shape_and_dtype for _ in tape.trainable_params)
    return tuple(tuple(_s for _ in tape.trainable_params) for _s in shape_and_dtype)


def make_pure_callback(device, config):
    def pure_callback_execution(tapes):
        shape_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

        parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)

        def callback_fn(params):
            new_tapes = tuple(
                t.bind_new_parameters(p, list(range(len(p)))) for t, p in zip(tapes, params)
            )
            new_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in new_tapes)
            return device.execute(new_tapes, execution_config=config)

        return jax.pure_callback(callback_fn, shape_dtype_structs, parameters)

    return pure_callback_execution


def _old_device_jac_via_callback(tapes, device, gradient_kwargs):
    parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)
    shape_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)

    def wrapper(inner_params):
        new_tapes = tuple(
            t.bind_new_parameters(p, list(range(len(p)))) for t, p in zip(tapes, inner_params)
        )
        new_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in new_tapes)
        return device.gradients(new_tapes, **gradient_kwargs)

    return jax.pure_callback(wrapper, shape_dtype_structs, parameters)


def _new_device_jac_via_callback(tapes, device, config):
    parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)
    shape_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)

    def wrapper(inner_params):
        new_tapes = tuple(
            t.bind_new_parameters(p, list(range(len(p)))) for t, p in zip(tapes, inner_params)
        )
        new_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in new_tapes)
        return device.compute_derivatives(new_tapes, execution_config=config)

    return jax.pure_callback(wrapper, shape_dtype_structs, parameters)


def _new_device_execute_and_jac(tapes, device, config):
    parameters = tuple(tuple(t.get_parameters(trainable_only=False)) for t in tapes)
    jac_dtype_structs = tuple(_jac_shape_dtype_struct(t, device) for t in tapes)
    res_dtype_structs = tuple(_create_shape_dtype_struct(t, device) for t in tapes)

    def wrapper(inner_params):
        new_tapes = tuple(
            t.bind_new_parameters(p, list(range(len(p)))) for t, p in zip(tapes, inner_params)
        )
        new_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in new_tapes)
        return device.execute_and_compute_derivatives(new_tapes, execution_config=config)

    return jax.pure_callback(wrapper, (res_dtype_structs, jac_dtype_structs), parameters)
