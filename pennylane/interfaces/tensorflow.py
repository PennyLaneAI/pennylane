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
This module contains functions for adding the TensorFlow interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments,too-many-branches
import inspect
import logging
import warnings

import tensorflow as tf
from tensorflow.python.eager import context

import pennylane as qml
from pennylane.measurements import Shots

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _set_copy_tape(t, a):
    """Copy a given tape with operations and set parameters"""
    return t.bind_new_parameters(a, list(range(len(a))))


def set_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_tape(t, a) for t, a in zip(tapes, params))


def _to_tensors(x):
    """
    Convert a nested tuple structure of arrays into a nested tuple
    structure of TF tensors
    """
    if isinstance(x, dict):
        # qml.counts returns a dict (list of dicts when broadcasted), can't form a valid tensor
        return x

    if isinstance(x, (tuple, list)):
        return tuple(_to_tensors(x_) for x_ in x)

    return tf.convert_to_tensor(x)


def _recursive_conj(dy):
    if isinstance(dy, (tf.Variable, tf.Tensor)):
        return tf.math.conj(dy)
    return tuple(_recursive_conj(d) for d in dy)


def _res_restructured(res, tapes):
    """
    Reconstruct the nested tuple structure of the output of a list of tapes
    """
    start = 0
    res_nested = []
    for tape in tapes:
        tape_shots = tape.shots or Shots(1)
        shot_res_nested = []
        num_meas = len(tape.measurements)

        for _ in range(tape_shots.num_copies):
            shot_res = tuple(res[start : start + num_meas])
            shot_res_nested.append(shot_res[0] if num_meas == 1 else shot_res)
            start += num_meas

        res_nested.append(
            tuple(shot_res_nested) if tape_shots.has_partitioned_shots else shot_res_nested[0]
        )

    return tuple(res_nested)


def execute(tapes, execute_fn, jpc, device=None, differentiable=False):
    """Execute a batch of tapes with TensorFlow parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        list[list[tf.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument

    parameters = []
    params_unwrapped = []

    for tape in tapes:
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]

        # store all unwrapped parameters
        params_unwrapped.append(
            [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in params]
        )

    tapes = tuple(tapes)
    res = _to_tensors(execute_fn(tapes))

    @tf.custom_gradient
    def _execute(*parameters):  # pylint:disable=unused-argument
        def grad_fn(*dy, **tfkwargs):
            # TF obeys the dL/dz_conj convention instead of the
            # dL/dz convention of PennyLane, autograd and jax. This converts between the formats
            dy = _recursive_conj(dy)

            if not differentiable or not context.executing_eagerly():
                if differentiable:
                    warnings.warn(
                        "PennyLane cannot provide the higher order derivatives of jacobians."
                    )
                inner_tapes = set_parameters_on_copy(tapes, params_unwrapped)
            else:
                inner_tapes = tapes

            # reconstruct the nested structure of dy
            dy = _res_restructured(dy, tapes)

            vjps = jpc.compute_vjp(inner_tapes, dy, original_tapes=tapes)

            if isinstance(vjps, tuple):
                extended_vjps = []
                for vjp in vjps:
                    if vjp is not None and 0 not in qml.math.shape(vjp):
                        extended_vjps.extend(qml.math.unstack(vjp))
                vjps = tuple(extended_vjps)

            variables = tfkwargs.get("variables")
            return (vjps, variables) if variables is not None else vjps

        return res, grad_fn

    return _execute(*parameters)
