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

import tensorflow as tf
from tensorflow.python.eager import context

import pennylane as qml
from pennylane.measurements import Shots

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _set_copy_tape(t, a):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, list(range(len(a))))
    return tc


def set_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_tape(t, a) for t, a in zip(tapes, params))


def _compute_vjp(dy, jacs, multi_measurements, has_partitioned_shots):
    # compute the vector-Jacobian product dy @ jac
    # for a list of dy's and Jacobian matrices.

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Entry with args=(dy=%s, jacs=%s, multi_measurements=%s, shots=%s) called by=%s",
            dy,
            jacs,
            multi_measurements,
            has_partitioned_shots,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    vjps = []

    for dy_, jac_, multi in zip(dy, jacs, multi_measurements):
        dy_ = dy_ if has_partitioned_shots else (dy_,)
        jac_ = jac_ if has_partitioned_shots else (jac_,)

        shot_vjps = []
        for d, j in zip(dy_, jac_):
            if multi:
                shot_vjps.append(qml.gradients.compute_vjp_multi(d, j))
            else:
                shot_vjps.append(qml.gradients.compute_vjp_single(d, j))

        vjp = qml.math.sum(qml.math.stack(shot_vjps), 0)

        if not context.executing_eagerly():
            vjp = qml.math.unstack(vjp)

        vjps.extend(vjp)

    return vjps


def _to_tensors(x):
    """
    Convert a nested tuple structure of arrays into a nested tuple
    structure of TF tensors
    """
    if isinstance(x, dict) or isinstance(x, list) and all(isinstance(i, dict) for i in x):
        # qml.counts returns a dict (list of dicts when broadcasted), can't form a valid tensor
        return x

    if isinstance(x, tuple):
        return tuple(_to_tensors(x_) for x_ in x)

    return tf.convert_to_tensor(x)


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


def execute(tapes, execute_fn, jpc, differentiable=False):
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

    # assumes all tapes have the same shot vector
    has_partitioned_shots = tapes[0].shots.has_partitioned_shots

    for tape in tapes:
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]

        # store all unwrapped parameters
        params_unwrapped.append(
            [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in params]
        )

    res = execute_fn(tuple(tapes))
    res = tuple(_to_tensors(r) for r in res)  # convert output to TensorFlow tensors

    @tf.custom_gradient
    def _execute(*parameters):  # pylint:disable=unused-argument
        def grad_fn(*dy, **tfkwargs):
            # reconstruct the nested structure of dy
            if differentiable:
                inner_tapes = tuple(tapes)
            else:
                inner_tapes = set_parameters_on_copy(tapes, params_unwrapped)

            dy = _res_restructured(dy, tapes)

            if tf.executing_eagerly():
                vjps = jpc.compute_vjp(inner_tapes, dy)
            else:
                jacs = jpc.compute_jacobian(inner_tapes)
                multi_measurements = [len(t.measurements) > 1 for t in tapes]
                vjps = _compute_vjp(dy, jacs, multi_measurements, has_partitioned_shots)

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
