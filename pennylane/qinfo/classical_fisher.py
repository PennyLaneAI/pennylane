# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable classical fisher information"""
# pylint: disable=import-outside-toplevel
import functools
import pennylane as qml

from pennylane.transforms import batch_transform

# TODO: create qml.jacobian and replace by it
def _torch_jac(circ):
    """Torch jacobian as a callable function"""
    import torch

    def wrapper(*args, **kwargs):
        loss = functools.partial(circ, **kwargs)
        if len(args) > 1:
            return torch.autograd.functional.jacobian(loss, (args))
        return torch.autograd.functional.jacobian(loss, *args)

    return wrapper


# TODO: create qml.jacobian and replace by it
def _tf_jac(circ):
    """TF jacobian as a callable function"""
    import tensorflow as tf

    def wrapper(*args, **kwargs):
        with tf.GradientTape() as tape:
            loss = circ(*args, **kwargs)
        return tape.jacobian(loss, (args))

    return wrapper


def classical_fisher(qnode, argnums=0):
    """Computing the classical fisher information matrix (classical_fisher) using the jacobian of the output probabilities
    as described in eq. (15) in https://arxiv.org/abs/2103.15191
    """
    new_qnode = _make_probs(qnode, post_processing_fn=lambda x: qml.math.squeeze(qml.math.stack(x)))

    interface = qnode.interface

    if interface == "jax":
        import jax

        jac = jax.jacobian(new_qnode, argnums=argnums)

    if interface == "torch":
        jac = _torch_jac(new_qnode)

    if interface == "autograd":
        jac = qml.jacobian(new_qnode)

    if interface == "tf":
        jac = _tf_jac(new_qnode)

    def wrapper(*args, **kwargs):
        p = qnode(*args, **kwargs)
        j = jac(*args, **kwargs)

        # In case multiple variables are used
        if isinstance(j, tuple) and len(j) > 1:
            res = []
            for j_i in j:
                if interface == "tf":
                    j_i = qml.math.transpose(qml.math.cast(j_i, dtype=p.dtype))

                res.append(_compute_cfim(p, j_i))

            return res

        return _compute_cfim(p, j)

    return wrapper


def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64

    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = qml.math.divide(one_over_p, nonzeros_p)

    # Multiply dp and p
    dp = qml.math.cast(dp, dtype=p.dtype)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    return dp_over_p @ dp  # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)


@batch_transform
def _make_probs(tape, wires=None, post_processing_fn=None):
    """Ignores the return types of any qnode and creates a new one that outputs probabilities"""
    if wires is None:
        wires = tape.wires

    with qml.tape.QuantumTape() as new_tape:
        for op in tape.operations:
            qml.apply(op)
        qml.probs(wires=wires)

    if post_processing_fn is None:
        post_processing_fn = lambda x: qml.math.squeeze(qml.math.stack(x))

    return [new_tape], post_processing_fn
