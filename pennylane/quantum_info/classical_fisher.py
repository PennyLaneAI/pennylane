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

import pennylane as qml
import pennylane.numpy as pnp

import jax
import torch
import tensorflow as tf
import tensorflow.python.ops.numpy_ops.np_config as np_config


def torch_jac(circ):
    def wrapper(params):
        return torch.autograd.functional.jacobian(circ, (params))
    return wrapper

def tf_jac(circ):
    def wrapper(params):
        with tf.GradientTape() as tape:
            loss = circ(params)
        return tape.jacobian(loss, params)
    return wrapper


def CFIM(qnode):
    """Computing the classical fisher information matrix (CFIM) using the jacobian of the output probabilities
    as described in eq. (15) in https://arxiv.org/abs/2103.15191
    """
    new_qnode = qml.transforms._make_probs(
        qnode, post_processing_fn=lambda x: qml.math.squeeze(qml.math.stack(x))
    )

    interface = qnode.interface

    if interface == "jax":
        jac = jax.jacobian(new_qnode)
    if interface == "torch":
        jac = torch_jac(new_qnode)
    if interface == "autograd":
        jac = qml.jacobian(new_qnode)
    if interface == "tf":
        jac = tf_jac(new_qnode)
        np_config.enable_numpy_behavior() # this allows for the manipulations in _compute_cfim with tensors

    def wrapper(*args, **kwargs):

        j = jac(*args, **kwargs)
        p = qnode(*args, **kwargs)

        cfim = _compute_cfim(p, j, interface)
        return cfim

    return wrapper


def _compute_cfim(p, dp, interface):
    """Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`CFIM_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Check if any value in p is zero, and assign zeros for 1/p accordingly
    if any(qml.math.isclose(p, qml.math.zeros_like(p))) and not interface == "tf":
        mask = p!=0
        one_over_p = qml.math.zeros_like(p)
        if interface == "jax":
            one_over_p = one_over_p.at[mask].set(1/p[mask])
        else:
            one_over_p[mask] = 1/p[mask]
    elif interface == "tf":
        one_over_p = tf.math.divide_no_nan(qml.math.ones_like(p), p)
    else:
        one_over_p = 1/p
    dp_over_p = dp.T * one_over_p # creates (n_params, n_probs) array
    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    if interface == "torch":
        return dp_over_p @ qml.math.cast(dp, dtype=p.dtype)
    else:
        return dp_over_p @ dp



def CFIM_alt(qnode):
    """Computing the classical fisher information matrix (CFIM) by computing the hessian of log(p)
    as described in eq. (14) in https://arxiv.org/abs/2103.15191
    """
    new_qnode = qml.transforms._make_probs(
        qnode, post_processing_fn=lambda x: qml.math.squeeze(qml.math.log(qml.math.stack(x)))
    )
    hessian = qml.jacobian(
        qml.jacobian(new_qnode)
    )  # this is very slow, can be sped up with other interfaces

    def wrapper(*args, **kwargs):
        h = hessian(*args, **kwargs)  # (2**n_wires, num_params, num_params)
        p = qnode(*args, **kwargs)[:, pnp.newaxis, pnp.newaxis]  # (2**n_wires, 1, 1)
        return -qml.math.sum(
            h * p, axis=0
        )  # TODO: dont understand why I need the minus sign, most likely some autograd peculiarity in the hessian

    return wrapper

def _compute_cfim_alt(p, d_sqrt_p, interface=None):
    """Computes :math:`CFIM_{ij} = \sum_\ell (\partial_i \sqrt{p_\ell}) (\partial_i \sqrt{p_\ell})`"""
    if any(qml.math.isclose(p, 0)):
        mask = qml.math.where(qml.math.isclose(p, 0))
        n_zeros = len(mask[0])
        n_params = d_sqrt_p.shape[-1]
        d_sqrt_p[mask] = qml.math.zeros((n_zeros, n_params))
    return qml.math.tensordot(d_sqrt_p, d_sqrt_p, axes=[[0], [0]])

    # if interface == "jax":
    #     dp_over_p = dp.T / p
    # if interface == "torch":
    #     # TODO: Casting is currently necessary, otherwise dp_over_p @ dp yields mismatching scalar types
    #     p = p.type(torch.float64)
    #     dp = dp.type(torch.float64)

    #     one_over_p = torch.full_like(p, fill_value=0.0)
    #     mask = p != 0
    #     one_over_p[mask] = 1 / p[mask]
    #     dp_over_p = one_over_p * dp.T
    # if interface == "autograd":
    #     dp_over_p = qml.math.divide(
    #         dp.T, p, where=(p != 0), out=qml.math.zeros_like(dp.T)