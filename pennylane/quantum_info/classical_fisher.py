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


def CFIM(qnode):
    """Computing the classical fisher information matrix (CFIM) using the jacobian of the output probabilities
    as described in eq. (15) in https://arxiv.org/abs/2103.15191
    """
    new_qnode = qml.transforms._make_probs(qnode)
    jac = qml.jacobian(new_qnode)

    def wrapper(*args, **kwargs):
        j = jac(*args, **kwargs)
        p = qnode(*args, **kwargs)

        if hasattr(j, "interface"):
            interface = getattr(j, "interface")
        else:
            interface = "autograd"

        cfim = _compute_cfim(p, j, interface)
        return cfim

    return wrapper

def CFIM_alt(qnode):
    """Computing the classical fisher information matrix (CFIM) by computing the hessian of log(p)
    as described in eq. (14) in https://arxiv.org/abs/2103.15191
    """
    new_qnode = qml.transforms._make_probs(qnode, post_processing_fn=lambda x: qml.math.squeeze(qml.math.log(qml.math.stack(x))))
    hessian = qml.jacobian(qml.jacobian(new_qnode)) # this is very slow, can be sped up with other interfaces

    def wrapper(*args, **kwargs):
        h = hessian(*args, **kwargs)                                # (2**n_wires, num_params, num_params)
        p = qnode(*args, **kwargs)[:, pnp.newaxis, pnp.newaxis]     # (2**n_wires, 1, 1)
        return -qml.math.sum(h*p, axis=0)                           # TODO: dont understand why I need the minus sign, most likely some autograd peculiarity in the hessian

    return wrapper


def _compute_cfim(p, dp, interface):
    """Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives"""
    # Assumes
    # dp shape: (n_probs, n_params)
    # p  shape: (n_probs)
    # outputs : (n_params, n_params)
    # by computing dp / p (n_params, n_probs)
    # and then (dp/p) @ dp (n_params, n_params)
    if interface == "jax":
        dp_over_p = dp.T / p
    else:
        dp_over_p = qml.math.divide(
            dp.T, p, where=(p != 0), out=qml.math.zeros_like(dp.T)
        )  # shape (n_params, n_probs)
    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dp
