# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native derivatives supplied through the legacy optimizers' explicit hooks."""

import jax
import numpy as np

import pennylane as qp


def _native_args_and_argnums(args):
    """Preserve legacy trainability while converting positional inputs to JAX."""
    indices = tuple(index for index, arg in enumerate(args) if getattr(arg, "requires_grad", False))
    argnums = indices[0] if len(indices) == 1 else indices
    native_args = tuple(jax.numpy.asarray(qp.math.toarray(arg)) for arg in args)
    return native_args, argnums


def native_gradient(qnode):
    """Return a NumPy gradient for a legacy optimizer's ``grad_fn`` hook."""
    native_qnode = qp.QNode(
        qnode.func, qnode.device, interface="jax", diff_method=qnode.diff_method
    )

    def gradient(*args, **kwargs):
        native_args, argnums = _native_args_and_argnums(args)
        derivative = qp.grad(native_qnode, argnums=argnums)
        result = derivative(*native_args, **kwargs)
        gradient.forward = np.asarray(qnode(*args, **kwargs))
        return jax.tree_util.tree_map(lambda value: np.array(value, copy=True), result)

    gradient.forward = None
    return gradient


def native_metric_tensor(qnode, approx="block-diag"):
    """Return a NumPy metric for a legacy optimizer's ``metric_tensor_fn`` hook."""
    native_qnode = qp.QNode(
        qnode.func, qnode.device, interface="jax", diff_method=qnode.diff_method
    )

    def metric(*args, **kwargs):
        native_args, argnums = _native_args_and_argnums(args)
        result = qp.metric_tensor(native_qnode, approx=approx, argnums=argnums)(
            *native_args, **kwargs
        )
        return jax.tree_util.tree_map(lambda value: np.array(value, copy=True), result)

    return metric
