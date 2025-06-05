# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum natural gradient optimizer for Jax/Catalyst interface"""

from pennylane import math
from pennylane.compiler import active_compiler
from pennylane.gradients.metric_tensor import metric_tensor

has_catalyst = True
try:
    import catalyst
except ImportError:
    has_catalyst = False

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


class QNGOptimizerJax:

    def __init__(self, stepsize=0.01, approx="block-diag", lam=0):
        self.stepsize = stepsize
        self.approx = approx
        self.lam = lam

    def init(self, params):
        return None

    def step(self, qnode, params, state, **kwargs):
        mt = self._get_metric_tensor(qnode, params, **kwargs)
        grad = self._get_grad(qnode, params, **kwargs)
        new_params, new_state = self._apply_grad(mt, grad, params, state)
        return new_params, new_state

    def step_and_cost(self, qnode, params, state, **kwargs):
        mt = self._get_metric_tensor(qnode, params, **kwargs)
        cost, grad = self._get_value_and_grad(qnode, params, **kwargs)
        new_params, new_state = self._apply_grad(mt, grad, params, state)
        return new_params, new_state, cost

    @staticmethod
    def _get_grad(qnode, params, **kwargs):
        if active_compiler() == "catalyst":
            if has_catalyst:
                return catalyst.grad(qnode)(params, **kwargs)
            else:
                raise ImportError("Catalyst is required.")
        if has_jax:
            return jax.grad(qnode)(params, **kwargs)
        else:
            raise ImportError("Jax is required.")

    @staticmethod
    def _get_value_and_grad(qnode, params, **kwargs):
        if active_compiler() == "catalyst":
            if has_catalyst:
                return catalyst.value_and_grad(qnode)(params, **kwargs)
            else:
                raise ImportError("Catalyst is required.")
        if has_jax:
            return jax.value_and_grad(qnode)(params, **kwargs)
        else:
            raise ImportError("Jax is required.")

    def _get_metric_tensor(self, qnode, params, **kwargs):
        mt = metric_tensor(qnode, approx=self.approx)(params, **kwargs)
        shape = math.shape(mt)
        size = 1 if shape == () else math.prod(shape[: len(shape) // 2])
        mt_matrix = math.reshape(mt, shape=(size, size))
        if self.lam != 0:
            mt_matrix += self.lam * math.eye(size, like=mt_matrix)
        return mt_matrix

    def _apply_grad(self, mt, grad, params, state):
        shape = math.shape(grad)
        grad_flat = math.flatten(grad)
        update_flat = math.linalg.pinv(mt) @ grad_flat
        update = math.reshape(update_flat, shape=shape)
        new_params = params - self.stepsize * update
        return new_params, state
