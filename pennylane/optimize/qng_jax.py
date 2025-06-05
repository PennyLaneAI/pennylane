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
from pennylane.optimize.qng import _reshape_and_regularize

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
        grad = self._compute_grad(qnode, params, **kwargs)
        mt = metric_tensor(qnode, approx=self.approx)(params, **kwargs)
        mt = _reshape_and_regularize(mt, lam=self.lam)
        new_params, new_state = self._apply_grad(mt, grad, params, state)
        return new_params, new_state

    def step_and_cost(self, qnode, params, state, **kwargs):
        new_params, new_state = self.step(qnode, params, state, **kwargs)
        cost = qnode(params, **kwargs)
        return new_params, new_state, cost

    def _compute_grad(self, qnode, params, **kwargs):
        if active_compiler() == "catalyst":
            if has_catalyst:
                grad = catalyst.grad
            else:
                # temp error message
                raise ImportError("Catalyst is required.")
        else:
            if has_jax:
                grad = jax.grad
            else:
                # temp error message
                raise ImportError("Jax is required.")
        return grad(qnode)(params, **kwargs)

    def _apply_grad(self, mt, grad, params, state):
        update = math.linalg.pinv(mt) @ grad
        new_params = params - self.stepsize * update
        return new_params, state
