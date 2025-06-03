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

import catalyst
import jax

from pennylane import math
from pennylane.compiler import active_compiler
from pennylane.gradients.metric_tensor import metric_tensor
from pennylane.optimize.qng import _reshape_and_regularize


class QNGOptimizerJax:

    def __init__(self, stepsize=0.01, approx="block-diag", lam=0):
        self.stepsize = stepsize
        self.approx = approx
        self.lam = lam

    def init(self, params):
        return None

    def step(self, qnode, params, state, **kwargs):
        mt = metric_tensor(qnode, approx=self.approx)(params, **kwargs)
        grad_fn = catalyst.grad if active_compiler() == "catalyst" else jax.grad
        grad = grad_fn(qnode)(params, **kwargs)
        new_params = self.apply_grad(mt, grad, params)
        return new_params, state

    def apply_grad(self, mt, grad, params):
        mt = _reshape_and_regularize(mt, lam=self.lam)
        update = math.linalg.pinv(mt) @ grad
        new_params = params - self.stepsize * update
        return new_params
