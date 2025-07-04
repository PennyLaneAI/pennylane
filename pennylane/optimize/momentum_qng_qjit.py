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
"""Quantum natural gradient optimizer with momentum for Jax/Catalyst interface"""

from pennylane import math
from pennylane.optimize.qng import _reshape_and_regularize

from .qng_qjit import QNGOptimizerQJIT


class MomentumQNGOptimizerQJIT(QNGOptimizerQJIT):

    def __init__(self, stepsize=0.01, momentum=0.9, approx="block-diag", lam=0):
        super().__init__(stepsize, approx, lam)
        self.momentum = momentum

    def init(self, params):
        return math.zeros_like(params)

    def apply_grad(self, mt, grad, params, state):
        mt = _reshape_and_regularize(mt, lam=self.lam)
        update = math.linalg.pinv(mt) @ grad
        state = self.momentum * state + self.stepsize * update
        new_params = params - state
        return new_params, state
