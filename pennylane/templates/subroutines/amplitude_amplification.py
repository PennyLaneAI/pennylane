# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule contains the template for Amplitude Amplification.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation

class AmpAmp(Operation):

    def __init__(self, U, O, iters = 1, fixed_point = False, aux_wire = None):

      self.operands = [U, O]
      self.queue()

      self.U = U
      self.O = O
      self.aux_wire = aux_wire
      self.fixed_point = fixed_point
      self.n_iterations = iters
      self.gamma = 0.99

      if fixed_point:
        super().__init__(wires = U.wires + qml.wires.Wires(aux_wire))
      else:
        super().__init__(wires = U.wires)


    def decomposition(self):

        alphas, betas = self.get_fixed_point_angles()

        ops = []

        if self.fixed_point:

          for iter in range(self.n_iterations // 2):


            ops.append(qml.Hadamard(wires=self.aux_wire))
            ops.append(qml.ctrl(self.O, control=self.aux_wire))
            ops.append(qml.Hadamard(wires=self.aux_wire))
            ops.append(qml.PhaseShift(-betas[iter], wires=self.aux_wire))
            ops.append(qml.Hadamard(wires=self.aux_wire))
            ops.append(qml.ctrl(self.O, control=self.aux_wire))
            ops.append(qml.Hadamard(wires=self.aux_wire))

            ops.append(qml.Reflection(self.U, alphas[iter]))

        else:
          for _ in range(self.n_iterations):

            ops.append(self.O)
            ops.append(qml.Reflection(self.generator, np.pi))


        return ops


    def get_fixed_point_angles(self):

      n_iterations = self.n_iterations

      alphas = [2 * np.arctan(1/(np.tan(2 * np.pi * j / n_iterations) * np.sqrt(1 - self.gamma ** 2))) for j in range(1, n_iterations // 2 + 1)]
      betas = [-alphas[-j] for j in range(1, n_iterations // 2 + 1)]
      return alphas[:n_iterations // 2 ], betas[:n_iterations // 2]


    def queue(self, context=qml.QueuingManager):
        for op in self.operands:
            context.remove(op)
        context.append(self)
        return self