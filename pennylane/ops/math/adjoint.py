# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines an operation modifier that indicates the adjoint of an operator.
"""
import pennylane as qml
from pennylane.operation import Operator, AnyWires


class Adjoint(Operator):

    num_wires = AnyWires

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters['base'] = base
        super().__init__(*base.parameters, wires=base.wires, do_queue=do_queue, id=id)
        self._name = f"Adjoint({self.base.name})"

    def queue(self, context=qml.QueuingContext):
        try:
            context.update_info(self.base, owner=self)
        except qml.queuing.QueuingError:
            self.base.queue(context=context)
            context.update_info(self.base, owner=self)

        context.append(self, owns=self.base)

        return self
    
    @property
    def num_params(self):
        return self.base.num_params
    
    @property
    def parameters(self):
        return self.base.parameters
    
    @property
    def wires(self):
        return self.base.wires

    def label(self, decimals=None, base_label=None):
        return self.base.label(decimals, base_label)+"†"
    
    @staticmethod
    def compute_matrix(*params, base=None, base_hyperparameters=None):

        base_matrix = base.compute_matrix(*params, **base_hyperparameters)
        return qml.math.transpose(qml.math.conjugate(base_matrix))
    
    @classmethod
    def compute_decomposition(cls, *params, wires, base=None, base_hyperparameters=None):
        try:
            return base.adjoint()
        except qml.operation.AdjointUndefinedError:
            base_decomp = base.compute_decomposition(*params, wires, **base_hyperparameters)
            return [Adjoint(op) for op in reversed(base_decomp)]
        
    @staticmethod
    def compute_sparse_matrix(*params, base=None, **hyperparams):
        base_matrix = base.compute_sparse_matrix(*params, **hyperparams)
        return qml.math.transpose(qml.math.conjugate(base_matrix))
    
    @staticmethod
    def compute_eigvals(*params, base=None, **hyperparams):
        base_eigvals = base.compute_eigvals(*params, **hyperparams)
        return [qml.math.conjugate(x) for x in base_eigvals]

    @property
    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles
        return -phi, -theta, -omega

    # get_parameter_shift ?

    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    # generator

