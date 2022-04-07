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
from pennylane.operation import Operator, Operation, AnyWires, AdjointUndefinedError
from pennylane.queuing import QueuingContext
from pennylane.math import transpose, conjugate

def adjoint(op):
    try:
        new_op = op.adjoint()
        QueuingContext.safe_update_info(op, owner=new_op)
        return new_op
    except AdjointUndefinedError:
        return Adjoint(op)

class Adjoint(Operator):

    num_wires = AnyWires

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters['base'] = base
        super().__init__(*base.parameters, wires=base.wires, do_queue=do_queue, id=id)
        self._name = f"Adjoint({self.base.name})"

    def queue(self, context=QueuingContext):
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
    def compute_matrix(*params, base=None):

        base_matrix = base.compute_matrix(*params, **base.hyperparameters)
        return transpose(conjugate(base_matrix))
    
    @classmethod
    def compute_decomposition(cls, *params, wires, base=None):
        try:
            return base.adjoint()
        except AdjointUndefinedError:
            base_decomp = base.compute_decomposition(*params, wires, **base.hyperparameters)
            return [Adjoint(op) for op in reversed(base_decomp)]
        
    @staticmethod
    def compute_sparse_matrix(*params, base=None):
        base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
        return transpose(conjugate(base_matrix))
    
    @staticmethod
    def compute_eigvals(*params, base=None):
        base_eigvals = base.compute_eigvals(*params, **base.hyperparameters)
        return [conjugate(x) for x in base_eigvals]

    @property
    def has_matrix(self):
        return self.base.has_matrix


    ## Operation specific properties ##########################################

    @property
    def grad_method(self):
        return self.base.grad_method

    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    @property
    def basis(self):
        return self.base.basis

    @property
    def control_wires(self):
        return self.base.control_wires

    @property
    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles
        return -phi, -theta, -omega

    # get_parameter_shift ?
    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    def generator(self):
        if isinstance(self, Operation): # stand in for being unitary and inverse=adjoint
            return -1.0 * self.base.generator()
        return super().generator()

    def adjoint(self):
        return self.base
