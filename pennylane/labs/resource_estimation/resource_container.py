# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Base classes for the resource objects and resource estimation."""


class CompressedResourceOp:

    def __init__(self, op_type: type, params_tuple: tuple) -> None:
        r"""Instantiate the light weight class corressponding to the operator type and parameters. 

        Args:
            op_type (Type): the PennyLane type of the operation 
            params_tuple (tuple): [tuple(tuple(parameter_name: str, parameter_value))] contains the minimal 
                pairs of parameter names and values required to compute the resources for the given operator!
        
        .. details::

            This representation is the minimal amount of information required to estimate resources for the operator.

            **Example**

            >>> op_tp = CompressedResourceOp(qml.Hadamard, (("num_wires", 1),))
            >>> print(op_tp)
            Hadamard(num_wires=1)
   
            >>> op_tp = CompressedResourceOp(
                    qml.QSVT, 
                    (
                        ("num_wires", 5), 
                        ("num_angles", 100),
                    ),
                )
            >>> print(op_tp)
            QSVT(num_wires=5, num_angles=100)
        """
        self.op_type = op_type
        self.params = params_tuple
    
    def __hash__(self) -> int:
        return hash((self.op_type.__name__, self.params))
    
    def __eq__(self, other: object) -> bool:
        return (self.op_type == other.op_type) and (dict(self.params) == dict(other.params))
    
    def __repr__(self) -> str:
        op_type_str = self.op_type.__name__ + "("
        params_str = ", ".join([f"{param[0]}={param[1]}" for param in self.params]) + ")"

        return op_type_str + params_str
