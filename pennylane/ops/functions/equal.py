# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.equal function.
"""
import pennylane as qml


def equal(op1, op2, rtol=1e-5, atol=1e-9, check_interface=True, check_trainability=True):
    try:
        assert op1.__class__ is op2.__class__
        assert all(
            qml.math.allclose(d1, d2, rtol=rtol, atol=atol) for d1, d2 in zip(op1.data, op2.data)
        )
        assert op1.wires == op2.wires
        for kwarg in op1.hyperparameters:
            assert op1.hyperparameters[kwarg] == op2.hyperparameters[kwarg]

        if check_trainability:
            for params_1, params_2 in zip(op1.data, op2.data):
                try:
                    assert params_1.requires_grad == params_2.requires_grad
                except AttributeError:
                    return False

        if check_interface:
            for params_1, params_2 in zip(op1.data, op2.data):
                assert isinstance(params_1, type(params_2))

        return True
    except AssertionError:
        return False
