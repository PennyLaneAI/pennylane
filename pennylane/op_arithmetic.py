# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Functions for activating, deactivating and checking the
use of new arithmetic operators via dunder methods
"""
# pylint: disable=global-statement
__use_op_arithmetic = False


def enable_op_arithmetic():
    """
    Change dunder methods to return arithmetic operators instead of Hamiltonians and Tensors

    **Example**

    >>> qml.active_op_arithmetic()
    False
    >>> type(qml.PauliX(0) @ qml.PauliZ(1))
    <class 'pennylane.operation.Tensor'>
    >>> qml.enable_op_arithmetic()
    >>> type(qml.PauliX(0) @ qml.PauliZ(1))
    <class 'pennylane.ops.op_math.prod.Prod'>
    """
    global __use_op_arithmetic
    __use_op_arithmetic = True


def disable_op_arithmetic():
    """
    Change dunder methods to return Hamiltonians and Tensors instead of arithmetic operators

    **Example**

    >>> qml.active_op_arithmetic()
    True
    >>> type(qml.PauliX(0) @ qml.PauliZ(1))
    <class 'pennylane.ops.op_math.prod.Prod'>
    >>> qml.disable_op_arithmetic()
    >>> type(qml.PauliX(0) @ qml.PauliZ(1))
    <class 'pennylane.operation.Tensor'>
    """
    global __use_op_arithmetic
    __use_op_arithmetic = False


def active_op_arithmetic():
    """
    Function that checks if the new arithmetic operator dunders are active

    Returns:
        bool: Returns ``True`` if the new arithmetic operator dunders are active

    **Example**

    >>> qml.active_op_arithmetic()
    False
    >>> qml.enable_op_arithmetic()
    >>> qml.active_op_arithmetic()
    True
    """
    return __use_op_arithmetic
