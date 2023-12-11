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
Pytest configuration file for ops.functions submodule.

Generates a parametrization of valid operators to test.
"""
from inspect import getmembers, isclass
import pytest

import pennylane as qml
from pennylane.operation import Operator, Operation, Observable, Tensor, Channel
from pennylane.ops.op_math.adjoint import Adjoint, AdjointObs, AdjointOperation, AdjointOpObs

# if you would like to validate one of these operators, add an instance to the parametrization
# of `test_explicit_list_of_ops` in `test_assert_valid.py`, and kindly move it below the
# "manually validated ops" comment in this dict for easy inspection in the future.
_SKIP_OP_TYPES = {
    # ops composed of more than one thing
    Adjoint,
    Tensor,
    qml.Hamiltonian,
    qml.ops.Pow,
    qml.ops.SProd,
    qml.ops.Prod,
    qml.ops.Controlled,
    qml.ops.Exp,
    qml.ops.Evolution,
    qml.ops.Conditional,
    # fails for unknown reason - should be registered in the test parametrization below
    qml.ops.ControlledQubitUnitary,
    qml.QubitStateVector,
    qml.GlobalPhase,
    qml.QubitChannel,
    qml.SparseHamiltonian,
    qml.MultiControlledX,
    qml.Projector,  # both basis-state and state-vector needed
    qml.pulse.ParametrizedEvolution,
    qml.THermitian,
    qml.resource.FirstQuantization,
    qml.SpecialUnitary,
    qml.IntegerComparator,
    qml.PauliRot,
    qml.PauliError,
    qml.ops.qubit.special_unitary.TmpPauliRot,  # private object
    qml.StatePrep,
    qml.PCPhase,
    qml.BlockEncode,
    qml.resource.DoubleFactorization,
    qml.resource.ResourcesOperation,
    # templates
    *[i[1] for i in getmembers(qml.templates) if isclass(i[1]) and issubclass(i[1], Operator)],
    # manually validated ops
    qml.ops.Sum,
    qml.BasisState,
}


# types that should not have actual instances created
_ABSTRACT_OR_META_TYPES = {
    Operator,
    Operation,
    Observable,
    Channel,
    qml.ops.SymbolicOp,
    qml.ops.ScalarSymbolicOp,
    qml.ops.CompositeOp,
    qml.ops.ControlledOp,
    qml.ops.qubit.BasisStateProjector,
    qml.ops.qubit.StateVectorProjector,
    qml.ops.qubit.StatePrepBase,
    AdjointOpObs,
    AdjointOperation,
    AdjointObs,
}


def get_all_classes(c):
    """Recursive function to generate a flat list of all child classes of ``c``.
    (first called with ``Operator``)."""
    if c.__module__[:10] != "pennylane.":
        return []
    subs = c.__subclasses__()
    classes = [] if c in _ABSTRACT_OR_META_TYPES else [c]
    for sub in subs:
        classes.extend(get_all_classes(sub))
    return classes


_CLASSES_TO_TEST = set(get_all_classes(Operator)) - _SKIP_OP_TYPES


@pytest.fixture(params=sorted(_CLASSES_TO_TEST, key=lambda op: op.__name__))
def class_to_validate(request):
    yield request.param
