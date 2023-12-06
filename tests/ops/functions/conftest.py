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
import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import Operator, Operation, Observable, Tensor, Channel
from pennylane.ops.op_math.adjoint import Adjoint, AdjointObs, AdjointOperation, AdjointOpObs

# these types need manual registration below
_SKIP_OP_TYPES = {
    # ops composed of more than one thing
    Adjoint,
    Tensor,
    qml.Hamiltonian,
    qml.ops.Pow,
    qml.ops.SProd,
    qml.ops.Prod,
    qml.ops.Sum,
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
    qml.ops.qubit.Projector,  # both basis-state and state-vector needed
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
    qml.resource.DoubleFactorization,
    qml.resource.ResourcesOperation,
    # templates
    *[i[1] for i in getmembers(qml.templates) if isclass(i[1]) and issubclass(i[1], Operator)],
}

# valid instances of types that don't get auto-generated properly
_REGISTERED_INSTANCES = [
    qml.BlockEncode(np.random.rand(2, 2), wires=[0, 1]),
    qml.QutritUnitary(np.eye(3), wires=[0]),
    qml.ControlledQutritUnitary(np.random.rand(3, 3), wires=[0], control_wires=[1]),
]

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
    subs = c.__subclasses__()
    classes = [] if c in _ABSTRACT_OR_META_TYPES else [c]
    for sub in subs:
        classes.extend(get_all_classes(sub))
    return classes


def create_op_instance(c):
    """Given an Operator class, create an instance of it."""
    n_wires = c.num_wires
    if n_wires == qml.operation.AllWires:
        raise ValueError("AllWires unsupported. Op needing whitelisting:", c)
    if n_wires == qml.operation.AnyWires:
        n_wires = 1

    wires = qml.wires.Wires(range(n_wires))
    if (num_params := c.num_params) == 0:
        return c(wires)

    # get ndim_params
    ndim_params = c.ndim_params
    if isinstance(ndim_params, property):
        if isinstance(num_params, property):
            num_params = 1
        ndim_params = (0,) * num_params

    # turn ndim_params into valid params
    [dim] = set(ndim_params)
    params = [1] * len(ndim_params) if dim == 0 else [np.eye(dim)]

    return c(*params, wires=wires)


_AUTO_TYPES = (
    set(get_all_classes(Operator)) - _SKIP_OP_TYPES - {type(o) for o in _REGISTERED_INSTANCES}
)


def create_and_catch(c):
    try:
        return create_op_instance(c)
    except Exception as e:
        raise Exception(f"failed to generate instance for {c.__name__}: {e}") from e


@pytest.fixture(params=list(map(create_and_catch, _AUTO_TYPES)) + _REGISTERED_INSTANCES)
def op_to_validate(request):
    yield request.param
