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

Generates parametrizations of operators to test in test_assert_valid.py.
"""
from inspect import getmembers, isclass

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Channel, Operation, Operator, StatePrepBase
from pennylane.ops.op_math import ChangeOpBasis
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation
from pennylane.ops.op_math.pow import PowOperation
from pennylane.templates.subroutines.time_evolution.trotter import TrotterizedQfunc


def _trotterize_qfunc_dummy(time, theta, phi, wires, flip=False):
    qml.RX(time * theta, wires[0])
    qml.RY(time * phi, wires[0])
    if flip:
        qml.CNOT(wires)


_INSTANCES_TO_TEST = [
    (qml.measurements.MidMeasureMP(wires=0), {"skip_capture": True}),
    (ChangeOpBasis(qml.PauliX(0), qml.PauliZ(0)), {}),
    (qml.sum(qml.PauliX(0), qml.PauliZ(0)), {}),
    (qml.sum(qml.X(0), qml.X(0), qml.Z(0), qml.Z(0)), {}),
    (qml.BasisState([1], wires=[0]), {"skip_differentiation": True}),
    (qml.ControlledQubitUnitary(np.eye(2), wires=[1, 0]), {"skip_differentiation": True}),
    (
        qml.ControlledQubitUnitary(np.eye(4), wires=[1, 2, 0], control_values=[0]),
        {"skip_differentiation": True},
    ),
    (
        qml.QubitChannel([np.array([[1, 0], [0, 0.8]]), np.array([[0, 0.6], [0, 0]])], wires=0),
        {"skip_differentiation": True},
    ),
    (qml.MultiControlledX(wires=[0, 1]), {}),
    (qml.Projector([1], 0), {"skip_differentiation": True}),
    (qml.Projector([1, 0], 0), {"skip_differentiation": True}),
    (qml.DiagonalQubitUnitary([1, 1, 1, 1], wires=[0, 1]), {"skip_differentiation": True}),
    (qml.QubitUnitary(np.eye(2), wires=[0]), {"skip_differentiation": True}),
    (qml.QubitUnitary(np.eye(4), wires=[0, 1]), {"skip_differentiation": True}),
    (
        qml.QubitUnitary(qml.Rot.compute_matrix(0.1, 0.2, 0.3), wires=[0]),
        {"skip_differentiation": True},
    ),
    (qml.SpecialUnitary([1, 1, 1], 0), {"skip_differentiation": True}),
    (qml.IntegerComparator(1, wires=[0, 1]), {"skip_differentiation": True}),
    (qml.PauliRot(1.1, "X", wires=[0]), {}),
    (qml.StatePrep([0, 1], 0), {"skip_differentiation": True}),
    (qml.PCPhase(0.27, dim=2, wires=[0, 1]), {}),
    (qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]), {"skip_differentiation": True}),
    (qml.adjoint(qml.PauliX(0)), {}),
    (qml.adjoint(qml.RX(1.1, 0)), {}),
    (qml.ops.LinearCombination([1.1, 2.2], [qml.PauliX(0), qml.PauliZ(0)]), {}),
    (qml.s_prod(1.1, qml.RX(1.1, 0)), {}),
    (qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0)), {}),
    (qml.ctrl(qml.RX(1.1, 0), 1), {}),
    (qml.exp(qml.PauliX(0), 1.1), {}),
    (qml.pow(qml.IsingXX(1.1, [0, 1]), 2.5), {}),
    (qml.ops.Evolution(qml.PauliX(0), 5.2), {}),
    (qml.QutritBasisState([1, 2, 0], wires=[0, 1, 2]), {"skip_differentiation": True}),
    (qml.resource.FirstQuantization(1, 2, 1), {}),
    (qml.prod(qml.RX(1.1, 0), qml.RY(2.2, 0), qml.RZ(3.3, 1)), {}),
    (qml.Snapshot(measurement=qml.expval(qml.Z(0)), tag="hi"), {}),
    (qml.Snapshot(tag="tag"), {}),
    (qml.Identity(0), {}),
    (
        TrotterizedQfunc(
            0.1,
            2.3,
            -4.5,
            qfunc=_trotterize_qfunc_dummy,
            n=10,
            order=2,
            wires=[1, 2],
            flip=True,
        ),
        {"skip_pickle": True},
    ),
    (
        qml.SelectPauliRot(
            np.array(
                [
                    0.69307448,
                    0.2574346,
                    0.84850003,
                    0.06706336,
                    0.33502536,
                    0.79254386,
                    0.76929339,
                    0.66070049,
                ]
            ),
            control_wires=[0, 1, 2],
            target_wire=3,
            rot_axis="Y",
        ),
        {},
    ),
]
"""Valid operator instances that could not be auto-generated."""


_INSTANCES_TO_FAIL = [
    (
        qml.SparseHamiltonian(qml.Hamiltonian([1.1], [qml.PauliX(0)]).sparse_matrix(), [0]),
        AssertionError,  # each data element must be tensorlike
    ),
    (
        qml.PauliError("X", 0.5, wires=0),
        DeviceError,  # not supported with default.qubit and does not provide a decomposition
    ),
    (
        qml.THermitian(np.eye(3), wires=0),
        (AssertionError, ValueError),  # qutrit ops fail validation
    ),
    (
        qml.ops.qubit.special_unitary.TmpPauliRot(1.1, "X", [0]),
        AssertionError,  # private type with has_matrix=False despite having one
    ),
    (
        qml.ops.Conditional(qml.measure(1), qml.S(0)),
        AssertionError,  # needs flattening helpers to be updated, also cannot be pickled
    ),
    (
        qml.GlobalPhase(1.1),
        AssertionError,  # empty decomposition, matrix differs from decomp's matrix
    ),
    (
        qml.pulse.ParametrizedEvolution(qml.PauliX(0) + sum * qml.PauliZ(0)),
        ValueError,  # binding parameters fail, and more
    ),
    (
        qml.resource.DoubleFactorization(np.eye(2), np.arange(16).reshape((2,) * 4)),
        TypeError,  # op.eigvals is a list (overwritten in the init)
    ),
]
"""
List[Tuple[Operator, Type[Exception]]]: List of tuples containing Operator instances that could
not be auto-generated, along with the exception type raised when trying to assert its validity.

These operators need to break PL conventions, and each one's reason is specified in a comment.
"""


_ABSTRACT_OR_META_TYPES = {
    Adjoint,
    AdjointOperation,
    Operator,
    Operation,
    Channel,
    qml.ops.Projector,
    qml.ops.SymbolicOp,
    qml.ops.ScalarSymbolicOp,
    qml.ops.Pow,
    qml.ops.CompositeOp,
    qml.ops.Controlled,
    qml.ops.ControlledOp,
    qml.ops.qubit.BasisStateProjector,
    qml.ops.qubit.StateVectorProjector,
    StatePrepBase,
    qml.resource.ResourcesOperation,
    qml.resource.ErrorOperation,
    PowOperation,
    qml.StatePrep,
    qml.FromBloq,
    qml.allocation.Allocate,  # no integer wires
    qml.allocation.Deallocate,  # no integer wires
}
"""Types that should not have actual instances created."""


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


_CLASSES_TO_TEST = (
    set(get_all_classes(Operator))
    - {i[1] for i in getmembers(qml.templates) if isclass(i[1]) and issubclass(i[1], Operator)}
    - {type(op) for (op, _) in _INSTANCES_TO_TEST}
    - {type(op) for (op, _) in _INSTANCES_TO_FAIL}
)
"""All operators, except those tested manually, abstract/meta classes, and templates."""


@pytest.fixture(params=sorted(_CLASSES_TO_TEST, key=lambda op: op.__name__))
def class_to_validate(request):
    yield request.param


@pytest.fixture(params=_INSTANCES_TO_TEST)
def valid_instance_and_kwargs(request):
    yield request.param


@pytest.fixture(params=_INSTANCES_TO_FAIL)
def invalid_instance_and_error(request):
    yield request.param
