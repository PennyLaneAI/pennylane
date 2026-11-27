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
Pytest configuration file for the pennylane.testing tests.

Generates parametrizations of operators to test in test_assert_valid.py.
"""

from inspect import getmembers, isclass

import jax
import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import (
    Channel,
    LegacyOperator,
    Operation,
    Operator,
    StatePrepBase,
    StatePrepBase2,
)
from pennylane.decomposition_rules.qubit.non_parametric_ops import _PPRBasis
from pennylane.decomposition_rules.qubit.parametric_ops_multi_qubit import (
    _IsingXYBasis,
    _IsingXYRotations,
    _MultiRZLadder,
    _PauliRotBasis,
)
from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import _YToZBasis, _ZToYBasis
from pennylane.decomposition_rules.qubit.qchem_ops import (
    _SingleExcitationBasis,
    _SingleExcitationCRZBasis,
    _SingleExcitationRotations,
    _SingleExcitationRZRotations,
)
from pennylane.drawer.label import LabelledOp
from pennylane.exceptions import DeviceError
from pennylane.ops.op_math import ChangeOpBasis
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import Controlled2, ControlledOp2
from pennylane.ops.op_math.pow import PowOperation
from pennylane.ops.op_math.pow2 import Pow2
from pennylane.ops.op_math.symbolicop2 import SymbolicOp2
from pennylane.templates.subroutines.arithmetic.out_adder import _OutAdderTarget
from pennylane.templates.subroutines.arithmetic.out_multiplier import _OutMultiplierHadamards
from pennylane.templates.subroutines.arithmetic.phase_adder import _PhaseAdderBasis
from pennylane.templates.subroutines.arithmetic.temporary_and import _TemporaryANDBasis
from pennylane.templates.subroutines.qubitization_thc import _THCPrepare
from pennylane.templates.subroutines.select_pauli_rot import (
    _SelectPauliRotYBasis,
    _UniformRZ,
)
from pennylane.templates.subroutines.select_thc import _SelectTHCBasis, _THCIndexSwaps
from pennylane.templates.subroutines.time_evolution.trotter import TrotterizedQfunc
from pennylane.transforms.decompositions.crz_phase_gradient import _CRZPhaseGradientCompute
from pennylane.transforms.decompositions.select_pauli_rot_phase_gradient import (
    _SelectPauliRotPhaseGradientCompute,
)


def _trotterize_qfunc_dummy(time, theta, phi, wires, flip=False):
    qp.RX(time * theta, wires[0])
    qp.RY(time * phi, wires[0])
    if flip:
        qp.CNOT(wires)


def _subcircuit_instances():
    """Construct private decomposition helpers with their actual argument signatures."""
    thc_wires = qp.registers(qp.qubitization_thc_wires(1, 2, 1, 1))
    return [
        (_PPRBasis("XY", [0, 1], reverse=False), {}),
        (_IsingXYBasis([0, 1], reverse=False), {}),
        (_IsingXYRotations(0.37, [0, 1]), {}),
        (_MultiRZLadder([0, 1], reverse=False), {}),
        (_PauliRotBasis(1, "XY", [0, 1]), {}),
        (_YToZBasis([0]), {}),
        (_ZToYBasis([0]), {}),
        (_SingleExcitationBasis([0, 1], reverse=False), {}),
        (_SingleExcitationCRZBasis([0, 1]), {}),
        (_SingleExcitationRotations(0.37, [0, 1]), {}),
        (_SingleExcitationRZRotations(0.37, [0, 1]), {}),
        (_OutAdderTarget([0], [1], [2, 3], 4, []), {}),
        (_OutMultiplierHadamards([0, 1]), {}),
        # The arithmetic input is an integer, not a differentiable gate parameter.
        (_PhaseAdderBasis(1, [0, 1], reverse=False), {"skip_differentiation": True}),
        (_TemporaryANDBasis([0, 1]), {}),
        # The legacy differentiator cannot evaluate this helper's captured loop.
        (_UniformRZ(np.array([0.2, 0.3]), [0], [1]), {"skip_differentiation": True}),
        (_SelectPauliRotYBasis([0], reverse=False), {}),
        (_SelectPauliRotYBasis([0], reverse=True), {}),
        # Boolean angle tables cannot be replaced by arbitrary floating-point parameters.
        (
            _CRZPhaseGradientCompute(jax.numpy.array([True]), [0, 1], [2], [3]),
            {"skip_differentiation": True, "skip_bind_new_parameters": True},
        ),
        (
            _SelectPauliRotPhaseGradientCompute(
                jax.numpy.array([[False], [True]]), [0, 1], [2], [3], []
            ),
            {"skip_differentiation": True, "skip_bind_new_parameters": True},
        ),
        (
            _THCPrepare(
                (((1.0,),), (1.0,), 1),
                thc_wires["index_wires"],
                thc_wires["prep_garbage_wires"],
                thc_wires["work_wires"],
            ),
            {},
        ),
        (_SelectTHCBasis((1, (), ()), [0, 1], [2, 3], [4, 5], []), {}),
        (_THCIndexSwaps([0], [1], [2, 3, 4], []), {}),
    ]


_INSTANCES_TO_TEST = [
    # GlobalPhase acts on no wires, so `_check_differentiation`'s `qp.probs(wires=op.wires)` cannot be constructed
    (qp.GlobalPhase(1.1), {"skip_differentiation": True}),
    (LabelledOp(qp.X(0), "my-x"), {}),
    # MidMeasure and PauliMeasure are only operators in the tape-based pipeline.
    (qp.ops.MidMeasure(wires=0), {"skip_capture": True}),
    (qp.ops.PauliMeasure("X", wires=0), {"skip_capture": True}),
    # ChangeOpBasis as an operator is only used in the tape-based pipeline. It is unrolled when capture is enabled
    (ChangeOpBasis(qp.T(0), qp.PauliZ(0)), {"skip_capture": True}),
    (qp.sum(qp.PauliX(0), qp.PauliZ(0)), {}),
    (qp.sum(qp.X(0), qp.X(0), qp.Z(0), qp.Z(0)), {}),
    (qp.BasisState([1], wires=[0]), {"skip_differentiation": True}),
    (qp.ControlledQubitUnitary(np.eye(2), wires=[1, 0]), {"skip_differentiation": True}),
    (
        qp.ControlledQubitUnitary(np.eye(4), wires=[1, 2, 0], control_values=[0]),
        {"skip_differentiation": True},
    ),
    (
        qp.QubitChannel([np.array([[1, 0], [0, 0.8]]), np.array([[0, 0.6], [0, 0]])], wires=0),
        {"skip_differentiation": True, "skip_capture": True},
    ),
    # Skipping bind_new_parameters test for `MultiControlledX` because it does not make sense for control values
    # Boolean control values are not differentiable.
    (
        qp.MultiControlledX(wires=[0, 1]),
        {"skip_bind_new_parameters": True, "skip_differentiation": True},
    ),
    (qp.Projector([1], 0), {"skip_differentiation": True}),
    (qp.Projector([1, 0], 0), {"skip_differentiation": True}),
    (qp.DiagonalQubitUnitary([1, 1, 1, 1], wires=[0, 1]), {"skip_differentiation": True}),
    (qp.QubitUnitary(np.eye(2), wires=[0]), {"skip_differentiation": True}),
    (qp.QubitUnitary(np.eye(4), wires=[0, 1]), {"skip_differentiation": True}),
    (
        qp.QubitUnitary(qp.Rot.compute_matrix(0.1, 0.2, 0.3), wires=[0]),
        {"skip_differentiation": True},
    ),
    (qp.SpecialUnitary([1, 1, 1], 0), {"skip_differentiation": True}),
    (qp.IntegerComparator(1, wires=[0, 1]), {"skip_differentiation": True}),
    (qp.PauliRot(1.1, "X", wires=[0]), {}),
    (qp.PPR_2(1, "XYZ", wires=[0, 1, 2]), {}),
    (qp.PPR_4(-1, "XYZ", wires=[0, 1, 2]), {}),
    (qp.PPR_8(1, "XYZ", wires=[0, 1, 2]), {}),
    (qp.StatePrep([0, 1], 0), {"skip_differentiation": True}),
    (qp.PCPhase(0.27, dim=2, wires=[0, 1]), {}),
    (
        qp.BlockEncode(jax.numpy.array([[0.1, 0.2], [0.3, 0.4]]), wires=[0, 1]),
        {"skip_differentiation": True},
    ),
    (qp.adjoint(qp.PauliX(0)), {}),
    (qp.adjoint(qp.RX(1.1, 0)), {}),
    (
        qp.ops.LinearCombination([1.1, 2.2], [qp.PauliX(0), qp.PauliZ(0)]),
        {"skip_differentiation": True},
    ),
    (qp.s_prod(1.1, qp.RX(1.1, 0)), {"skip_differentiation": True}),
    (qp.prod(qp.PauliX(0), qp.PauliY(1), qp.PauliZ(0)), {}),
    # NOTE: qp.prod dispatches to Prod2 now, keep legacy Prod for testing
    (qp.ops.Prod(qp.PauliX(0), qp.PauliY(1), qp.PauliZ(0)), {}),
    (qp.ctrl(qp.RX(1.1, 0), 1), {}),
    pytest.param(
        (qp.exp(qp.PauliX(0), 1.1), {}),
        marks=pytest.mark.xfail(
            reason=(
                "Real exponent decomposes to complex PauliRot, which is not "
                "supported with Operator"
            )
        ),
    ),
    (qp.pow(qp.IsingXX(1.1, [0, 1]), 2.5), {}),
    (qp.ops.Evolution(qp.PauliX(0), 5.2), {}),
    (qp.estimator.FirstQuantization(1, 2, 1), {"skip_bind_new_parameters": True}),
    (qp.prod(qp.RX(1.1, 0), qp.RY(2.2, 0), qp.RZ(3.3, 1)), {}),
    (qp.Snapshot(measurement=qp.expval(qp.Z(0)), tag="hi"), {}),
    (qp.Snapshot(tag="tag"), {}),
    (qp.Identity(0), {}),
    (qp.MultiRZ(0.123, wires=[0, 1, 2]), {}),
    (qp.MultiRZ(0.123, wires=[0]), {}),
    (qp.Hermitian(np.eye(2), wires=[0]), {"skip_differentiation": True}),
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
        qp.SelectPauliRot(
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
    *[
        pytest.param(instance, id=type(instance[0]).__name__)
        for instance in _subcircuit_instances()
    ],
]
"""Valid operator instances that could not be auto-generated."""


_INSTANCES_TO_FAIL = [
    (
        qp.SparseHamiltonian(qp.Hamiltonian([1.1], [qp.PauliX(0)]).sparse_matrix(), [0]),
        AssertionError,  # each data element must be tensorlike
    ),
    (
        qp.PauliError("X", 0.5, wires=0),
        DeviceError,  # not supported with default.qubit and does not provide a decomposition
    ),
    (
        qp.ops.qubit.special_unitary.TmpPauliRot(1.1, "X", [0]),
        AssertionError,  # private type with has_matrix=False despite having one
    ),
    (
        qp.ops.Conditional(qp.measure(1), qp.S(0)),
        AssertionError,  # needs flattening helpers to be updated, also cannot be pickled
    ),
    (
        qp.estimator.DoubleFactorization(np.eye(2), np.arange(16).reshape((2,) * 4)),
        TypeError,  # op.eigvals is a list (overwritten in the init)
    ),
]
"""
List[Tuple[LegacyOperator, Type[Exception]]]: List of tuples containing LegacyOperator instances that could
not be auto-generated, along with the exception type raised when trying to assert its validity.

These operators need to break PL conventions, and each one's reason is specified in a comment.
"""


_ABSTRACT_OR_META_TYPES = {
    LabelledOp,
    Adjoint,
    Adjoint2,
    AdjointOperation,
    LegacyOperator,
    Operator,
    Operation,
    Channel,
    qp.ops.Projector,
    qp.ops.SymbolicOp,
    SymbolicOp2,
    qp.ops.ScalarSymbolicOp,
    qp.ops.Pow,
    Pow2,
    qp.ops.CompositeOp,
    qp.ops.CompositeOp2,
    qp.ops.Controlled,
    qp.ops.ControlledOp,
    qp.ops.ChangeOpBasis,
    qp.ops.ChangeOpBasis2,
    qp.ops.Prod,
    qp.ops.Prod2,
    Controlled2,
    ControlledOp2,
    qp.ops.qubit.BasisStateProjector,
    qp.ops.qubit.StateVectorProjector,
    qp.templates.core.CollectedSubroutine,
    # pylint: disable-next=protected-access
    qp.templates.subroutines.arithmetic.out_square._SquareArithmeticOp,
    qp.ops.qubit.non_parametric_ops._FixedAnglePPR,  # pylint: disable=protected-access
    StatePrepBase,
    StatePrepBase2,
    PowOperation,
    qp.StatePrep,
    qp.allocation.Allocate,  # no integer wires
    qp.allocation.Deallocate,  # no integer wires
}
"""Types that should not have actual instances created."""


def get_all_classes(c):
    """Recursive function to generate a flat list of all child classes of ``c``.
    (first called with ``LegacyOperator``)."""
    if c.__module__[:10] != "pennylane.":
        return []
    subs = c.__subclasses__()
    classes = [] if c in _ABSTRACT_OR_META_TYPES else [c]
    for sub in subs:
        classes.extend(get_all_classes(sub))
    return classes


_CLASSES_TO_TEST = (
    (set(get_all_classes(LegacyOperator)) | set(get_all_classes(Operator)))
    - {i[1] for i in getmembers(qp.templates) if isclass(i[1]) and issubclass(i[1], LegacyOperator)}
    # `pytest.param` returns a `ParameterSet` (a namedtuple)
    - {type(v.values[0][0] if hasattr(v, "values") else v[0]) for v in _INSTANCES_TO_TEST}
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
