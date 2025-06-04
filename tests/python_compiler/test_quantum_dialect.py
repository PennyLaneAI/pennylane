# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit test module for pennylane/compiler/python_compiler/quantum_dialect.py."""

import io

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from xdsl.dialects.test import TestOp
from xdsl.ir import AttributeCovT, OpResult

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect

all_ops = list(QuantumDialect.operations)
all_attrs = list(QuantumDialect.attributes)

expected_ops_names = {
    "AdjointOp": "quantum.adjoint",
    "AllocOp": "quantum.alloc",
    "ComputationalBasisOp": "quantum.compbasis",
    "CountsOp": "quantum.counts",
    "CustomOp": "quantum.custom",
    "DeallocOp": "quantum.dealloc",
    "DeviceInitOp": "quantum.device_init",
    "DeviceReleaseOp": "quantum.device_release",
    "ExpvalOp": "quantum.expval",
    "ExtractOp": "quantum.extract",
    "FinalizeOp": "quantum.finalize",
    "GlobalPhaseOp": "quantum.gphase",
    "HamiltonianOp": "quantum.hamiltonian",
    "HermitianOp": "quantum.hermitian",
    "InitializeOp": "quantum.init",
    "InsertOp": "quantum.insert",
    "MeasureOp": "quantum.measure",
    "MultiRZOp": "quantum.multirz",
    "NamedObsOp": "quantum.namedobs",
    "ProbsOp": "quantum.probs",
    "QubitUnitaryOp": "quantum.unitary",
    "SampleOp": "quantum.sample",
    "SetBasisStateOp": "quantum.set_basis_state",
    "SetStateOp": "quantum.set_state",
    "StateOp": "quantum.state",
    "TensorOp": "quantum.tensor",
    "VarianceOp": "quantum.var",
    "YieldOp": "quantum.yield",
}

expected_attrs_names = {
    "ObservableType": "quantum.obs",
    "QubitType": "quantum.bit",
    "QuregType": "quantum.reg",
    "ResultType": "quantum.res",
    "NamedObservableAttr": "quantum.named_observable",
}


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = TestOp(result_types=(t,))
    return op.results[0]


def test_quantum_dialect_name():
    """Test that the QuantumDialect name is correct."""
    assert QuantumDialect.name == "quantum"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected operation {op_class_name} found in QuantumDialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected attribute {attr_class_name} found in QuantumDialect"
    assert attr.name == expected_name


def test_assembly_format():
    program = """
    // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
    %qreg = "test.op"() : () -> !quantum.reg
    """

    ctx = xdsl.context.Context()
    from xdsl.dialects import builtin, func, test

    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(test.Test)
    ctx.load_dialect(QuantumDialect)

    module = xdsl.parser.Parser(ctx, program).parse_module()
    print(module)

    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(module)),
        Parser(opts, io.StringIO(program), *pattern_for_opts(opts)),
    )

    assert matcher.run() == 0
