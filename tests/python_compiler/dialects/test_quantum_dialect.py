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

"""Unit test module for pennylane/compiler/python_compiler/dialects/quantum.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from xdsl.dialects.test import TestOp
from xdsl.ir import AttributeCovT, OpResult

from pennylane.compiler.python_compiler.dialects import Quantum

all_ops = list(Quantum.operations)
all_attrs = list(Quantum.attributes)

expected_ops_names = {
    "AdjointOp": "quantum.adjoint",
    "AllocOp": "quantum.alloc",
    "AllocQubitOp": "quantum.alloc_qb",
    "ComputationalBasisOp": "quantum.compbasis",
    "CountsOp": "quantum.counts",
    "CustomOp": "quantum.custom",
    "DeallocOp": "quantum.dealloc",
    "DeallocQubitOp": "quantum.dealloc_qb",
    "DeviceInitOp": "quantum.device",
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
    "NumQubitsOp": "quantum.num_qubits",
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
    assert Quantum.name == "quantum"


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


class TestAssemblyFormat:
    """Lit tests for assembly format of operations/attributes in the Quantum
    dialect."""

    def test_qubit_qreg_operations(self, run_filecheck):
        """Test that the assembly format for operations for allocation/deallocation of
        qubits/quantum registers works correctly."""

        # Tests for allocation/deallocation ops: AllocOp, DeallocOp, AllocQubitOp, DeallocQubitOp
        # Tests for extraction/insertion ops: ExtractOp, InsertOp
        program = """
        // **Allocation of register with dynamic number of wires**
        // CHECK: [[NQUBITS:%.+]] = "test.op"() : () -> i64
        // CHECK: [[QREG_DYN:%.+]] = quantum.alloc([[NQUBITS]]) : !quantum.reg
        %nqubits = "test.op"() : () -> i64
        %qreg_dynamic = quantum.alloc(%nqubits) : !quantum.reg

        // **Deallocation of dynamic register**
        // CHECK: quantum.dealloc [[QREG_DYN]] : !quantum.reg
        quantum.dealloc %qreg_dynamic : !quantum.reg

        // **Allocation of register with static number of wires**
        // CHECK: [[QREG_STATIC:%.+]] = quantum.alloc(10) : !quantum.reg
        %qreg_static = quantum.alloc(10) : !quantum.reg

        // **Deallocation of static register**
        // CHECK: quantum.dealloc [[QREG_STATIC]] : !quantum.reg
        quantum.dealloc %qreg_static : !quantum.reg

        // **Dynamic qubit allocation**
        // CHECK: [[DYN_QUBIT:%.+]] = quantum.alloc_qb : !quantum.bit
        %dyn_qubit = quantum.alloc_qb : !quantum.bit

        // **Dynamic qubit deallocation**
        // CHECK: quantum.dealloc_qb [[DYN_QUBIT]] : !quantum.bit
        quantum.dealloc_qb %dyn_qubit : !quantum.bit

        //////////////////////////////////////////////////////////////////////////
        //////////// Quantum register to use with the remaining tests ////////////
        //////////////////////////////////////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        // **Static qubit extraction**
        // CHECK: [[STATIC_QUBIT:%.+]] = quantum.extract [[QREG]][[[STATIC_INDEX:0]]] : !quantum.reg -> !quantum.bit
        %static_qubit = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit

        // **Dynamic qubit extraction**
        // CHECK: [[DYN_INDEX:%.+]] = "test.op"() : () -> i64
        // CHECK: [[DYN_QUBIT1:%.+]] = quantum.extract [[QREG]][[[DYN_INDEX]]] : !quantum.reg -> !quantum.bit
        %dyn_index = "test.op"() : () -> i64
        %dyn_qubit1 = quantum.extract %qreg[%dyn_index] : !quantum.reg -> !quantum.bit

        // **Static qubit insertion**
        // CHECK: [[QREG1:%.+]] = quantum.insert [[QREG]][[[STATIC_INDEX]]], [[STATIC_QUBIT]] : !quantum.reg, !quantum.bit
        %qreg1 = quantum.insert %qreg[0], %static_qubit : !quantum.reg, !quantum.bit

        // **Dynamic qubit insertion**
        // CHECK: quantum.insert [[QREG1]][[[DYN_INDEX]]], [[DYN_QUBIT1]] : !quantum.reg, !quantum.bit
        %qreg2 = quantum.insert %qreg1[%dyn_index], %dyn_qubit1 : !quantum.reg, !quantum.bit
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_gates(self, run_filecheck):
        """Test that the assembly format for operations for quantum gates works correctly."""

        # Tests for CustomOp, GlobalPhaseOp, MultiRZOp, QubitUnitaryOp
        # program = """
        # //////////////////////////////////////////////////////////////////////////
        # //////////// Quantum register to use with the remaining tests ////////////
        # //////////////////////////////////////////////////////////////////////////
        # // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        # %qreg = "test.op"() : () -> !quantum.reg

        # // **Adjoint operation and yield operation**
        # // CHECK: [[NEW_QREG:%.+]] = quantum.adjoint([[QREG]]) : !quantum.reg {
        # // CHECK: ^bb0([[ARG_QREG:%.+]]: !quantum.reg):
        #     // CHECK: quantum.yield [[ARG_QREG]] : !quantum.reg
        # // CHECK: }
        # %new_qreg = quantum.adjoint(%qreg) : !quantum.reg {
        # ^bb0(%arg0: !quantum.reg):
        #     quantum.yield %arg0 : !quantum.reg
        # }

        # // CHECK: [[COMPBASIS:%.+]] = quantum.compbasis qreg [[QREG]] : !quantum.obs
        # %compbasis = quantum.compbasis qreg %qreg : !quantum.obs

        # // CHECK: quantum.counts [[COMPBASIS]] : tensor<2xf64>, tensor<2xi64>
        # %eigvals, %counts = quantum.counts %compbasis : tensor<2xf64>, tensor<2xi64>

        # // CHECK: [[PARAM:%.+]] = "test.op"() : () -> f64
        # %cst = "test.op"() : () -> f64

        # // CHECK: [[QUBIT:%.+]] = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
        # %qubit = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit

        # // CHECK: [[WIRE_DYN:%.+]] = "test.op"() : () -> i64
        # %cst2 = "test.op"() : () -> i64

        # // CHECK: [[QUBIT_DYN:%.+]] = quantum.extract %qreg[%cst2] : !quantum.reg -> !quantum.bit
        # %qubit_dyn = quantum.extract %qreg[%cst2] : !quantum.reg -> !quantum.bit

        # // CHECK: quantum.insert [[QREG:%.+]][0], [[QUBIT]] : !quantum.reg, !quantum.bit
        # %qreg_insert_static = quantum.insert %qreg[0], %qubit : !quantum.reg, !quantum.bit

        # // CHECK: quantum.insert [[QREG:%.+]][[[WIRE_DYN]]], [[QUBIT]] : !quantum.reg, !quantum.bit
        # %qreg_insert_dyn = quantum.insert %qreg[%cst2], %qubit : !quantum.reg, !quantum.bit

        # // CHECK: [[QUBIT2:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]]
        # %out_qubit = quantum.custom "RX"(%cst) %qubit : !quantum.bit

        # // CHECK: [[QUBIT3:%.+]] = quantum.alloc_qb : !quantum.bit
        # %alloc_qubit = quantum.alloc_qb : !quantum.bit

        # // CHECK: [[NUM_QUBITS:%.+]] = quantum.num_qubits : i64
        # %num_qubits = quantum.num_qubits : i64

        # // CHECK: quantum.dealloc_qb [[QUBIT3]] : !quantum.bit
        # quantum.dealloc_qb %alloc_qubit : !quantum.bit

        # // CHECK: quantum.device["some-library.so", "pennylane-lightning", "kwargs"]
        # quantum.device["some-library.so", "pennylane-lightning", "kwargs"]

        # // CHECK: [[mres2:%.+]], [[out_qubit2:%.+]] = quantum.measure [[QUBIT]] postselect 0 : i1, !quantum.bit
        # %mres2, %out_qubit2 = quantum.measure %qubit postselect 0 : i1, !quantum.bit
        # """

        # run_filecheck(program, roundtrip=True, verify=True)

    def test_state_prep(self, run_filecheck):
        """Test that the assembly format for state prep operations works correctly."""

        # Tests for SetBasisStateOp, SetStateOp
        # program = """
        # """

        # run_filecheck(program, roundtrip=True, verify=True)

    def test_observable_measurements(self, run_filecheck):
        """Test that the assembly format for observable and measurement operations
        works correctly."""

        # Tests for observables: ComputationalBasisOp, HamiltonianOp, HermitianOp,
        #                        NamedObsOp, TensorOp
        # Tests for measurements: CountsOp, ExpvalOp, MeasureOp, ProbsOp, SampleOp,
        #                         StateOp, VarianceOp
        # program = """
        # """

        # run_filecheck(program, roundtrip=True, verify=True)

    def test_miscellaneous_operations(self, run_filecheck):
        """Test that the assembly format for miscelleneous operations
        works correctly."""

        # Tests for AdjointOp, DeviceInitOp, DeviceReleaseOp, FinalizeOp, InitializeOp,
        # NumQubitsOp, YieldOp
        # program = """
        # """

        # run_filecheck(program, roundtrip=True, verify=True)
