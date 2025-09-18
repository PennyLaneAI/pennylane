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
        ////////////////// **Allocation of register with dynamic number of wires** //////////////////
        // CHECK: [[NQUBITS:%.+]] = "test.op"() : () -> i64
        // CHECK: [[QREG_DYN:%.+]] = quantum.alloc([[NQUBITS]]) : !quantum.reg
        %nqubits = "test.op"() : () -> i64
        %qreg_dynamic = quantum.alloc(%nqubits) : !quantum.reg

        ////////////////// **Deallocation of dynamic register** //////////////////
        // CHECK: quantum.dealloc [[QREG_DYN]] : !quantum.reg
        quantum.dealloc %qreg_dynamic : !quantum.reg

        ////////////////// **Allocation of register with static number of wires** //////////////////
        // CHECK: [[QREG_STATIC:%.+]] = quantum.alloc(10) : !quantum.reg
        %qreg_static = quantum.alloc(10) : !quantum.reg

        ////////////////// **Deallocation of static register** //////////////////
        // CHECK: quantum.dealloc [[QREG_STATIC]] : !quantum.reg
        quantum.dealloc %qreg_static : !quantum.reg

        ////////////////// **Dynamic qubit allocation** //////////////////
        // CHECK: [[DYN_QUBIT:%.+]] = quantum.alloc_qb : !quantum.bit
        %dyn_qubit = quantum.alloc_qb : !quantum.bit

        ////////////////// **Dynamic qubit deallocation** //////////////////
        // CHECK: quantum.dealloc_qb [[DYN_QUBIT]] : !quantum.bit
        quantum.dealloc_qb %dyn_qubit : !quantum.bit

        //////////////////////////////////////////////////////
        ////////////////// Quantum register //////////////////
        //////////////////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        ////////////////// **Static qubit extraction** //////////////////
        // CHECK: [[STATIC_QUBIT:%.+]] = quantum.extract [[QREG]][[[STATIC_INDEX:0]]] : !quantum.reg -> !quantum.bit
        %static_qubit = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit

        ////////////////// **Dynamic qubit extraction** //////////////////
        // CHECK: [[DYN_INDEX:%.+]] = "test.op"() : () -> i64
        // CHECK: [[DYN_QUBIT1:%.+]] = quantum.extract [[QREG]][[[DYN_INDEX]]] : !quantum.reg -> !quantum.bit
        %dyn_index = "test.op"() : () -> i64
        %dyn_qubit1 = quantum.extract %qreg[%dyn_index] : !quantum.reg -> !quantum.bit

        ////////////////// **Static qubit insertion** //////////////////
        // CHECK: [[QREG1:%.+]] = quantum.insert [[QREG]][[[STATIC_INDEX]]], [[STATIC_QUBIT]] : !quantum.reg, !quantum.bit
        %qreg1 = quantum.insert %qreg[0], %static_qubit : !quantum.reg, !quantum.bit

        ////////////////// **Dynamic qubit insertion** //////////////////
        // CHECK: quantum.insert [[QREG1]][[[DYN_INDEX]]], [[DYN_QUBIT1]] : !quantum.reg, !quantum.bit
        %qreg2 = quantum.insert %qreg1[%dyn_index], %dyn_qubit1 : !quantum.reg, !quantum.bit
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_quantum_ops(self, run_filecheck):
        """Test that the assembly format for quantum non-terminal operations works correctly."""

        # Tests for CustomOp, GlobalPhaseOp, MeasureOp, MultiRZOp, QubitUnitaryOp
        program = """
        ////////////////////////////////////////////////////////////////////////
        ////////////////// Qubits, params, and control values //////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////// **Qubits** //////////////////
        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q2:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q3:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit
        %q2 = "test.op"() : () -> !quantum.bit
        %q3 = "test.op"() : () -> !quantum.bit

        ////////////////// **Params** //////////////////
        // CHECK: [[PARAM1:%.+]] = "test.op"() : () -> f64
        // CHECK: [[PARAM2:%.+]] = "test.op"() : () -> f64
        // CHECK: [[MAT_TENSOR:%.+]] = "test.op"() : () -> tensor<4x4xcomplex<f64>>
        // CHECK: [[MAT_MEMREF:%.+]] = "test.op"() : () -> memref<4x4xcomplex<f64>>
        %param1 = "test.op"() : () -> f64
        %param2 = "test.op"() : () -> f64
        %mat_tensor = "test.op"() : () -> tensor<4x4xcomplex<f64>>
        %mat_memref = "test.op"() : () -> memref<4x4xcomplex<f64>>

        ////////////////// **Control values** //////////////////
        // CHECK: [[TRUE_CST:%.+]] = "test.op"() : () -> i1
        // CHECK: [[FALSE_CST:%.+]] = "test.op"() : () -> i1
        %true_cst = "test.op"() : () -> i1
        %false_cst = "test.op"() : () -> i1

        ///////////////////////////////////////////////////////////////////////
        ///////////////////////// **Operation tests** /////////////////////////
        ///////////////////////////////////////////////////////////////////////

        ////////////////// **CustomOp tests** //////////////////
        // No params, no control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "Gate"() [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qc1, %qc2 = quantum.custom "Gate"() %q0, %q1 : !quantum.bit, !quantum.bit

        // Params, no control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "ParamGate"([[PARAM1]], [[PARAM2]]) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qc3, %qc4 = quantum.custom "ParamGate"(%param1, %param2) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "ControlledGate"() [[Q0]] ctrls([[Q1]]) ctrlvals([[TRUE_CST]]) : !quantum.bit ctrls !quantum.bit
        %qc5, %qc6 = quantum.custom "ControlledGate"() %q0 ctrls(%q1) ctrlvals(%true_cst) : !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}} = quantum.custom "AdjGate"() [[Q0]] adj : !quantum.bit
        %qc8 = quantum.custom "AdjGate"() %q0 adj : !quantum.bit

        ////////////////// **GlobalPhaseOp tests** //////////////////
        // No control wires
        // CHECK: quantum.gphase([[PARAM1]]) :
        quantum.gphase(%param1) :

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} = quantum.gphase([[PARAM1]]) ctrls([[Q0]], [[Q1]]) ctrlvals([[FALSE_CST]], [[TRUE_CST]]) : !quantum.bit, !quantum.bit
        %qg1, %qg2 = quantum.gphase(%param1) ctrls(%q0, %q1) ctrlvals(%false_cst, %true_cst) : !quantum.bit, !quantum.bit

        // Adjoint
        // CHECK: {{%.+}} = quantum.gphase([[PARAM1]]) {adjoint} ctrls([[Q0]]) ctrlvals([[TRUE_CST]]) : !quantum.bit
        %qg3 = quantum.gphase(%param1) {adjoint} ctrls(%q0) ctrlvals(%true_cst) : !quantum.bit

        ////////////////// **MultiRZOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qm1, %qm2 = quantum.multirz(%param1) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[TRUE_CST]]) : !quantum.bit, !quantum.bit
        %qm3, %qm4, %qm5 = quantum.multirz(%param1) %q0, %q1 ctrls(%q2) ctrlvals(%true_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qm6, %qm7 = quantum.multirz(%param1) %q0, %q1 adj : !quantum.bit, !quantum.bit

        ////////////////// **QubitUnitaryOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qb1, %qb2 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[FALSE_CST]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        %qb3, %qb4, %qb5 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls(%q2) ctrlvals(%false_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qb6, %qb7 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 adj : !quantum.bit, !quantum.bit

        // MemRef
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_MEMREF]] : memref<4x4xcomplex<f64>>) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qb8, %qb9 = quantum.unitary(%mat_memref : memref<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

        ////////////////// **MeasureOp tests** //////////////////
        // No postselection
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q0]] : i1, !quantum.bit
        %mres1, %mqubit1 = quantum.measure %q0 : i1, !quantum.bit

        // Postselection
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q1]] postselect 0 : i1, !quantum.bit
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q2]] postselect 1 : i1, !quantum.bit
        %mres2, %mqubit2 = quantum.measure %q1 postselect 0 : i1, !quantum.bit
        %mres3, %mqubit3 = quantum.measure %q2 postselect 1 : i1, !quantum.bit
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_state_prep(self, run_filecheck):
        """Test that the assembly format for state prep operations works correctly."""

        # Tests for SetBasisStateOp, SetStateOp
        program = """
        ////////////////////////////////////////////
        ////////////////// Qubits //////////////////
        ////////////////////////////////////////////
        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit

        ////////////////// **SetBasisStateOp tests** //////////////////
        // Basis state containers
        // CHECK: [[BASIS_TENSOR:%.+]] = "test.op"() : () -> tensor<2xi1>
        // CHECK: [[BASIS_MEMREF:%.+]] = "test.op"() : () -> memref<2xi1>
        %basis_tensor = "test.op"() : () -> tensor<2xi1>
        %basis_memref = "test.op"() : () -> memref<2xi1>

        // Basis state operations
        // CHECK: [[Q2:%.+]], [[Q3:%.+]] = quantum.set_basis_state([[BASIS_TENSOR]]) [[Q0]], [[Q1]] : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: [[Q4:%.+]], [[Q5:%.+]] = quantum.set_basis_state([[BASIS_MEMREF]]) [[Q2]], [[Q3]] : (memref<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q2, %q3 = quantum.set_basis_state(%basis_tensor) %q0, %q1 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q4, %q5 = quantum.set_basis_state(%basis_memref) %q2, %q3 : (memref<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)

        ////////////////// **SetStateOp tests** //////////////////
        // State vector containers
        // CHECK: [[STATE_TENSOR:%.+]] = "test.op"() : () -> tensor<4xcomplex<f64>>
        // CHECK: [[STATE_MEMREF:%.+]] = "test.op"() : () -> memref<4xcomplex<f64>>
        %state_tensor = "test.op"() : () -> tensor<4xcomplex<f64>>
        %state_memref = "test.op"() : () -> memref<4xcomplex<f64>>

        // State prep operations
        // CHECK: [[Q6:%.+]], [[Q7:%.+]] = quantum.set_state([[STATE_TENSOR]]) [[Q4]], [[Q5]] : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: quantum.set_state([[STATE_MEMREF]]) [[Q6]], [[Q7]] : (memref<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q6, %q7 = quantum.set_state(%state_tensor) %q4, %q5 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q8, %q9 = quantum.set_state(%state_memref) %q6, %q7 : (memref<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_observables(self, run_filecheck):
        """Test that the assembly format for observable operations works correctly."""

        # Tests for observables: ComputationalBasisOp, HamiltonianOp, HermitianOp,
        #                        NamedObsOp, TensorOp
        program = """
        //////////////////////////////////////////////////////
        //////////// Quantum register  and qubits ////////////
        //////////////////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q2:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q3:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q4:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit
        %q2 = "test.op"() : () -> !quantum.bit
        %q3 = "test.op"() : () -> !quantum.bit
        %q4 = "test.op"() : () -> !quantum.bit

        //////////////////////////////////////////////
        //////////// **Observable tests** ////////////
        //////////////////////////////////////////////

        //////////// **NamedObsOp** ////////////
        // CHECK: [[X_OBS:%.+]] = quantum.namedobs [[Q0]][PauliX] : !quantum.obs
        // CHECK: [[Y_OBS:%.+]] = quantum.namedobs [[Q1]][PauliY] : !quantum.obs
        // CHECK: [[Z_OBS:%.+]] = quantum.namedobs [[Q2]][PauliZ] : !quantum.obs
        // CHECK: [[H_OBS:%.+]] = quantum.namedobs [[Q3]][Hadamard] : !quantum.obs
        // CHECK: [[I_OBS:%.+]] = quantum.namedobs [[Q4]][Identity] : !quantum.obs
        %x_obs = quantum.namedobs %q0[PauliX] : !quantum.obs
        %y_obs = quantum.namedobs %q1[PauliY] : !quantum.obs
        %z_obs = quantum.namedobs %q2[PauliZ] : !quantum.obs
        %h_obs = quantum.namedobs %q3[Hadamard] : !quantum.obs
        %i_obs = quantum.namedobs %q4[Identity] : !quantum.obs

        //////////// **HermitianOp** ////////////
        // Create tensor/memref
        // CHECK: [[HERM_TENSOR:%.+]] = "test.op"() : () -> tensor<2x2xcomplex<f64>>
        // CHECK: [[HERM_MEMREF:%.+]] = "test.op"() : () -> memref<2x2xcomplex<f64>>
        %herm_tensor = "test.op"() : () -> tensor<2x2xcomplex<f64>>
        %herm_memref = "test.op"() : () -> memref<2x2xcomplex<f64>>

        // Create Hermitians
        // CHECK: [[HERM1:%.+]] = quantum.hermitian([[HERM_TENSOR]] : tensor<2x2xcomplex<f64>>) [[Q0]] : !quantum.obs
        // CHECK: [[HERM2:%.+]] = quantum.hermitian([[HERM_MEMREF]] : memref<2x2xcomplex<f64>>) [[Q1]] : !quantum.obs
        %herm1 = quantum.hermitian(%herm_tensor : tensor<2x2xcomplex<f64>>) %q0 : !quantum.obs
        %herm2 = quantum.hermitian(%herm_memref : memref<2x2xcomplex<f64>>) %q1 : !quantum.obs

        //////////// **TensorOp** ////////////
        // CHECK: [[TENSOR_OBS:%.+]] = quantum.tensor [[X_OBS]], [[HERM2]], [[I_OBS]] : !quantum.obs
        %tensor_obs = quantum.tensor %x_obs, %herm2, %i_obs : !quantum.obs

        //////////// **HamiltonianOp** ////////////
        // Create tensor/memref
        // CHECK: [[HAM_TENSOR:%.+]] = "test.op"() : () -> tensor<3xf64>
        // CHECK: [[HAM_MEMREF:%.+]] = "test.op"() : () -> memref<3xf64>
        %ham_tensor = "test.op"() : () -> tensor<3xf64>
        %ham_memref = "test.op"() : () -> memref<3xf64>

        // Create Hamiltonians
        // CHECK: {{%.+}} = quantum.hamiltonian([[HAM_TENSOR]] : tensor<3xf64>) [[TENSOR_OBS]], [[X_OBS]], [[HERM1]] : !quantum.obs
        // CHECK: {{%.+}} = quantum.hamiltonian([[HAM_MEMREF]] : memref<3xf64>) [[TENSOR_OBS]], [[X_OBS]], [[HERM1]] : !quantum.obs
        %ham1 = quantum.hamiltonian(%ham_tensor : tensor<3xf64>) %tensor_obs, %x_obs, %herm1 : !quantum.obs
        %ham2 = quantum.hamiltonian(%ham_memref : memref<3xf64>) %tensor_obs, %x_obs, %herm1 : !quantum.obs

        //////////// **ComputationalBasisOp** ////////////
        // CHECK: {{%.+}} = quantum.compbasis qubits [[Q0]], [[Q1]] : !quantum.obs
        // CHECK: {{%.+}} = quantum.compbasis qreg [[QREG]] : !quantum.obs
        %cb_01 = quantum.compbasis qubits %q0, %q1 : !quantum.obs
        %cb_all = quantum.compbasis qreg %qreg : !quantum.obs
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_measurements(self, run_filecheck):
        """Test that the assembly format for measurement operations works correctly."""

        # Tests for measurements: CountsOp, ExpvalOp, MeasureOp, ProbsOp, SampleOp,
        #                         StateOp, VarianceOp
        program = """
        ///////////////////////////////////////////////////
        //////////// Observables and constants ////////////
        ///////////////////////////////////////////////////
        // CHECK: [[OBS:%.+]] = "test.op"() : () -> !quantum.obs
        %obs = "test.op"() : () -> !quantum.obs

        // CHECK: [[DYN_WIRES:%.+]] = "test.op"() : () -> i64
        %dyn_wires = "test.op"() : () -> i64
        // CHECK: [[DYN_SHOTS:%.+]] = "test.op"() : () -> i64
        %dyn_shots = "test.op"() : () -> i64

        ///////////////////////////////////////////////
        //////////// **Measurement tests** ////////////
        ///////////////////////////////////////////////

        ///////////////////// **ExpvalOp** /////////////////////
        // CHECK: {{%.+}} = quantum.expval [[OBS]] : f64
        %expval = quantum.expval %obs : f64

        ///////////////////// **VarianceOp** /////////////////////
        // CHECK: {{%.+}} = quantum.var [[OBS]] : f64
        %var = quantum.var %obs : f64

        ///////////////////// **CountsOp** /////////////////////
        // Counts with static shape
        // CHECK: {{%.+}}, {{%.+}} = quantum.counts [[OBS]] : tensor<6xf64>, tensor<6xi64>
        %eigvals1, %counts1 = quantum.counts %obs : tensor<6xf64>, tensor<6xi64>

        // Counts with dynamic shape
        // CHECK: {{%.+}}, {{%.+}} = quantum.counts [[OBS]] shape [[DYN_WIRES]] : tensor<?xf64>, tensor<?xi64>
        %eigvals2, %counts2 = quantum.counts %obs shape %dyn_wires : tensor<?xf64>, tensor<?xi64>

        // Counts with no results (mutate memref in-place)
        // CHECK: [[EIGVALS_IN:%.+]] = "test.op"() : () -> memref<16xf64>
        // CHECK: [[COUNTS_IN:%.+]] = "test.op"() : () -> memref<16xi64>
        // CHECK: quantum.counts [[OBS]] in([[EIGVALS_IN]] : memref<16xf64>, [[COUNTS_IN]] : memref<16xi64>)
        %eigvals_in = "test.op"() : () -> memref<16xf64>
        %counts_in = "test.op"() : () -> memref<16xi64>
        quantum.counts %obs in(%eigvals_in : memref<16xf64>, %counts_in : memref<16xi64>)

        ///////////////////// **ProbsOp** /////////////////////
        // Probs with static shape
        // CHECK: {{%.+}} = quantum.probs [[OBS]] : tensor<8xf64>
        %probs1 = quantum.probs %obs : tensor<8xf64>

        // Probs with dynamic shape
        // CHECK: {{%.+}} = quantum.probs [[OBS]] shape [[DYN_WIRES]] : tensor<?xf64>
        %probs2 = quantum.probs %obs shape %dyn_wires : tensor<?xf64>

        // Probs with no results (mutate memref in-place)
        // CHECK: [[PROBS_IN:%.+]] = "test.op"() : () -> memref<16xf64>
        // CHECK: quantum.probs [[OBS]] in([[PROBS_IN]] : memref<16xf64>)
        %probs_in = "test.op"() : () -> memref<16xf64>
        quantum.probs %obs in(%probs_in : memref<16xf64>)

        ///////////////////// **StateOp** /////////////////////
        // State with static shape
        // CHECK: {{%.+}} = quantum.state [[OBS]] : tensor<8xcomplex<f64>>
        %state1 = quantum.state %obs : tensor<8xcomplex<f64>>

        // State with dynamic shape
        // CHECK: {{%.+}} = quantum.state [[OBS]] shape [[DYN_WIRES]] : tensor<?xcomplex<f64>>
        %state2 = quantum.state %obs shape %dyn_wires : tensor<?xcomplex<f64>>

        // State with no results (mutate memref in-place)
        // CHECK: [[STATE_IN:%.+]] = "test.op"() : () -> memref<16xcomplex<f64>>
        // CHECK: quantum.state [[OBS]] in([[STATE_IN]] : memref<16xcomplex<f64>>)
        %state_in = "test.op"() : () -> memref<16xcomplex<f64>>
        quantum.state %obs in(%state_in : memref<16xcomplex<f64>>)

        ///////////////////// **SampleOp** /////////////////////
        // Samples with static shape
        // CHECK: {{%.+}} = quantum.sample [[OBS]] : tensor<10x3xf64>
        %samples1 = quantum.sample %obs : tensor<10x3xf64>

        // Samples with dynamic wires
        // CHECK: {{%.+}} = quantum.sample [[OBS]] shape [[DYN_WIRES]] : tensor<10x?xf64>
        %samples2 = quantum.sample %obs shape %dyn_wires : tensor<10x?xf64>

        // Samples with dynamic shots
        // CHECK: {{%.+}} = quantum.sample [[OBS]] shape [[DYN_SHOTS]] : tensor<?x3xf64>
        %samples3 = quantum.sample %obs shape %dyn_shots : tensor<?x3xf64>

        // Samples with dynamic wires and shots
        // CHECK: {{%.+}} = quantum.sample [[OBS]] shape [[DYN_SHOTS]], [[DYN_WIRES]] : tensor<?x?xf64>
        %samples4 = quantum.sample %obs shape %dyn_shots, %dyn_wires : tensor<?x?xf64>

        // Samples with no results (mutate memref in-place)
        // CHECK: [[SAMPLES_IN:%.+]] = "test.op"() : () -> memref<7x4xf64>
        // CHECK: quantum.sample [[OBS]] in([[SAMPLES_IN]] : memref<7x4xf64>)
        %samples_in = "test.op"() : () -> memref<7x4xf64>
        quantum.sample %obs in(%samples_in : memref<7x4xf64>)
        """

        run_filecheck(program, roundtrip=True, verify=True)

    def test_miscellaneous_operations(self, run_filecheck):
        """Test that the assembly format for miscelleneous operations
        works correctly."""

        # Tests for AdjointOp, DeviceInitOp, DeviceReleaseOp, FinalizeOp, InitializeOp,
        # NumQubitsOp, YieldOp
        program = """
        //////////////////////////////////////////
        //////////// Quantum register ////////////
        //////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        //////////// **AdjointOp and YieldOp tests** ////////////
        // CHECK:      quantum.adjoint([[QREG]]) : !quantum.reg {
        // CHECK-NEXT: ^bb0([[ARG_QREG:%.+]] : !quantum.reg):
        // CHECK-NEXT:   quantum.yield [[ARG_QREG]] : !quantum.reg
        // CHECK-NEXT: }
        %qreg1 = quantum.adjoint(%qreg) : !quantum.reg {
        ^bb0(%arg_qreg: !quantum.reg):
          quantum.yield %arg_qreg : !quantum.reg
        }

        //////////// **DeviceInitOp tests** ////////////
        // Integer SSA value for shots
        // CHECK: [[SHOTS:%.+]] = "test.op"() : () -> i64
        %shots = "test.op"() : () -> i64

        // No auto qubit management
        // CHECK: quantum.device shots([[SHOTS]]) ["foo", "bar", "baz"]
        quantum.device shots(%shots) ["foo", "bar", "baz"]

        // Auto qubit management
        // CHECK: quantum.device shots([[SHOTS]]) ["foo", "bar", "baz"] {auto_qubit_management}
        quantum.device shots(%shots) ["foo", "bar", "baz"] {auto_qubit_management}

        //////////// **DeviceReleaseOp tests** ////////////
        // CHECK: quantum.device_release
        quantum.device_release

        //////////// **FinalizeOp tests** ////////////
        // CHECK: quantum.finalize
        quantum.finalize

        //////////// **InitializeOp tests** ////////////
        // CHECK: quantum.init
        quantum.init

        //////////// **NumQubitsOp tests** ////////////
        // CHECK: quantum.num_qubits : i64
        %nqubits = quantum.num_qubits : i64
        """

        run_filecheck(program, roundtrip=True, verify=True)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
