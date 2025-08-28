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

"""Unit test module for pennylane/compiler/python_compiler/dialects/stablehlo.py."""

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external


def test_all_unary_operations(run_filecheck):
    """Test all unary elementwise operations."""
    program = r"""
    // CHECK: %[[tf32:.*]] = "test.op"() : () -> tensor<f32>
    %tf32 = "test.op"() : () -> tensor<f32>
    
    // CHECK: %[[tf64:.*]] = "test.op"() : () -> tensor<f64>
    %tf64 = "test.op"() : () -> tensor<f64>
    
    // CHECK: %[[tcomplex:.*]] = "test.op"() : () -> tensor<complex<f32>>
    %tcomplex = "test.op"() : () -> tensor<complex<f32>>
    
    // CHECK: %convert = "stablehlo.convert"(%[[tf32]]) : (tensor<f32>) -> tensor<f64>
    %convert = "stablehlo.convert"(%tf32) : (tensor<f32>) -> tensor<f64>
    
    // CHECK: %cos = "stablehlo.cosine"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %cos = "stablehlo.cosine"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %exp = "stablehlo.exponential"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %exp = "stablehlo.exponential"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %exponential_minus_one = "stablehlo.exponential_minus_one"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %exponential_minus_one = "stablehlo.exponential_minus_one"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %floor = "stablehlo.floor"(%[[tf64]]) : (tensor<f64>) -> tensor<f64>
    %floor = "stablehlo.floor"(%tf64) : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %imag = "stablehlo.imag"(%[[tcomplex]]) : (tensor<complex<f32>>) -> tensor<f32>
    %imag = "stablehlo.imag"(%tcomplex) : (tensor<complex<f32>>) -> tensor<f32>
    
    // CHECK: %is_finite = "stablehlo.is_finite"(%[[tf32]]) : (tensor<f32>) -> tensor<i1>
    %is_finite = "stablehlo.is_finite"(%tf32) : (tensor<f32>) -> tensor<i1>
    
    // CHECK: %log = "stablehlo.log"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %log = "stablehlo.log"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %log_plus_one = "stablehlo.log_plus_one"(%[[tf64]]) : (tensor<f64>) -> tensor<f64>
    %log_plus_one = "stablehlo.log_plus_one"(%tf64) : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %logistic = "stablehlo.logistic"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %logistic = "stablehlo.logistic"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %negate = "stablehlo.negate"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %negate = "stablehlo.negate"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %real = "stablehlo.real"(%[[tcomplex]]) : (tensor<complex<f32>>) -> tensor<f32>
    %real = "stablehlo.real"(%tcomplex) : (tensor<complex<f32>>) -> tensor<f32>
    
    // CHECK: %round_afz = "stablehlo.round_nearest_afz"(%[[tf64]]) : (tensor<f64>) -> tensor<f64>
    %round_afz = "stablehlo.round_nearest_afz"(%tf64) : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %round_even = "stablehlo.round_nearest_even"(%[[tf64]]) : (tensor<f64>) -> tensor<f64>
    %round_even = "stablehlo.round_nearest_even"(%tf64) : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %rsqrt = "stablehlo.rsqrt"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %rsqrt = "stablehlo.rsqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %sign = "stablehlo.sign"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %sign = "stablehlo.sign"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %sin = "stablehlo.sine"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %sin = "stablehlo.sine"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %sqrt = "stablehlo.sqrt"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %sqrt = "stablehlo.sqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
    
    // CHECK: %tan = "stablehlo.tan"(%[[tf64]]) : (tensor<f64>) -> tensor<f64>
    %tan = "stablehlo.tan"(%tf64) : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %tanh = "stablehlo.tanh"(%[[tf32]]) : (tensor<f32>) -> tensor<f32>
    %tanh = "stablehlo.tanh"(%tf32) : (tensor<f32>) -> tensor<f32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_all_binary_operations(run_filecheck):
    """Test all binary elementwise operations."""
    program = r"""
    // CHECK: %[[tf32_1:.*]] = "test.op"() : () -> tensor<f32>
    %tf32_1 = "test.op"() : () -> tensor<f32>
    
    // CHECK: %[[tf32_2:.*]] = "test.op"() : () -> tensor<f32>
    %tf32_2 = "test.op"() : () -> tensor<f32>
    
    // CHECK: %[[tf64_1:.*]] = "test.op"() : () -> tensor<f64>
    %tf64_1 = "test.op"() : () -> tensor<f64>
    
    // CHECK: %[[tf64_2:.*]] = "test.op"() : () -> tensor<f64>
    %tf64_2 = "test.op"() : () -> tensor<f64>

    // CHECK: %complex = "stablehlo.complex"(%[[tf32_1]], %[[tf32_2]]) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
    %complex = "stablehlo.complex"(%tf32_1, %tf32_2) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
    
    // CHECK: %divide = "stablehlo.divide"(%[[tf32_1]], %[[tf32_2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %divide = "stablehlo.divide"(%tf32_1, %tf32_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // CHECK: %maximum = "stablehlo.maximum"(%[[tf32_1]], %[[tf32_2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %maximum = "stablehlo.maximum"(%tf32_1, %tf32_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // CHECK: %minimum = "stablehlo.minimum"(%[[tf32_1]], %[[tf32_2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %minimum = "stablehlo.minimum"(%tf32_1, %tf32_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // CHECK: %power = "stablehlo.power"(%[[tf64_1]], %[[tf64_2]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %power = "stablehlo.power"(%tf64_1, %tf64_2) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    
    // CHECK: %remainder = "stablehlo.remainder"(%[[tf32_1]], %[[tf32_2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %remainder = "stablehlo.remainder"(%tf32_1, %tf32_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_all_other_operations(run_filecheck):
    """Test all other elementwise operations."""
    program = r"""
    // CHECK: %[[tf32:.*]] = "test.op"() : () -> tensor<f32>
    %tf32 = "test.op"() : () -> tensor<f32>

    // CHECK: %[[tf64:.*]] = "test.op"() : () -> tensor<f64>
    %tf64 = "test.op"() : () -> tensor<f64>

    // CHECK: %[[ti1:.*]] = "test.op"() : () -> tensor<i1>
    %ti1 = "test.op"() : () -> tensor<i1>

    // CHECK: %clamp = "stablehlo.clamp"(%[[tf32]], %[[tf32]], %[[tf32]]) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %clamp = "stablehlo.clamp"(%tf32, %tf32, %tf32) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // CHECK: %compare = stablehlo.compare EQ, %[[tf32]], %[[tf32]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %compare = "stablehlo.compare"(%tf32, %tf32) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    
    // CHECK: %map = "stablehlo.map"(%[[tf32]], %[[tf32]]) ({
    // CHECK:   ^[[bb0:.*]](%arg0 : tensor<f32>, %arg1 : tensor<f32>):
    // CHECK:     %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK:     "stablehlo.return"(%0) : (tensor<f32>) -> ()
    // CHECK: }) {dimensions = array<i64: 0, 1>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %map = "stablehlo.map"(%tf32, %tf32) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%0) : (tensor<f32>) -> ()
    }) {
      dimensions = array<i64: 0, 1>
    } : (tensor<f32>, tensor<f32>) -> tensor<f32>
    
    // CHECK: %reduce_precision = "stablehlo.reduce_precision"(%[[tf64]]) {exponent_bits = 5 : i32, mantissa_bits = 10 : i32} : (tensor<f64>) -> tensor<f64>
    %reduce_precision = "stablehlo.reduce_precision"(%tf64) {exponent_bits = 5 : i32, mantissa_bits = 10 : i32} : (tensor<f64>) -> tensor<f64>
    
    // CHECK: %select = "stablehlo.select"(%[[ti1]], %[[tf32]], %[[tf32]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %select = "stablehlo.select"(%ti1, %tf32, %tf32) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_ir_shape_mismatch(run_filecheck):
    """Test that operations with shape mismatches are properly rejected."""
    program = r"""
    %tf32_2x3 = "test.op"() : () -> tensor<2x3xf32>
    %tf64_3x2 = "test.op"() : () -> tensor<3x2xf64>
    
    // This should fail verification due to shape mismatch
    %convert = "stablehlo.convert"(%tf32_2x3) : (tensor<2x3xf32>) -> tensor<3x2xf64>
    """

    with pytest.raises(
        Exception, match="all non-scalar operands/results must have the same shape and base type"
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_ir_type_mismatch(run_filecheck):
    """Test that operations with type mismatches are properly rejected."""
    program = r"""
    %ti32 = "test.op"() : () -> tensor<2x3xi32>
    
    // This should fail verification due to type mismatch (cosine expects float/complex)
    %cos = "stablehlo.cosine"(%ti32) : (tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    with pytest.raises(Exception, match="operand at position 0 does not verify"):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_ir_missing_operands(run_filecheck):
    """Test that operations with missing operands are properly rejected."""
    program = r"""
    %result = "stablehlo.convert"() : () -> tensor<2x3xf64>
    """

    with pytest.raises(Exception, match="Expected 1 operand"):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_ir_trait_verification_failure(run_filecheck):
    """Test that operations that violate trait constraints are properly rejected."""
    program = r"""
    %tf32_2x3 = "test.op"() : () -> tensor<2x3xf32>
    %tf64_3x2 = "test.op"() : () -> tensor<3x2xf64>
    
    // This should fail verification due to shape mismatch between operands
    %complex = "stablehlo.complex"(%tf32_2x3, %tf64_3x2) : (tensor<2x3xf32>, tensor<3x2xf64>) -> tensor<2x3xcomplex<f32>>
    """

    with pytest.raises(Exception, match="requires the same shape"):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_ir_operand_result_shape_mismatch(run_filecheck):
    """Test that operations with operand vs result shape mismatches are properly rejected."""
    program = r"""
    %tf32_2x3 = "test.op"() : () -> tensor<2x3xf32>
    
    // This should fail verification due to shape mismatch between operand and result
    %convert = "stablehlo.convert"(%tf32_2x3) : (tensor<2x3xf32>) -> tensor<3x2xf64>
    """

    with pytest.raises(
        Exception, match="all non-scalar operands/results must have the same shape and base type"
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_control_flow_operations(run_filecheck):
    """Test the IfOp operation."""
    program = r"""
    // Test IfOp:
    
    // CHECK:   %[[pred:.*]] = "test.op"() : () -> tensor<i1>
    %pred = "test.op"() : () -> tensor<i1>
    
    // CHECK:   %[[result:.*]] = "stablehlo.if"(%[[pred]]) ({
    // CHECK:     "stablehlo.return"(%[[pred]]) : (tensor<i1>) -> ()
    // CHECK:   }, {
    // CHECK:     "stablehlo.return"(%[[pred]]) : (tensor<i1>) -> ()
    // CHECK:   }) : (tensor<i1>) -> tensor<i1>
    %result = "stablehlo.if"(%pred) ({
        "stablehlo.return"(%pred) : (tensor<i1>) -> ()
    }, {
        "stablehlo.return"(%pred) : (tensor<i1>) -> ()
    }) : (tensor<i1>) -> tensor<i1>
    
    // Test WhileOp:

    // CHECK:   %[[init_i:.*]] = "test.op"() : () -> tensor<i64>
    %init_i = "test.op"() : () -> tensor<i64>
    
    // CHECK:   %[[init_sum:.*]] = "test.op"() : () -> tensor<i64>
    %init_sum = "test.op"() : () -> tensor<i64>
    
    // CHECK:   %[[ten:.*]] = "test.op"() : () -> tensor<i64>
    %ten = "test.op"() : () -> tensor<i64>
    
    // CHECK:   %[[one:.*]] = "test.op"() : () -> tensor<i64>
    %one = "test.op"() : () -> tensor<i64>
    
    // CHECK:   %[[results:.*]], %[[results_1:.*]] = "stablehlo.while"(%[[init_i]], %[[init_sum]]) ({
    // CHECK:   ^{{.*}}(%[[arg0:.*]] : tensor<i64>, %[[arg1:.*]] : tensor<i64>):
    // CHECK:     %[[cond:.*]] = stablehlo.compare LT, %[[arg0]], %[[ten]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
    // CHECK:     "stablehlo.return"(%[[cond]]) : (tensor<i1>) -> ()
    // CHECK:   }, {
    // CHECK:   ^{{.*}}(%[[arg0_1:.*]] : tensor<i64>, %[[arg1_1:.*]] : tensor<i64>):
    // CHECK:     %[[new_sum:.*]] = "stablehlo.add"(%[[arg1_1]], %[[one]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // CHECK:     %[[new_i:.*]] = "stablehlo.add"(%[[arg0_1]], %[[one]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // CHECK:     "stablehlo.return"(%[[new_i]], %[[new_sum]]) : (tensor<i64>, tensor<i64>) -> ()
    // CHECK:   }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    %results:2 = "stablehlo.while"(%init_i, %init_sum) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %cond = "stablehlo.compare"(%arg0, %ten) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %new_sum = "stablehlo.add"(%arg1, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %new_i = "stablehlo.add"(%arg0, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%new_i, %new_sum) : (tensor<i64>, tensor<i64>) -> ()
    }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    
    // Test OptimizationBarrierOp:

    // CHECK:   %[[operand:.*]] = "test.op"() : () -> tensor<i1>
    %operand = "test.op"() : () -> tensor<i1>
    
    // CHECK:   %[[result2:.*]] = "stablehlo.optimization_barrier"(%[[operand]]) : (tensor<i1>) -> tensor<i1>
    %result2 = "stablehlo.optimization_barrier"(%operand) : (tensor<i1>) -> tensor<i1>
    """

    run_filecheck(program, roundtrip=True, verify=True)
