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


def test_data_movement_operations(run_filecheck):
    """Test all data movement operations."""
    program = r"""
    ////////////////// Setup test operations //////////////////
    // CHECK: %[[input1:.*]] = "test.op"() : () -> tensor<3x2xi64>
    %input1 = "test.op"() : () -> tensor<3x2xi64>
    
    // CHECK: %[[input2:.*]] = "test.op"() : () -> tensor<1x2xi64>
    %input2 = "test.op"() : () -> tensor<1x2xi64>
    
    // CHECK: %[[operand:.*]] = "test.op"() : () -> tensor<2x3x4x2xi32>
    %operand = "test.op"() : () -> tensor<2x3x4x2xi32>
    
    // CHECK: %[[start_indices:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
    %start_indices = "test.op"() : () -> tensor<2x2x3x2xi64>
    
    // CHECK: %[[reshape_input:.*]] = "test.op"() : () -> tensor<2xf32>
    %reshape_input = "test.op"() : () -> tensor<2xf32>
    
    // CHECK: %[[scatter_input:.*]] = "test.op"() : () -> tensor<2x3x4x2xi64>
    %scatter_input = "test.op"() : () -> tensor<2x3x4x2xi64>
    
    // CHECK: %[[scatter_indices:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
    %scatter_indices = "test.op"() : () -> tensor<2x2x3x2xi64>
    
    // CHECK: %[[scatter_updates:.*]] = "test.op"() : () -> tensor<2x2x3x2x2xi64>
    %scatter_updates = "test.op"() : () -> tensor<2x2x3x2x2xi64>
    
    // CHECK: %[[slice_input:.*]] = "test.op"() : () -> tensor<3x4xi64>
    %slice_input = "test.op"() : () -> tensor<3x4xi64>
    
    // CHECK: %[[broadcast_input:.*]] = "test.op"() : () -> tensor<1x3xi32>
    %broadcast_input = "test.op"() : () -> tensor<1x3xi32>
    
    ////////////////// Test ConcatenateOp //////////////////
    // CHECK: %concatenate = "stablehlo.concatenate"(%[[input1]], %[[input2]]) <{dimension = 0 : i64}> : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    %concatenate = "stablehlo.concatenate"(%input1, %input2) {dimension = 0 : i64} : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    
    ////////////////// Test GatherOp //////////////////
    // CHECK: %gather = "stablehlo.gather"(%[[operand]], %[[start_indices]]) 
    // CHECK-SAME: dimension_numbers = #stablehlo.gather<
    // CHECK-NEXT:   offset_dims = [3, 4],
    // CHECK-NEXT:   collapsed_slice_dims = [1],
    // CHECK-NEXT:   operand_batching_dims = [0],
    // CHECK-NEXT:   start_indices_batching_dims = [1],
    // CHECK-NEXT:   start_index_map = [2, 1],
    // CHECK-NEXT:   index_vector_dim = 3
    // CHECK-NEXT: slice_sizes = array<i64: 1, 1, 2, 2>, indices_are_sorted = false
    %gather = "stablehlo.gather"(%operand, %start_indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = array<i64: 1, 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
    
    ////////////////// Test ReshapeOp //////////////////
    // CHECK: %reshape = stablehlo.reshape %[[reshape_input]] : (tensor<2xf32>) -> tensor<1x2xf32>
    %reshape = "stablehlo.reshape"(%reshape_input) : (tensor<2xf32>) -> tensor<1x2xf32>
    
    ////////////////// Test ScatterOp //////////////////
    // CHECK: %scatter = "stablehlo.scatter"(%[[scatter_input]], %[[scatter_indices]], %[[scatter_updates]]) 
    // CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<
    // CHECK-NEXT:   update_window_dims = [3, 4],
    // CHECK-NEXT:   inserted_window_dims = [1],
    // CHECK-NEXT:   input_batching_dims = [0],
    // CHECK-NEXT:   scatter_indices_batching_dims = [1],
    // CHECK-NEXT:   scatter_dims_to_operand_dims = [2, 1],
    // CHECK-NEXT:   index_vector_dim = 3
    // CHECK-NEXT: indices_are_sorted = false, unique_indices = false
    // CHECK-NEXT: ^[[bb0:.*]](%arg0 : tensor<i64>, %arg1 : tensor<i64>):
    // CHECK-NEXT:   %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // CHECK-NEXT:   "stablehlo.return"(%0) : (tensor<i64>) -> ()
    %scatter = "stablehlo.scatter"(%scatter_input, %scatter_indices, %scatter_updates) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [3, 4],
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [1],
        scatter_dims_to_operand_dims = [2, 1],
        index_vector_dim = 3>,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
    
    ////////////////// Test SliceOp //////////////////
    // CHECK: %slice = "stablehlo.slice"(%[[slice_input]])
    // CHECK-SAME:   start_indices = array<i64: 1, 2>,
    // CHECK-SAME:   limit_indices = array<i64: 3, 4>,
    // CHECK-SAME:   strides = array<i64: 1, 1>
    // CHECK-SAME:  : (tensor<3x4xi64>) -> tensor<2x2xi64>
    %slice = "stablehlo.slice"(%slice_input) {
      start_indices = array<i64: 1, 2>,
      limit_indices = array<i64: 3, 4>,
      strides = array<i64: 1, 1>
    } : (tensor<3x4xi64>) -> tensor<2x2xi64>
    
    ////////////////// Test BroadcastInDimOp //////////////////
    // CHECK: %broadcast = stablehlo.broadcast_in_dim %[[broadcast_input]], dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    %broadcast = "stablehlo.broadcast_in_dim"(%broadcast_input) {broadcast_dimensions = array<i64: 2, 1>} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>

    ////////////////// Test DynamicSliceOp //////////////////
    // CHECK: %[[dyn_operand:.*]] = "test.op"() : () -> tensor<4x4xi32>
    %dyn_operand = "test.op"() : () -> tensor<4x4xi32>

    // CHECK: %[[start0:.*]] = "test.op"() : () -> tensor<i64>
    %start0 = "test.op"() : () -> tensor<i64>

    // CHECK: %[[start1:.*]] = "test.op"() : () -> tensor<i64>
    %start1 = "test.op"() : () -> tensor<i64>

    // CHECK: %dynamic_slice = "stablehlo.dynamic_slice"(%[[dyn_operand]], %[[start0]], %[[start1]])
    // CHECK-SAME:   slice_sizes = array<i64: 2, 3>
    // CHECK-SAME:  : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>
    %dynamic_slice = "stablehlo.dynamic_slice"(%dyn_operand, %start0, %start1) {
      slice_sizes = array<i64: 2, 3>
    } : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_slice_operations(run_filecheck):
    """Test invalid slice operations that should fail verification."""
    program_slice_mismatch = r"""
    // CHECK: %input = "test.op"() : () -> tensor<3x8xi64>
    %input = "test.op"() : () -> tensor<3x8xi64>

    // This should fail verification due to mismatched array sizes
    // CHECK: %slice = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 4>, limit_indices = array<i64: 3, 8, 10>, strides = array<i64: 1, 2>} : (tensor<3x8xi64>) -> tensor<2x2xi64>
    %slice = "stablehlo.slice"(%input) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8, 10>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    """

    with pytest.raises(
        Exception,
        match="all of \\{start_indices, limit_indices, strides\\} must have the same size: got sizes 2, 3, 2",
    ):
        run_filecheck(program_slice_mismatch, roundtrip=True, verify=True)


def test_invalid_slice_element_type_mismatch(run_filecheck):
    """Test that SliceOp rejects mismatched operand/result element types."""
    program = r"""
    %slice_input = "test.op"() : () -> tensor<3x4xi64>
    // CHECK: %slice_input = "test.op"() : () -> tensor<3x4xi64>
    // Mismatched element type: operand is i64, result is f32
    %slice = "stablehlo.slice"(%slice_input) {
      start_indices = array<i64: 1, 2>,
      limit_indices = array<i64: 3, 4>,
      strides = array<i64: 1, 1>
    } : (tensor<3x4xi64>) -> tensor<2x2xf32>
    """

    # Expect verification failure due to element type mismatch
    with pytest.raises(
        Exception, match="requires the same element type for all operands and results"
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_gather_element_type_mismatch(run_filecheck):
    """Test that GatherOp rejects mismatched operand/result element types."""
    program = r"""
    %operand = "test.op"() : () -> tensor<2x3x4x2xi32>
    %start_indices = "test.op"() : () -> tensor<2x2x3x2xi64>

    // Mismatched element type: operand is i32, result is f32
    %gather_bad = "stablehlo.gather"(%operand, %start_indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = array<i64: 1, 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xf32>
    """

    # Expect verification failure due to element type mismatch between operand and result
    with pytest.raises(
        Exception, match=r"all of \{operand, result\} must have the same element type"
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_reshape_operations(run_filecheck):
    """Test invalid reshape operations that should fail verification."""
    program_reshape_mismatch = r"""
    %reshape_input = "test.op"() : () -> tensor<2xf32>

    // This should fail verification due to element count mismatch (2 != 4)
    %reshape_bad = "stablehlo.reshape"(%reshape_input) : (tensor<2xf32>) -> tensor<2x2xf32>
    """

    with pytest.raises(Exception, match="number of output elements"):
        run_filecheck(program_reshape_mismatch, roundtrip=True, verify=True)


def test_invalid_broadcast_in_dim_operations(run_filecheck):
    """Test invalid broadcast_in_dim operations that should fail verification."""
    # Test dims size mismatch.
    program_broadcast_dims_size_mismatch = r"""
    %broadcast_input = "test.op"() : () -> tensor<1x3xi32>

    // dims has size 1, but operand rank is 2
    %broadcast_bad = "stablehlo.broadcast_in_dim"(%broadcast_input) {broadcast_dimensions = array<i64: 1>} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    """

    with pytest.raises(Exception, match="broadcast_dimensions size .* does not match operand rank"):
        run_filecheck(program_broadcast_dims_size_mismatch, roundtrip=True, verify=True)

    # Test duplicate dims.
    program_broadcast_duplicate_dims = r"""
    %broadcast_input = "test.op"() : () -> tensor<1x3xi32>

    // duplicate entries in broadcast_dimensions are not allowed
    %broadcast_bad = "stablehlo.broadcast_in_dim"(%broadcast_input) {broadcast_dimensions = array<i64: 1, 1>} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    """

    with pytest.raises(Exception, match="broadcast_dimensions should not have duplicates"):
        run_filecheck(program_broadcast_duplicate_dims, roundtrip=True, verify=True)

    # Test dim index out of bounds.
    program_broadcast_dim_oob = r"""
    %broadcast_input = "test.op"() : () -> tensor<1x3xi32>

    // result rank is 2, but dims contains 2 (out of bounds)
    %broadcast_bad = "stablehlo.broadcast_in_dim"(%broadcast_input) {broadcast_dimensions = array<i64: 2, 1>} : (tensor<1x3xi32>) -> tensor<2x3xi32>
    """

    with pytest.raises(Exception, match="broadcast_dimensions contains invalid value"):
        run_filecheck(program_broadcast_dim_oob, roundtrip=True, verify=True)

    # Test operand dim not 1 and not equal to result dim.
    program_broadcast_dim_mismatch = r"""
    %broadcast_input = "test.op"() : () -> tensor<2x3xi32>

    // operand[0] = 2, result[0] = 4; dims = [0, 2] -> mismatch on dim 0
    %broadcast_bad = "stablehlo.broadcast_in_dim"(%broadcast_input) {broadcast_dimensions = array<i64: 0, 2>} : (tensor<2x3xi32>) -> tensor<4x3x2xi32>
    """

    with pytest.raises(
        Exception,
        match="size of operand dimension .* is not equal to 1 or size of result dimension",
    ):
        run_filecheck(program_broadcast_dim_mismatch, roundtrip=True, verify=True)


def test_dynamism_operations(run_filecheck):
    """Test all dynamism operations."""
    program = r"""
    ////////////////// Setup //////////////////
    // CHECK: %[[operand:.*]] = "test.op"() : () -> tensor<1x3xi64>
    %operand = "test.op"() : () -> tensor<1x3xi64>

    // CHECK: %[[out_dims:.*]] = "test.op"() : () -> tensor<3xi64>
    %out_dims = "test.op"() : () -> tensor<3xi64>

    ////////////////// Test DynamicBroadcastInDimOp //////////////////
    // CHECK: %dynamic_bcast = stablehlo.dynamic_broadcast_in_dim %[[operand]], %[[out_dims]], dims = [2, 1] : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    %dynamic_bcast = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out_dims) {
      broadcast_dimensions = array<i64: 2, 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_reduction_operations(run_filecheck):
    """Test all reduction operations."""
    program = r"""
    ////////////////// Setup //////////////////
    // CHECK: %[[input:.*]] = "test.op"() : () -> tensor<1x6xi64>
    %input = "test.op"() : () -> tensor<1x6xi64>

    // CHECK: %[[init:.*]] = "test.op"() : () -> tensor<i64>
    %init = "test.op"() : () -> tensor<i64>

    ////////////////// Test ReduceOp //////////////////
    // CHECK: %reduce = "stablehlo.reduce"(%[[input]], %[[init]]) <{dimensions = array<i64: 1>}> ({
    // CHECK:   ^[[bb0:.*]](%arg0 : tensor<i64>, %arg1 : tensor<i64>):
    // CHECK:     %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // CHECK:     "stablehlo.return"(%0) : (tensor<i64>) -> ()
    // CHECK: }) : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_reduction_operations(run_filecheck):
    """Test invalid cases for ReduceOp verifier."""

    # Duplicate dimensions
    program_dup_dims = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<i64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {dimensions = array<i64: 1, 1>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"dimensions should not have duplicates"):
        run_filecheck(program_dup_dims, roundtrip=True, verify=True)

    # Dimension out of range
    program_dim_oob = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<i64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {dimensions = array<i64: 2>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"dimensions contains an invalid value"):
        run_filecheck(program_dim_oob, roundtrip=True, verify=True)

    # Input/init element type mismatch
    program_elem_mismatch = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<f64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x6xi64>, tensor<f64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"input and init_value must have the same element type"):
        run_filecheck(program_elem_mismatch, roundtrip=True, verify=True)

    # Reducer wrong arity (expects 2 args per input; give 1)
    program_wrong_arity = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<i64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%acc: tensor<i64>):
        "stablehlo.return"(%acc) : (tensor<i64>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"reducer must take 2 arguments, got 1"):
        run_filecheck(program_wrong_arity, roundtrip=True, verify=True)

    # Reducer arg wrong rank (should be 0D)
    program_arg_rank = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<i64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<2xi64>, %arg1: tensor<2xi64>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
        "stablehlo.return"(%0) : (tensor<2xi64>) -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"reducer arguments must be rank-0 tensors"):
        run_filecheck(program_arg_rank, roundtrip=True, verify=True)

    # Reducer return wrong count
    program_return_count = r"""
    %input = "test.op"() : () -> tensor<1x6xi64>
    %init = "test.op"() : () -> tensor<i64>

    %reduce = "stablehlo.reduce"(%input, %init) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        "stablehlo.return"() : () -> ()
    }) {dimensions = array<i64: 1>} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    """

    with pytest.raises(Exception, match=r"reducer must return exactly one value per input"):
        run_filecheck(program_return_count, roundtrip=True, verify=True)


def test_custom_call_basic(run_filecheck):
    """CustomCallOp minimal form without layouts should verify."""
    program = r"""
    // CHECK: %[[ARG:.*]] = "test.op"() : () -> tensor<2x3xi32>
    %arg = "test.op"() : () -> tensor<2x3xi32>

    // CHECK: %[[RES:.*]] = "stablehlo.custom_call"(%[[ARG]])
    // CHECK-SAME: call_target_name = "foo"
    // CHECK-SAME: api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>
    %res = "stablehlo.custom_call"(%arg) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      output_operand_aliases = []
    } : (tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_custom_call_with_layouts(run_filecheck):
    """CustomCallOp with matching operand/result layouts should verify."""
    program = r"""
    // CHECK: %[[ARG:.*]] = "test.op"() : () -> tensor<2x3xi32>
    %arg = "test.op"() : () -> tensor<2x3xi32>

    // CHECK: %[[RES:.*]] = "stablehlo.custom_call"(%[[ARG]])
    // CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>]
    // CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    %res = "stablehlo.custom_call"(%arg) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>],
      output_operand_aliases = []
    } : (tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    run_filecheck(program, roundtrip=True, verify=True)


def test_custom_call_missing_result_layouts(run_filecheck):
    """Providing only operand_layouts should fail (must provide both or none)."""
    program = r"""
    %arg = "test.op"() : () -> tensor<2x3xi32>

    %res = "stablehlo.custom_call"(%arg) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      output_operand_aliases = []
    } : (tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    with pytest.raises(
        Exception,
        match=r"either both operands and results or none",
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_custom_call_layouts_mismatch(run_filecheck):
    """Number of layouts must match number of operands/results."""
    program = r"""
    %arg0 = "test.op"() : () -> tensor<2x3xi32>
    %arg1 = "test.op"() : () -> tensor<2x3xi32>

    %res = "stablehlo.custom_call"(%arg0, %arg1) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>],
      output_operand_aliases = []
    } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    with pytest.raises(
        Exception, match=r"Number of operands must match the number of operand layouts"
    ):
        run_filecheck(program, roundtrip=True, verify=True)


def test_custom_call_incorrect_layout_perm(run_filecheck):
    """Layout must be a permutation of [0, rank)."""
    program = r"""
    %arg = "test.op"() : () -> tensor<2x3xi32>

    %res = "stablehlo.custom_call"(%arg) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      operand_layouts = [dense<[0]> : tensor<1xindex>],
      result_layouts = [dense<[0]> : tensor<1xindex>],
      output_operand_aliases = []
    } : (tensor<2x3xi32>) -> tensor<2x3xi32>
    """

    with pytest.raises(Exception, match=r"layout must be a permutation of \[0, 2\)"):
        run_filecheck(program, roundtrip=True, verify=True)


def test_custom_call_single_tuple_result_with_element_layouts(run_filecheck):
    """Single tuple result with element-wise layouts should verify (common case)."""
    program = r"""
    // CHECK: %[[ARG0:.*]] = "test.op"() : () -> tensor<2x3xi32>
    // CHECK: %[[ARG1:.*]] = "test.op"() : () -> tensor<1xi32>
    %arg0 = "test.op"() : () -> tensor<2x3xi32>
    %arg1 = "test.op"() : () -> tensor<1xi32>

    // CHECK: %[[RES:.*]] = "stablehlo.custom_call"(%[[ARG0]])
    // CHECK-SAME: call_target_name = "foo"
    // CHECK-SAME: api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>
    // CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>]
    // CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
    %res = "stablehlo.custom_call"(%arg0) {
      call_target_name = "foo",
      api_version = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>,
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
      output_operand_aliases = []
    } : (tensor<2x3xi32>) -> tuple<tensor<2x3xi32>, tensor<1xi32>>
    """
    run_filecheck(program, roundtrip=True, verify=True)


def test_invalid_dynamic_broadcast_in_dim_operations(run_filecheck):
    """Test invalid dynamic_broadcast_in_dim cases that should fail verification."""

    # dims size mismatch (c2)
    program_dims_size_mismatch = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<3xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    """

    with pytest.raises(
        Exception, match=r"broadcast_dimensions size \(1\) does not match operand rank \(2\)"
    ):
        run_filecheck(program_dims_size_mismatch, roundtrip=True, verify=True)

    # result rank < operand rank (c3)
    program_result_rank_too_small = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<1xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0, 0>
    } : (tensor<1x3xi64>, tensor<1xi64>) -> tensor<3xi64>
    """

    with pytest.raises(Exception, match=r"result rank \(1\) is less than operand rank \(2\)"):
        run_filecheck(program_result_rank_too_small, roundtrip=True, verify=True)

    # duplicate dims (c4)
    program_duplicate_dims = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<2xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0, 0>
    } : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    """

    with pytest.raises(Exception, match=r"broadcast_dimensions should not have duplicates"):
        run_filecheck(program_duplicate_dims, roundtrip=True, verify=True)

    # dim index out of bounds (c5 bounds)
    program_dim_oob = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<2xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0, 2>
    } : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    """

    with pytest.raises(
        Exception, match=r"broadcast_dimensions contains invalid value 2 for result with rank 2"
    ):
        run_filecheck(program_dim_oob, roundtrip=True, verify=True)

    # per-dimension size compatibility (c5 compatibility)
    program_dim_incompatible = r"""
    %operand = "test.op"() : () -> tensor<2x3xi32>
    %out = "test.op"() : () -> tensor<3xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0, 2>
    } : (tensor<2x3xi32>, tensor<3xi64>) -> tensor<4x3x2xi32>
    """

    with pytest.raises(
        Exception,
        match=r"size of operand dimension 0 \(2\) is not compatible with size of result dimension 0 \(4\)",
    ):
        run_filecheck(program_dim_incompatible, roundtrip=True, verify=True)

    # output_dimensions length incompatible with result rank when static (c7)
    program_outlen_mismatch = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<2xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 2, 1>
    } : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<2x3x2xi64>
    """

    with pytest.raises(
        Exception,
        match=r"length of output_dimensions \(2\) is not compatible with result rank \(3\)",
    ):
        run_filecheck(program_outlen_mismatch, roundtrip=True, verify=True)

    # duplicate expansion hints across both lists (c8)
    program_dup_hints = r"""
    %operand = "test.op"() : () -> tensor<1x1xi64>
    %out = "test.op"() : () -> tensor<2xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 1, 0>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 0>
    } : (tensor<1x1xi64>, tensor<2xi64>) -> tensor<2x1xi64>
    """

    with pytest.raises(
        Exception, match=r"duplicate expansion hint for at least one operand dimension"
    ):
        run_filecheck(program_dup_hints, roundtrip=True, verify=True)

    # hint refers to invalid operand dimension (c9/c10)
    program_hint_oob = r"""
    %operand = "test.op"() : () -> tensor<1x3xi64>
    %out = "test.op"() : () -> tensor<2xi64>

    %bad = "stablehlo.dynamic_broadcast_in_dim"(%operand, %out) {
      broadcast_dimensions = array<i64: 0, 1>,
      known_expanding_dimensions = array<i64: 5>
    } : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    """

    with pytest.raises(
        Exception,
        match=r"hint for expanding dimension 5 does not refer to a valid operand dimension",
    ):
        run_filecheck(program_hint_oob, roundtrip=True, verify=True)
