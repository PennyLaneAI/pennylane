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

"""Unit tests for xDSL utilities."""

import pytest

pytestmark = [pytest.mark.external, pytest.mark.capture]
xdsl = pytest.importorskip("xdsl")
jax = pytest.importorskip("jax")

# pylint: disable=wrong-import-position
from jaxlib.mlir.ir import Module as jaxModule  # pylint: disable=no-name-in-module
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, tensor, test

import pennylane as qml
from pennylane.compiler.python_compiler import QuantumParser
from pennylane.compiler.python_compiler.conversion import (
    generic_str,
    inline_jit_to_module,
    inline_module,
    mlir_from_docstring,
    mlir_module,
    parse_generic_to_mlir_module,
    parse_generic_to_xdsl_module,
    xdsl_from_docstring,
    xdsl_from_qjit,
    xdsl_module,
)
from pennylane.compiler.python_compiler.dialects import stablehlo
from pennylane.compiler.python_compiler.utils import get_constant_from_ssa


class TestGetConstantFromSSA:
    """Unit tests for ``get_constant_from_ssa``."""

    def test_non_constant(self):
        """Test that ``None`` is returned if the input is not a constant."""
        val = test.TestOp(result_types=(builtin.Float64Type(),)).results[0]
        assert get_constant_from_ssa(val) is None

    @pytest.mark.parametrize(
        "const, attr_type, dtype",
        [
            (11, builtin.IntegerAttr, builtin.IntegerType(64)),
            (5, builtin.IntegerAttr, builtin.IndexType()),
            (2.5, builtin.FloatAttr, builtin.Float64Type()),
        ],
    )
    def test_scalar_constant_arith(self, const, attr_type, dtype):
        """Test that constants created by ``arith.constant`` are returned correctly."""
        const_attr = attr_type(const, dtype)
        val = arith.ConstantOp(value=const_attr).results[0]

        assert get_constant_from_ssa(val) == const

    @pytest.mark.parametrize(
        "const, elt_type",
        [
            (11, builtin.IntegerType(64)),
            (9, builtin.IndexType()),
            (2.5, builtin.Float64Type()),
            (-1.1 + 2.3j, builtin.ComplexType(builtin.Float64Type())),
        ],
    )
    @pytest.mark.parametrize("constant_op", [arith.ConstantOp, stablehlo.ConstantOp])
    def test_scalar_constant_extracted_from_rank0_tensor(self, const, elt_type, constant_op):
        """Test that constants created by ``stablehlo.constant`` are returned correctly."""
        data = const
        if isinstance(const, complex):
            # For complex numbers, the number must be split into a 2-tuple containing
            # the real and imaginary part when initializing a dense elements attr.
            data = (const.real, const.imag)

        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=elt_type, shape=()),
            data=(data,),
        )
        tensor_ = constant_op(value=dense_attr).results[0]
        val = tensor.ExtractOp(tensor=tensor_, indices=[], result_type=elt_type).results[0]

        assert get_constant_from_ssa(val) == const

    def test_tensor_constant_arith(self):
        """Test that ``None`` is returned if the input is a tensor created by ``arith.constant``."""
        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1, 2, 3),
        )
        val = arith.ConstantOp(value=dense_attr).results[0]

        assert get_constant_from_ssa(val) is None

    def test_tensor_constant_stablehlo(self):
        """Test that ``None`` is returned if the input is a tensor created by ``stablehlo.constant``."""
        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1.0, 2.0, 3.0),
        )
        val = stablehlo.ConstantOp(value=dense_attr).results[0]

        assert get_constant_from_ssa(val) is None

    def test_extract_scalar_from_constant_tensor_stablehlo(self):
        """Test that ``None`` is returned if the input is a scalar constant, but it was extracted
        from a non-scalar constant."""
        # Index SSA value to be used for extracting a value from a tensor
        dummy_index = test.TestOp(result_types=(builtin.IndexType(),)).results[0]

        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1.0, 2.0, 3.0),
        )
        tensor_ = stablehlo.ConstantOp(value=dense_attr).results[0]
        val = tensor.ExtractOp(
            tensor=tensor_, indices=[dummy_index], result_type=builtin.Float64Type()
        ).results[0]
        # val is a value that we got by indexing into a constant tensor with rank >= 1
        assert isinstance(val.type, builtin.Float64Type)

        assert get_constant_from_ssa(val) is None


class TestConversionUtils:
    """Unit tests for utilities for converting Python code to xDSL modules."""

    def test_generic_str(self):
        """Test that the generic_str function works correctly."""

        @jax.jit
        def f(x):
            return x + 1

        gen_str = generic_str(f)(1)
        context = Context()
        module = QuantumParser(context, gen_str).parse_module()

        assert len(module.regions[0].blocks[0].ops) == 1
        func_op = module.regions[0].blocks[0].first_op
        assert isinstance(func_op, func.FuncOp)

        expected_op_names = ["stablehlo.constant", "stablehlo.add", "func.return"]
        for op, expected_op_name in zip(func_op.body.ops, expected_op_names):
            assert op.name == expected_op_name

    def test_mlir_module(self):
        """Test that the mlir_module function works correctly."""

        @jax.jit
        def f(x):
            return x + 1

        mod = mlir_module(f)(1)
        assert isinstance(mod, jaxModule)

    def test_xdsl_module(self):
        """Test that the xdsl_module function works correctly."""

        @jax.jit
        def f(x):
            return x + 1

        mod = xdsl_module(f)(1)
        assert isinstance(mod, builtin.ModuleOp)

        assert len(mod.regions[0].blocks[0].ops) == 1
        func_op = mod.regions[0].blocks[0].first_op
        assert isinstance(func_op, func.FuncOp)

        expected_op_names = ["stablehlo.constant", "stablehlo.add", "func.return"]
        for op, expected_op_name in zip(func_op.body.ops, expected_op_names):
            assert op.name == expected_op_name

    def test_parse_generic_to_mlir_module(self):
        """Test that the parse_generic_to_mlir_module function works correctly."""
        program_str = """
            "builtin.module"() ({
                %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
            }) : () -> ()
        """

        mod = parse_generic_to_mlir_module(program_str)
        assert isinstance(mod, jaxModule)

    def test_parse_generic_to_xdsl_module(self):
        """Test that the parse_generic_to_xdsl_module function works correctly."""
        program_str = """
            "builtin.module"() ({
                %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
            }) : () -> ()
        """

        mod = parse_generic_to_xdsl_module(program_str)
        assert isinstance(mod, builtin.ModuleOp)

        assert len(mod.regions[0].blocks[0].ops) == 1
        assert isinstance(mod.regions[0].blocks[0].first_op, arith.ConstantOp)

    def test_mlir_from_docstring(self):
        """Test that the mlir_from_docstring function works correctly."""

        def f():
            """
            %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
            """

        mod = mlir_from_docstring(f)
        assert isinstance(mod, jaxModule)

    def test_xdsl_from_docstring(self):
        """Test that the xdsl_from_docstring function works correctly."""

        def f():
            """
            %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
            """

        mod = xdsl_from_docstring(f)
        assert isinstance(mod, builtin.ModuleOp)

        assert len(mod.regions[0].blocks[0].ops) == 1
        assert isinstance(mod.regions[0].blocks[0].first_op, arith.ConstantOp)

    def test_xdsl_from_qjit(self):
        """Test that the xdsl_from_qjit function works correctly."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            return qml.state()

        mod = xdsl_from_qjit(circuit)()
        assert isinstance(mod, builtin.ModuleOp)

        nested_modules = []
        for op in mod.body.walk():
            if isinstance(op, builtin.ModuleOp):
                nested_modules.append(op)

        funcs = []
        assert len(nested_modules) == 1
        for op in nested_modules[0].body.walk():
            if isinstance(op, func.FuncOp):
                funcs.append(op)

        assert len(funcs) == 1
        # All qnodes have a UnitAttr attribute called qnode
        assert funcs[0].attributes.get("qnode", None) is not None


class TestInliningUtils:
    """Unit tests for utilities for inlining operations into xDSL modules."""

    def test_inline_module(self):
        """Test that the inline_module function works correctly."""

    def test_inline_jit_to_module(self):
        """Test that the inline_jit_to_module function works correctly."""


class TestXDSLToPLUtils:
    """Unit tests for utilities that convert xDSL constructs to PennyLane gates and measurements."""

    @pytest.mark.parametrize("", [])
    def test_xdsl_to_qml_op_static_wires(self):
        """Test that the xdsl_to_qml_op function works correctly."""

    @pytest.mark.parametrize("", [])
    def test_xdsl_to_qml_op_dynamic_wires(self):
        """Test that the xdsl_to_qml_op function works correctly."""

    def test_xdsl_to_qml_op_invalid_op(self):
        """Test that the xdsl_to_qml_op raises an error if given an invalid operation as input."""

    @pytest.mark.parametrize("", [])
    def test_xdsl_to_qml_measurement_static_wires(self):
        """Test that the xdsl_to_qml_measurement function works correctly with qubits corresponding to
        static wires."""

    @pytest.mark.parametrize("", [])
    def test_xdsl_to_qml_measurement_dynamic_wires(self):
        """Test that the xdsl_to_qml_measurement function works correctly."""

    def test_xdsl_to_qml_measurement_invalid_op(self):
        """Test that the xdsl_to_qml_measurement raises an error if given an invalid operation as input."""


if __name__ == "__main__":
    pytest.main(["-x", __file__])
