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

"""This module contains the implementation of the measurements_from_samples transform,
written using xDSL."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler import quantum_dialect as quantum
from pennylane.compiler.python_compiler.jax_utils import xdsl_module
from pennylane.compiler.python_compiler.transforms.utils import xdsl_transform


@xdsl_module
@jax.jit
def postprocessing_expval(samples, wire):
    return jnp.mean(1.0 - 2.0 * samples[:, wire])


@xdsl_module
@jax.jit
def postprocessing_var(samples, wire):
    # We have to compute the variance manually here, rather than using jnp.var.
    # The reason is that jnp.var does not get lowered (for some reason) and is left as a call to an
    # undefined `_var` function in the generated module.
    a = 1.0 - 2.0 * samples[:, wire]
    return jnp.sum((a - jnp.mean(a)) ** 2) / a.size


@xdsl_module
@jax.jit
def postprocessing_probs(samples, wire):
    n_samples = samples.size
    probs_0 = jnp.sum(samples == 0) / n_samples
    probs_1 = jnp.sum(samples == 1) / n_samples

    return jnp.array([probs_0, probs_1], dtype=float)


class MeasurementsFromSamplesPattern(pattern_rewriter.RewritePattern):
    # pylint: disable=too-few-public-methods
    """Rewrite pattern for the ``measurements_from_samples`` transform, which replaces all terminal
    measurements in a program with a single :func:`pennylane.sample` measurement, and adds
    postprocessing instructions to recover the original measurement.
    """

    @classmethod
    def _validate_observable_op(cls, op: quantum.NamedObsOp):
        """TODO"""
        assert isinstance(
            op, quantum.NamedObsOp
        ), f"Expected a quantum.NamedObsOp but got {type(op).__name__}"
        if not op.type.data == "PauliZ":
            raise NotImplementedError(
                f"Observable '{op.type.data}' used as input to measurement operation is not "
                f"supported for the measurements_from_samples transform; currently only the "
                f"PauliZ observable is permitted"
            )

    @classmethod
    def _get_static_shots_value_from_device_op(cls, op: quantum.DeviceInitOp):
        """TODO"""
        print("[DEBUG]: Getting number of shots from DeviceInitOp")

        assert isinstance(
            op, quantum.DeviceInitOp
        ), f"Expected a quantum.DeviceInitOp but got {type(op).__name__}"

        # The number of shots is passed as an SSA value operand to the DeviceInitOp
        shots_operand = op.shots
        shots_extract_op = shots_operand.owner
        assert isinstance(shots_extract_op, tensor.ExtractOp), (
            f"Expected owner of shots operand to be a tensor.ExtractOp but got "
            f"{type(shots_extract_op).__name__}"
        )

        # This should be a stablehlo.ConstantOp that stores the values of `shots`
        shots_constant_op = shots_extract_op.operands[0].owner
        shots_value_attribute: builtin.DenseIntOrFPElementsAttr = shots_constant_op.properties.get(
            "value"
        )
        assert (
            shots_value_attribute is not None
        ), f"Cannot extract number of shots, the constant op has no 'value' attribute"

        shots_int_values = shots_value_attribute.get_int_values()
        assert (
            len(shots_int_values) == 1
        ), f"Expected a single shots value, got {len(shots_int_values)}"

        return shots_int_values[0]

    @classmethod
    def _rewrite_expval_or_var(
        cls,
        op: quantum.ExpvalOp | quantum.VarianceOp,
        shots: int,
        func_op: func.FuncOp,
        rewriter: pattern_rewriter.PatternRewriter,
    ):
        """TODO"""
        assert isinstance(
            op, (quantum.ExpvalOp, quantum.VarianceOp)
        ), f"Expected `op` to be a quantum.ExpvalOp or a quantum.VarianceOp but got {type(op).__name__}"
        assert isinstance(
            shots, int
        ), f"Expected `shots` to be an integer value but got {type(shots).__name__}"
        assert isinstance(
            func_op, func.FuncOp
        ), f"Expected `func_op` to be a func.FuncOP but got {type(func_op).__name__}"

        observable_op = op.operands[0].owner
        cls._validate_observable_op(observable_op)

        in_qubit = observable_op.operands[0]

        # Steps [TODO: are these steps still up to date?]:
        #   1. Insert op quantum.compbasis with input = in_qubit, output is a quantum.obs
        #   2. Insert op quantum.sample with input = the quantum.obs vs output of compbasis
        #   3. Erase op quantum.expval
        #   4. Erase op quantum.namedobs [PauliZ] (assuming we only support PauliZ basis!)
        compbasis_op = quantum.ComputationalBasisOp(
            operands=[in_qubit, None], result_types=[quantum.ObservableType()]
        )
        rewriter.insert_op(compbasis_op, insertion_point=InsertPoint.before(observable_op))

        # TODO: this assumes MP acts on 1 wire, what if there are more?
        sample_op = quantum.SampleOp(
            operands=[compbasis_op.results[0], None, None],
            result_types=[builtin.TensorType(builtin.Float64Type(), [shots, 1])],
        )
        rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(compbasis_op))

        # Insert the post-processing function into current module or
        # get handle to it if already inserted
        op_type_str = op.name.lstrip("quantum.")
        postprocessing_func_name = f"{op_type_str}_from_samples.tensor.{shots}x1xf64"
        postprocessing_func_is_inserted = False

        for block_op in func_op.parent.ops:
            if (
                isinstance(block_op, func.FuncOp)
                and block_op.sym_name.data == postprocessing_func_name
            ):
                postprocessing_func_op = block_op
                postprocessing_func_is_inserted = True
                break

        if not postprocessing_func_is_inserted:
            op_type = type(op)
            postprocessor = (
                postprocessing_expval if op_type is quantum.ExpvalOp else postprocessing_var
            )

            # TODO: Do we have to set the shape of the samples array statically here? Or can the
            # shape (shots, wire) be dynamic and given as SSA values?
            # Same goes for the column/wire indices (the second argument).
            postprocessing_module = postprocessor(jax.core.ShapedArray([shots, 1], float), 0)

            postprocessing_func_op = postprocessing_module.body.ops.first
            assert isinstance(postprocessing_func_op, func.FuncOp), (
                f"Expected the first operator of `postprocessing_module` to be a func.FuncOp but "
                f"got {type(postprocessing_func_op).__name__}"
            )

            # Need to clone since original postprocessing_func_op is attached to the jax.jit module
            postprocessing_func_op = postprocessing_func_op.clone()

            # The function name from jax.jit is 'main'; rename it here
            postprocessing_func_op.sym_name = builtin.StringAttr(data=postprocessing_func_name)

            func_op.parent.insert_op_after(postprocessing_func_op, func_op)

        # Insert the op that specifies which column in samples array to access
        # TODO: This also assumes MP acts on a single wire (hence we always use column 0 of
        # the samples here); what if MP acts on multiple wires?
        column_index = builtin.IntegerAttr(0, value_type=64)
        column_index_op = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.create_dense_int(
                type=builtin.TensorType(column_index.type, shape=()), data=column_index
            )
        )
        rewriter.insert_op(column_index_op, insertion_point=InsertPoint.after(sample_op))

        # Insert the call to the post-processing function
        postprocessing_func_call_op = func.CallOp(
            callee=builtin.FlatSymbolRefAttr(postprocessing_func_op.sym_name),
            arguments=[sample_op.results[0], column_index_op],
            return_types=[builtin.TensorType(builtin.Float64Type(), shape=())],
        )

        # The op to replace is not expval/var op itself, but the tensor.from_elements op that follows
        op_to_replace = list(op.results[0].uses)[0].operation
        op_to_replace_name_attr = op_to_replace.attributes.get("op_name__")
        assert (
            op_to_replace_name_attr is not None
            and op_to_replace_name_attr.data == "tensor.from_elements"
        ), f"Expected to replace a tensor.from_elements op but got {type(op_to_replace).__name__}"
        rewriter.replace_op(op_to_replace, postprocessing_func_call_op)

        # Finally, erase the expval/var op and its associated observable op
        rewriter.erase_op(op)
        rewriter.erase_op(observable_op)

    # pylint: disable=arguments-differ,no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Implementation of the match-and-rewrite pattern for FuncOps that may contain terminal
        measurements to be replaced with a single sample measurement."""
        supported_measure_ops = (
            quantum.ExpvalOp,
            quantum.VarianceOp,
            quantum.SampleOp,
            quantum.ProbsOp,
        )
        unsupported_measure_ops = (quantum.CountsOp, quantum.StateOp)
        measure_ops = supported_measure_ops + unsupported_measure_ops

        static_shots_val = None

        print("[DEBUG]: Starting walk")

        for op in func_op.body.walk():
            print(f"[DEBUG]: Visiting op `{op}`")

            if static_shots_val is None and isinstance(op, quantum.DeviceInitOp):
                static_shots_val = self._get_static_shots_value_from_device_op(op)

            if not isinstance(op, measure_ops):
                continue

            if isinstance(op, unsupported_measure_ops):
                raise NotImplementedError(
                    f"The measurements_from_samples transform does not support operations of type "
                    f"'{op.name}'"
                )

            if isinstance(op, quantum.SampleOp):
                print("[DEBUG]: Found sample op")
                continue

            if isinstance(op, quantum.ProbsOp):
                print("[DEBUG]: Found probs op")
                pass

            if isinstance(op, quantum.ExpvalOp):
                print("[DEBUG]: Found expval op")
                self._rewrite_expval_or_var(op, static_shots_val, func_op, rewriter)

            if isinstance(op, quantum.VarianceOp):
                print("[DEBUG]: Found var op")
                self._rewrite_expval_or_var(op, static_shots_val, func_op, rewriter)


@xdsl_transform
@dataclass(frozen=True)
class MeasurementsFromSamplesPass(passes.ModulePass):
    """Pass that replaces all terminal measurements in a program with a single
    :func:`pennylane.sample` measurement, and adds postprocessing instructions to recover the
    original measurement.
    """

    name = "measurements-from-samples"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the measurements-from-samples pass."""
        # TODO: I think we want a walker, rather than a greedy applier; TBD
        pattern_rewriter.PatternRewriteWalker(MeasurementsFromSamplesPattern()).rewrite_module(
            module
        )
        # pattern_rewriter.PatternRewriteWalker(
        #     pattern_rewriter.GreedyRewritePatternApplier([MeasurementsFromSamplesPattern()])
        # ).rewrite_module(module)
