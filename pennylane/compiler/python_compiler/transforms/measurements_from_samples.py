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

from xdsl import context, passes, pattern_rewriter, ir
from xdsl.dialects import builtin, func, tensor
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler import quantum_dialect as quantum
from pennylane.compiler.python_compiler.jax_utils import xdsl_module

# ----- Stuff that should be upstreamed ---------------------------------------------------------- #

import pennylane as qml
from pennylane.devices.preprocess import null_postprocessing

from .apply_transform_sequence import register_pass


def xdsl_transform(_klass):
    """Register the xdsl transform into the plxpr to catalyst map.

    NOTE: This function will eventually live somewhere in the pennylane.compiler.python_compiler
    module, we have to add it here locally until it's added upstream.
    """

    # avoid dependency on catalyst
    import catalyst  # pylint: disable=import-outside-toplevel

    def identity_transform(tape):
        """Stub, we only need the name to be unique"""
        return tape, null_postprocessing

    identity_transform.__name__ = "xdsl_transform" + _klass.__name__
    transform = qml.transform(identity_transform)

    # Map from plxpr to register transform
    catalyst.from_plxpr.register_transform(transform, _klass.name, False)

    # Register this pass as available in the apply-transform-sequence
    # interpreter
    def get_pass_instance():
        return _klass

    # breakpoint()
    register_pass(_klass.name, get_pass_instance)
    return transform


# ----- (END) Stuff that should be upstreamed ---------------------------------------------------- #


@xdsl_module
@jax.jit
def postprocessing_expval(samples, wire):
    return jnp.mean(1.0 - 2.0 * samples[:, wire])


@xdsl_module
@jax.jit
def postprocessing_var(samples, wire):
    return jnp.var(1.0 - 2.0 * samples[:, wire])


@xdsl_module
@jax.jit
def postprocessing_probs(samples, wire):
    raise NotImplementedError("probs not implemented")


@xdsl_module
@jax.jit
def postprocessing_counts(samples, wire):
    raise NotImplementedError("counts not implemented")


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
        ), f"Expected op to be a quantum.NamedObsOp, but got {type(op)}"
        if not op.type.data == "PauliZ":
            raise NotImplementedError(
                f"Observable '{op.type.data}' used as input to expval is not "
                f"supported for the measurements_from_samples transform; currently only "
                f"PauliZ operations are permitted"
            )

    # @classmethod
    # def _get_shots_from_device_op(cls, op: quantum.DeviceInitOp):
    #     """TODO"""
    #     assert isinstance(op, quantum.DeviceInitOp), (
    #         f"Expected op to be a quantum.DeviceInitOp, but got {type(op)}"
    #     )
    #     return op.shots

    # pylint: disable=arguments-differ,no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Implementation of the match-and-rewrite pattern for FuncOps that may contain terminal
        measurements to be replaced with a single sample measurement."""
        # Walk through the func operations
        supported_measure_ops = (quantum.ExpvalOp, quantum.VarianceOp, quantum.SampleOp)
        unsupported_measure_ops = (quantum.ProbsOp, quantum.CountsOp, quantum.StateOp)
        measure_ops = supported_measure_ops + unsupported_measure_ops

        shots = None
        have_inserted_postproc = False

        for op in funcOp.body.walk():
            if shots is None and isinstance(op, quantum.DeviceInitOp):
                # shots = self._get_shots_from_device_op(op)
                shots = op.shots

            if not isinstance(op, measure_ops):
                continue

            if isinstance(op, unsupported_measure_ops):
                raise NotImplementedError(
                    f"The measurements_from_samples transform does not support "
                    f"operations of type '{op.name}'"
                )

            if isinstance(op, quantum.SampleOp):
                return

            if isinstance(op, quantum.ExpvalOp):
                observable_op = op.operands[0].owner
                self._validate_observable_op(observable_op)

                in_qubit = observable_op.operands[0]

                # Steps:
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
                    operands=[compbasis_op.results[0], shots, None],
                    result_types=[builtin.TensorType(builtin.Float64Type(), [-1, 1])],
                )
                rewriter.insert_op(sample_op, insertion_point=InsertPoint.after(compbasis_op))

                # TODO: We can't delete these yet because there are other ops that use their output
                # rewriter.erase_matched_op()
                # rewriter.erase_op(observable_op)

                # Insert the post-processing function
                if not have_inserted_postproc:
                    postprocessing_module = postprocessing_expval(
                        jax.core.ShapedArray([10, 1], int), 1  # FIXME
                    )

                    for _func in postprocessing_module.body.walk():
                        postprocessing_func = _func
                        break
                    postprocessing_func = postprocessing_func.clone()

                    postprocessing_func.sym_name = builtin.StringAttr(data="expval_from_samples")

                    funcOp.parent.insert_op_after(postprocessing_func, funcOp)

                    have_inserted_postproc = True

                # Insert the call to the post-processing function
                postprocessing_func_call_op = func.CallOp(
                    callee=builtin.FlatSymbolRefAttr(postprocessing_func.sym_name),
                    arguments=[sample_op.results[0], in_qubit],  # FIXME
                    return_types=[builtin.Float64Type()],
                )
                rewriter.insert_op(
                    postprocessing_func_call_op, insertion_point=InsertPoint.after(sample_op)
                )

            elif isinstance(op, quantum.VarianceOp):
                pass


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
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([MeasurementsFromSamplesPattern()])
        ).rewrite_module(module)
