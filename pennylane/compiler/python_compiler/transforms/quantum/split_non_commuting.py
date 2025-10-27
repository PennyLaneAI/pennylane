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

"""This file contains a limited prototype of the split_non_commuting pass.

Known Limitations
-----------------

  * Only single-term observables with no coefficients are supported - there is no support for CompositeOp or SymbolicOp observables
  * Only the Expval measurement process is supported
  * There is no option to specify a grouping strategy (this will be more relevant once CompositeOp support is added)
  * Hence, only the "wires" grouping strategy is implemented, not taking into account observable-commutation logic yet.
  * There is no efficient handling of duplicate observables - a circuit that returns multiple measurements on the same observable will split into multiple executions (this will be more relevant once CompositeOp support is added)

Example:
------------------
For the following IR:
```
func.func public circ(...) {
    ...
    %reg0 = quantum.insert ...
    %reg1 = quantum.insert ...
    %0 = func.call @circ.state_evolution(%reg0, ...)
    %q0 = quantum.extract ...
    %q1 = quantum.extract ...
    ...
    %1 = quantum.expval %q0
    %2 = quantum.expval %q1
    ...
    func.return %1, %2
}
```

We want to split the function into two functions based on the wire-based grouping strategy.
```
func.func public circ(...) {
    %0 = func.call @circ.dup0(...)
    %1 = func.call @circ.dup1(...)
    func.return %0, %1
}
func.func circ.dup0(...) {
    ...
    %reg0 = quantum.insert ...
    %reg1 = quantum.insert ...
    %0 = func.call @circ.state_evolution(%reg1, ...)
    %q0 = quantum.extract ...
    ...
    %1 = quantum.expval %q0
    return %1
}
func.func circ.dup1(...) {
    ...
    %reg1 = quantum.insert ...
    %0 = func.call @circ.state_evolution(%reg1, ...)
    %q1 = quantum.extract ...
    ...
    %1 = quantum.expval %q1
    return %1
}
func.func circ.state_evolution(%reg, ...) {
    %q0 = quantum.extract ...
    %q1 = quantum.extract ...
    ...
    %new_reg0 = quantum.insert ...
    %new_reg1 = quantum.insert ...
    return %new_reg1
}
```

reference: https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.split_non_commuting.html
"""

from dataclasses import dataclass
from typing import Type, TypeVar

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum


@dataclass(frozen=True)
class SplitNonCommutingPass(passes.ModulePass):
    """Pass that splits quantum functions measuring non-commuting observables.

    This pass groups measurements using the "wires" grouping strategy and splits
    the function into multiple executions, one per group of measurements.
    """

    name = "split-non-commuting"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the split non-commuting pass to all QNode functions in the module."""
        for op in module.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                rewriter = pattern_rewriter.PatternRewriter(op)
                SplitNonCommutingPattern().match_and_rewrite(op, rewriter)


split_non_commuting_pass = compiler_transform(SplitNonCommutingPass)


class SplitNonCommutingPattern(pattern_rewriter.RewritePattern):
    """Pattern that splits a quantum function into multiple functions based on wire-based grouping.

    Measurements acting on different wires are grouped together, while measurements
    acting on the same wire are split into separate groups.
    """

    def __init__(self):
        self.module: builtin.ModuleOp = None

    T = TypeVar("T")

    def get_parent_of_type(self, op: Operation, kind: Type[T]) -> T | None:
        """Walk up the parent tree until an op of the specified type is found."""
        while (op := op.parent_op()) and not isinstance(op, kind):
            pass
        if not isinstance(op, kind):
            raise ValueError(f"get_parent_of_type: expected {kind} but got {type(op)}, op: {op}")
        return op

    def clone_operations_to_block(self, ops_to_clone, target_block, value_mapper):
        """Clone operations to target block, use value_mapper to update references"""
        for op in ops_to_clone:
            cloned_op = op.clone(value_mapper)
            target_block.add_op(cloned_op)

    def create_dup_function(
        self, func_op: func.FuncOp, i: int, rewriter: pattern_rewriter.PatternRewriter
    ):
        """Create a new function for the dup region by fully cloning the original function."""
        # Use the same signature as the original function
        original_func_type = func_op.function_type
        input_types = list(original_func_type.inputs.data)
        output_types = list(original_func_type.outputs.data)
        fun_type = builtin.FunctionType.from_lists(input_types, output_types)

        dup_func = func.FuncOp(func_op.sym_name.data + ".dup." + str(i), fun_type)
        rewriter.insert_op(dup_func, InsertPoint.at_end(self.module.body.block))

        # Map original function arguments to dup function arguments
        dup_block = dup_func.regions[0].block
        orig_block = func_op.body.block
        value_mapper = {}
        for orig_arg, dup_arg in zip(orig_block.args, dup_block.args):
            value_mapper[orig_arg] = dup_arg

        # Clone all operations except the return statement
        ops_to_clone = []
        return_op = None
        for op in orig_block.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
            else:
                ops_to_clone.append(op)

        # Clone operations
        self.clone_operations_to_block(ops_to_clone, dup_block, value_mapper)

        # Clone the return statement
        if return_op:
            return_values = [value_mapper.get(val, val) for val in return_op.operands]
            new_return_op = func.ReturnOp(*return_values)
            dup_block.add_op(new_return_op)

        # Remove expvals from other groups and update return statement
        self.remove_group(dup_func, i)

        return dup_func

    def remove_group(self, dup_func: func.FuncOp, target_group: int):
        """Remove measurement operations from other groups and update return statement."""
        # Find the return operation in the dup function
        return_op = list(dup_func.body.ops)[-1]

        return_values_to_remove = set[SSAValue]()
        for operand in return_op.operands:
            group_id = self.find_group_for_return_value(operand)
            if group_id != target_group:
                return_values_to_remove.add(operand)

        # collect all operations to remove
        remove_ops = list[Operation]([value.owner for value in return_values_to_remove])

        # update return statement
        self.update_return_statement(dup_func, return_values_to_remove)

        # remove operations
        while remove_ops:
            op = remove_ops.pop(0)
            users = [use.operation for result in list(op.results) for use in list(result.uses)]

            # if the operation has users, skip it
            if len(users) > 0:
                continue

            if not self.is_observable_op(op):
                # keep walking up the chain
                for operand in op.operands:
                    if operand not in remove_ops:
                        remove_ops.append(operand.owner)

            op.detach()
            op.erase()

    def update_return_statement(self, func_op: func.FuncOp, values_to_remove: set[SSAValue]):
        """Update the return statement to remove specified values."""
        # Find the return operation
        return_op = None
        for op in func_op.body.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
                break

        if not return_op:
            return

        # Filter out values to remove
        new_return_values = [val for val in return_op.operands if val not in values_to_remove]

        # Create new return operation
        new_return_op = func.ReturnOp(*new_return_values)

        # Replace the old return operation
        return_op.detach()
        return_op.erase()  # Important: erase to remove operand uses
        func_op.body.block.add_op(new_return_op)

        # Update function signature
        new_output_types = [val.type for val in new_return_values]
        input_types = [arg.type for arg in func_op.body.block.args]
        new_fun_type = builtin.FunctionType.from_lists(input_types, new_output_types)
        func_op.function_type = new_fun_type

    def is_measurement_op(self, op: Operation) -> bool:
        """Check if an operation is a measurement operation."""
        # TODO: support more measurement operations
        if isinstance(op, quantum.ExpvalOp):
            return True
        if isinstance(op, (quantum.VarianceOp, quantum.ProbsOp, quantum.SampleOp)):
            raise NotImplementedError(
                f"measurement operations other than expval are not supported: {op}"
            )
        return False

    def is_observable_op(self, op: Operation) -> bool:
        """Check if an operation is an observable operation."""
        if isinstance(
            op,
            (
                quantum.NamedObsOp,
                quantum.ComputationalBasisOp,
                quantum.HamiltonianOp,
                quantum.TensorOp,
            ),
        ):
            return True
        return False

    def calculate_num_groups(self, func_op: func.FuncOp) -> int:
        """Calculate the number of groups using the "wires" grouping strategy.

        This function groups measurements based on wire overlaps only, disregarding
        the actual commutation relations between observables. Measurements acting on
        different wires are grouped together, while measurements acting on the same
        wire are split into different groups.

        The function also stores the group ID in the "group" attribute of each
        measurement operation, which is later used to handle the splitting mechanics.

        Args:
            func_op: The function operation containing measurements to group.

        Returns:
            The number of groups created.
        """
        # Find all measurement operations in the current function
        measurement_ops = [op for op in func_op.body.ops if self.is_measurement_op(op)]

        # For each measurement operation, find the qubits the operation acts on
        op_to_acted_qubits: dict[Operation, set[SSAValue]] = {
            measurement_op: set() for measurement_op in measurement_ops
        }

        for measurement_op in measurement_ops:
            observable = measurement_op.operands[0]
            op_to_acted_qubits[measurement_op].update(self.get_qubits_from_observable(observable))

        # Group measurements: operations on different qubits can be in the same group
        # Operations on the same qubit must be in different groups
        groups: list[dict[Operation, set[SSAValue]]] = []  # Each group stores op -> qubits mapping

        for measurement_op, qubits in op_to_acted_qubits.items():
            if len(qubits) > 1:
                raise NotImplementedError("operations acting on multiple qubits are not supported")

            # Find a group where no operation acts on any of the same qubits
            assigned_group_id = None

            for group_id, group in enumerate(groups):
                # Get all qubits already used in this group
                used_qubits = set()
                for group_qubits in group.values():
                    used_qubits.update(group_qubits)

                # Check if this measurement's qubits conflict with the group
                if not qubits.intersection(used_qubits):
                    # No conflict - can add to this group
                    group[measurement_op] = qubits
                    assigned_group_id = group_id
                    break

            # If no suitable group found, create a new one
            if assigned_group_id is None:
                assigned_group_id = len(groups)
                groups.append({measurement_op: qubits})

            # Tag the measurement operation with the group attribute
            measurement_op.attributes["group"] = builtin.IntegerAttr(
                assigned_group_id, builtin.IntegerType(64)
            )

        return len(groups)

    def get_qubits_from_observable(self, observable: SSAValue) -> set[SSAValue] | None:
        """Get the qubit used by an observable operation.

        Traces back from an observable to find the qubit it operates on.
        Handles NamedObsOp, ComputationalBasisOp, HamiltonianOp, and TensorOp.
        """
        assert observable.owner is not None, "observable should have an owner"

        acted_qubits = set[SSAValue]()

        obs_op = observable.owner

        # For NamedObsOp, the first operand is the qubit
        if isinstance(obs_op, (quantum.NamedObsOp)):
            acted_qubits.add(obs_op.operands[0])

        # For other observable operations, we need to handle multiple qubits
        elif isinstance(
            obs_op, (quantum.HamiltonianOp, quantum.TensorOp, quantum.ComputationalBasisOp)
        ):
            raise NotImplementedError(f"unsupported observable operation: {obs_op}")

        return acted_qubits

    def analyze_group_return_positions(
        self, func_op: func.FuncOp, num_groups: int
    ) -> dict[int, list[int]]:
        """Analyze which return value positions belong to each group.

        Returns a dict mapping group_id -> list of final return value positions
        Example: {0: [0, 2], 1: [1]} for
        return qml.expval(qml.X(0)), qml.expval(qml.X(1)), qml.expval(qml.Y(0))
        """
        # Find the return operation
        return_op = list(func_op.body.ops)[-1]

        # For each return value, trace back to find its group
        group_positions = {i: [] for i in range(num_groups)}

        for position, return_value in enumerate(return_op.operands):
            # Trace back to find the expval operation
            group_id = self.find_group_for_return_value(return_value)
            if group_id is not None:
                group_positions[group_id].append(position)

        return group_positions

    def find_group_for_return_value(self, return_value: SSAValue) -> int | None:
        """Trace back from a return value to find which group's expval produced it."""
        # BFS backward to find expval
        to_check = [return_value]
        checked = set()

        while to_check:
            val = to_check.pop(0)
            if val in checked:
                continue
            checked.add(val)

            op = val.owner

            # If we found a measurement operation, check its group
            if self.is_measurement_op(op) and "group" in op.attributes:
                group_attr = op.attributes["group"]
                return group_attr.value.data

            # Otherwise, check operands
            to_check.extend([operand for operand in op.operands if operand not in checked])

        return None

    def replace_original_with_calls(
        self,
        func_op: func.FuncOp,
        dup_functions: list[func.FuncOp],
        group_return_positions: dict[int, list[int]],
    ):
        """Replace original function body with calls to dup functions.

        Args:
            dup_functions: List of duplicate functions (one per group)
            group_return_positions: Dict mapping group_id -> list of return positions
        """
        original_block = func_op.body.block

        for op in reversed(func_op.body.ops):
            op.detach()
            op.erase()

        # Collect parameters needed for dup function calls
        # Dup functions take the same parameters as the original begin/end region
        # Look at what original function was using and find corresponding values
        call_args = list(original_block.args)  # Use function arguments as base

        group_results = dict[int, list[SSAValue]]()  # group_id -> list of result values

        for group_id, dup_func in enumerate(dup_functions):
            # Get the function signature to determine result types
            func_type = dup_func.function_type
            result_types = list(func_type.outputs.data)

            # Create the call operation
            call_op = func.CallOp(dup_func.sym_name.data, call_args, result_types)
            original_block.add_op(call_op)

            # Store results for this group
            group_results[group_id] = list(call_op.results)

        # Reconstruct the return statement in the original order
        # Calculate total number of return values
        total_returns = sum(len(positions) for positions in group_return_positions.values())
        final_return_values = [None] * total_returns

        for group_id, positions in group_return_positions.items():
            group_vals = group_results[group_id]
            assert len(group_vals) == len(
                positions
            ), "number of group values and positions must match"

            for i, position in enumerate(positions):
                final_return_values[position] = group_vals[i]

        assert all(
            v is not None for v in final_return_values
        ), "final return values should not be None"

        # Create new return operation
        return_op = func.ReturnOp(*final_return_values)
        original_block.add_op(return_op)

    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Split a quantum function into multiple functions using wire-based grouping.

        Creates one duplicate function per group, where each duplicate function contains
        only the measurements from that group. The original function is replaced with
        calls to these duplicate functions, and the results are combined in the original
        return order.

        Args:
            func_op: The function operation to split.
            rewriter: The pattern rewriter for creating new operations.
        """
        self.module = self.get_parent_of_type(func_op, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"

        # Calculate the number of groups using wires-based grouping strategy
        num_groups = self.calculate_num_groups(func_op)

        # Analyze return value positions for each group
        group_return_positions = self.analyze_group_return_positions(func_op, num_groups)

        # Create dup function for each group
        dup_functions = []
        for i in range(num_groups):
            dup_func = self.create_dup_function(func_op, i, rewriter)
            dup_functions.append(dup_func)

        # Replace original function body with calls to dup functions
        self.replace_original_with_calls(func_op, dup_functions, group_return_positions)
