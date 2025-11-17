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
"""Pattern rewriter API for quantum compilation passes."""

from collections.abc import Sequence
from numbers import Number

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.ir import BlockArgument
from xdsl.ir import Operation as xOperation
from xdsl.ir import Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriterListener, PatternRewriteWalker
from xdsl.rewriter import InsertPoint

from pennylane import math, measurements, ops
from pennylane.exceptions import TransformError
from pennylane.operation import Operator

from ..dialects import quantum
from ..utils import get_constant_from_ssa

_named_observables = (ops.PauliX, ops.PauliY, ops.PauliZ, ops.Identity, ops.Hadamard)
_gate_like_ops = (
    quantum.CustomOp,
    quantum.GlobalPhaseOp,
    quantum.MultiRZOp,
    # TODO: Uncomment once PCPhaseOp is added to Quantum dialect
    # quantum.PCPhaseOp,
    quantum.QubitUnitaryOp,
)


class PLPatternRewriter(PatternRewriter):
    """A ``PatternRewriter`` with abstractions for quantum compilation passes.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriter`` that exposes
    methods to abstract away low-level pattern-rewriting details relevant to
    quantum compilation passes.
    """

    def __init__(self, current_operation: xOperation):
        super().__init__(current_operation)

    def get_qnode(
        self, start_op: xOperation | None = None, get_func: bool = False
    ) -> builtin.ModuleOp | func.FuncOp:
        """Get the module corresponding to the QNode containing the given operation.

        The input operation, or the operation used to initialize the rewriter if an operation
        is not provided, are used to search outer scopes until the module corresponding
        to a QNode is found.

        .. note::

            This method assumes that the module corresponding to a QNode will not contain any
            modules in its body.

        Args:
            start_op (xdsl.ir.Operation): The operation used to begin the search. If ``None``,
                the operation used to initialize the rewriter will be used for the search.
            get_func (bool): If ``True``, the FuncOp corresponding to the QNode will be returned
                instead of the module. ``False`` by default.

        Returns:
            ModuleOp | FuncOp: The QNode module surrounding the current operation, or the QNode
            function if ``get_func`` is ``True``.
        """
        current_op: xOperation = start_op or self.current_operation
        while not isinstance(current_op, builtin.ModuleOp):
            current_op = current_op.parent_op()

        qnode_func = None
        for op in current_op.body.ops():
            if isinstance(op, func.FuncOp) and op.attributes.get("qnode", None):
                qnode_func = op
                break

        if qnode_func is None:
            raise TransformError(f"{current_op} is not inside a QNode's scope.")

        return qnode_func if get_func else current_op

    def insert_constant(self, cst: Number, insertion_point: InsertPoint) -> xOperation:
        """Create a scalar ConstantOp and insert it into the IR.

        Args:
            cst (Number): The scalar to insert into the IR
            insertion_point (InsertPoint): The point in the IR where the ``ConstantOp``
                that creates the constant SSAValue should be inserted

        Returns:
            SSAValue: The SSA value corresponding to the constant
        """
        data = [cst]
        match cst:
            case int():
                elem_type = builtin.IntegerType(64)
            case float():
                elem_type = builtin.Float64Type()
            case cst, bool():
                elem_type = builtin.IntegerType(1)
            case complex():
                elem_type = builtin.ComplexType()
                data = [[cst.real, cst.imag]]
            case _:
                raise TypeError(f"{cst} is not a valid type to insert as a constant.")

        type_ = builtin.TensorType(element_type=elem_type, shape=[])
        constAttr = builtin.DenseIntOrFPElementsAttr.from_list(type_, data)
        constantOp = arith.ConstantOp(constAttr)
        extractOp = tensor.ExtractOp(tensor=constantOp.result, indices=[], result_type=elem_type)

        self.insert_op(constantOp, insertion_point)
        self.insert_op(extractOp, InsertPoint.after(constantOp))

        return extractOp

    def get_num_qubits(self, insertion_point: InsertPoint) -> SSAValue[builtin.I64]:
        """Get the number of qubits.

        All qubits available in the QNode, whether statically or dynamically allocated,
        at a given point in the QNode are returned as a 64-bit integer SSAValue.

        Args:
            insertion_point (InsertPoint): The point in the IR where the instruction
                that gets the number of qubits should be inserted. This is necessary
                because the number of qubits in a program can be different at different
                points in the program when dynamically allocated qubits are present.

        Returns:
            SSAValue[I64]: A 64-bit integer SSAValue corresponding to the number of allocated
            qubits.
        """
        numQubitsOp = quantum.NumQubitsOp()
        self.insert_op(numQubitsOp, insertion_point=insertion_point)
        return numQubitsOp.results[0]

    def get_num_shots(self, as_literal: bool = False) -> SSAValue[builtin.I64] | int | None:
        """Get the number of shots.

        Args:
            as_literal (bool): If ``True``, the shots will be returned as a Python
                integer. If the shots are dynamic, the returned value will be -1.
                If ``False``, an int64 ``SSAValue`` corresponding to the number of shots
                will be returned. False by default

        Returns:
            SSAValue[I64] | int | None: ``int`` if ``is_literal`` is ``True``, else an int64
            ``SSAValue``. If the execution is analytic, ``None`` will be returned.
        """
        try:
            qnode: func.FuncOp = self.get_qnode(get_func=True)
        except TransformError as e:
            raise TransformError(
                "Cannot get the number of shots when rewriting an operation outside the "
                "scope of a QNode."
            ) from e

        # The qnode function always initializes a quantum device using the quantum.DeviceInitOp
        # operation.
        device_init = None
        for op in qnode.body.ops:
            if isinstance(op, quantum.DeviceInitOp):
                device_init = op
                break

        assert device_init is not None

        # If the device is **known** to be analytic, it will not have any operands. Note that even
        # if the DeviceInitOp has shots as its operand, it may be analytic if the shots operand is
        # a constant == 0.
        if len(device_init.operands) == 0:
            return None

        shots = device_init.operands[0]
        if not as_literal:
            return shots

        # If shots are dynamic, they will **always** be an argument to the QNode, else
        # they will be static, and created using a constant-like operation.
        if isinstance(shots, BlockArgument):
            return -1

        shots = get_constant_from_ssa(shots)
        return None if shots == 0 else shots

    def erase_gate(self, op: xOperation) -> None:
        """Erase a quantum gate.

        Safely erase a quantum gate from the module being transformed. This method automatically
        handles and pre-processing required before safely erasing an operation. To erase quantum
        gates, which include ``CustomOp``, ``GlobalPhaseOp``, ``MultiRZOp``, ``PCPhaseOp``, and
        ``QubitUnitaryOp``, it is recommended to use this method instead of ``erase_op``.

        Args:
            op (xdsl.ir.Operation): The operation to erase
        """
        if not isinstance(op, _gate_like_ops):
            raise TypeError(
                f"Cannot erase {op}. 'PLPatternRewriter.erase_op' can only erase "
                "gate-like operations."
            )

        # GlobalPhaseOp does not have any target qubits
        in_qubits = (
            op.in_ctrl_qubits
            if isinstance(op, quantum.GlobalPhaseOp)
            else (op.in_qubits + op.in_ctrl_qubits)
        )
        self.replace_op(op, (), in_qubits)

    def insert_mid_measure(
        self, mcm: measurements.MidMeasureMP, insertion_point: InsertPoint
    ) -> SSAValue[quantum.QubitType]:
        """Insert a PL mid-circuit measurement into the IR at the provided insertion point.

        Args:
            mcm (pennylane.ops.MidMeasureMP): The mid-circuit measurement to insert into
                the IR. Note that the measurement qubit must be an SSAValue.
            insertion_point (InsertPoint): The point in the IR where the operation must
                be inserted.

        Returns:
            xdsl.ir.SSAValue[quantum.QubitType]: The qubit returned by the mid-circuit measurement.
        """
        in_qubit: SSAValue[quantum.QubitType] = mcm.wires[0]
        midMeasureOp = quantum.MeasureOp(in_qubit=in_qubit, postselect=mcm.postselect)
        self.insert_op(midMeasureOp, insertion_point=insertion_point)
        out_qubit = midMeasureOp.out_qubit

        # If resetting, we need to insert a conditional statement that applies a PauliX
        # if we measured |1>. The else block just yields a qubit.
        if mcm.reset:
            true_region = Region()
            with ImplicitBuilder(true_region):
                gate = quantum.CustomOp(gate_name="PauliX", in_qubits=(midMeasureOp.out_qubit,))
                _ = scf.YieldOp(gate.out_qubits[0])

            false_region = Region()
            with ImplicitBuilder(false_region):
                _ = scf.YieldOp(midMeasureOp.out_qubit)

            ifOp = scf.IfOp(
                cond=midMeasureOp.mres,
                return_types=(quantum.QubitType(),),
                true_region=true_region,
                false_region=false_region,
            )
            self.insert_op(ifOp, InsertPoint.after(midMeasureOp))
            out_qubit = ifOp.results[0]

        mcm_qubit_uses = [use for use in in_qubit.uses if use.operation != midMeasureOp]
        in_qubit.replace_by_if(out_qubit, lambda use: use.operation != midMeasureOp)
        for use in mcm_qubit_uses:
            self.notify_op_modified(use.operation)

        return out_qubit

    @staticmethod
    def _get_gate_base(gate: Operator):
        """Get the base op of a gate."""

        if isinstance(gate, ops.Controlled):
            base_gate, ctrl_wires, ctrl_vals, adjoint = PLPatternRewriter._get_gate_base(gate.base)
            ctrl_wires = tuple(gate.control_wires) + tuple(ctrl_wires)
            ctrl_vals = tuple(gate.control_values) + tuple(ctrl_vals)
            return base_gate, ctrl_wires, ctrl_vals, adjoint

        if isinstance(gate, ops.Adjoint):
            base_gate, ctrl_wires, ctrl_vals, adjoint = PLPatternRewriter._get_gate_base(gate.base)
            adjoint = adjoint ^ True
            return base_gate, tuple(ctrl_wires), tuple(ctrl_vals), adjoint

        return gate, (), (), False

    # TODO: fix too-many-statements warning
    # pylint: disable=too-many-statements, too-many-branches
    def insert_gate(
        self, gate: Operator, insertion_point: InsertPoint, params: Sequence[SSAValue] | None = None
    ) -> xOperation:
        r"""Insert a PL gate into the IR at the provided insertion point.

        .. note::

            Inserting state-preparation operations is currently not supported.

        Args:
            gate (~pennylane.operation.Operator): The gate to insert. The wires of the gate must be
                ``QubitType`` ``SSAValue``\ s.
            insertion_point (InsertPoint): The point where the operation should be inserted.
            params (Sequence[SSAValue] | None): For parametric gates, the list of ``SSAValue``\ s that
                should be used as the gate's operands. If not provided, the parameters to ``gate`` will
                be inserted as constants into the program.

        Returns:
            xdsl.ir.Operation: The xDSL operation corresponding to the gate being inserted.
        """
        # TODO: Add support for StatePrep, BasisState
        gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate)
        op_args = {}

        # If the gate is a QubitUnitary and an SSA tensor is not provided as its matrix, then
        # we need to create a constant matrix SSAValue using the gate's matrix.
        if isinstance(gate, ops.QubitUnitary) and not params:
            mat = gate.matrix()
            mat_attr = builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(
                    builtin.ComplexType(builtin.Float64Type()), shape=math.shape(mat)
                ),
                mat,
            )
            constantOp = arith.ConstantOp(value=mat_attr)
            self.insert_op(constantOp, insertion_point=insertion_point)
            insertion_point = InsertPoint.after(constantOp)
            params = [constantOp.results[0]]

        # Create static parameters
        elif not params:
            params = []
            for d in gate.data:
                try:
                    d = float(d)
                except ValueError as e:
                    raise TransformError(
                        "Only values that can be cast into floats can be used as gate "
                        f"parameters. Got {d}."
                    ) from e
                constOp = self.insert_constant(d, insertion_point)
                params.append(constOp.results[0])
                insertion_point = InsertPoint.after(constOp)

            # TODO: Uncomment after PCPhaseOp is added to Quantum dialect
            # # PCPhase has a `dim` hyperparameter which also needs to be inserted into the IR.
            # if isinstance(gate, ops.PCPhase):
            #     constOp = self.insert_constant(float(gate.hyperparameters["dimension"][0]), insertion_point)
            #     params.append(constOp.results[0])
            #     insertion_point = InsertPoint.after(constOp)

        # Different gate types may be represented in MLIR by different operations, which may
        # take slightly different arguments
        match type(gate):
            case ops.GlobalPhase:
                op_class = quantum.GlobalPhaseOp
                assert len(params) == 1
                op_args["params"] = params[0]
            case ops.MultiRZ:
                op_class = quantum.MultiRZOp
                assert len(params) == 1
                op_args["theta"] = params[0]
            case ops.QubitUnitary:
                op_class = quantum.QubitUnitaryOp
                assert len(params) == 1
                op_args["matrix"] = params[0]
            # TODO: Uncomment after PCPhaseOp is added to Quantum dialect
            # case ops.PCPhase:
            #     op_class = quantum.PCPhaseOp
            #     op_args["theta"] = params[0]
            #     op_args["dim"] = params[1]
            case _:
                op_class = quantum.CustomOp
                op_args["gate_name"] = gate.name
                op_args["params"] = params

        # Add qubits/control qubits to args. GlobalPhaseOp does not take qubits, only
        # control qubits
        if not isinstance(gate, ops.GlobalPhase):
            op_args["in_qubits"] = tuple(gate.wires)
        op_args["in_ctrl_qubits"] = tuple(ctrl_wires) if ctrl_wires else None
        in_ctrl_values = None

        # Add ctrl values to args
        if ctrl_vals:
            true_cst = None
            false_cst = None
            if any(ctrl_vals):
                true_cst = self.insert_constant(True, insertion_point=insertion_point)
                insertion_point = InsertPoint.after(true_cst)
            if not all(ctrl_vals):
                false_cst = self.insert_constant(False, insertion_point=insertion_point)
                insertion_point = InsertPoint.after(false_cst)
            in_ctrl_values = tuple(
                true_cst.results[0] if v else false_cst.results[0] for v in ctrl_vals
            )
        op_args["in_ctrl_values"] = in_ctrl_values
        op_args["adjoint"] = adjoint

        gateOp = op_class(**op_args)
        self.insert_op(gateOp, insertion_point=insertion_point)

        # Use getattr for in/out_qubits because GlobalPhaseOp does not have in/out_qubits
        for iq, oq in zip(
            getattr(gateOp, "in_qubits", ()) + tuple(gateOp.in_ctrl_qubits),
            getattr(gateOp, "out_qubits", ()) + tuple(gateOp.out_ctrl_qubits),
            strict=True,
        ):
            in_qubit_uses = [use for use in iq.uses if use.operation != gateOp]
            iq.replace_by_if(oq, lambda use: use.operation != gateOp)
            for use in in_qubit_uses:
                self.notify_op_modified(use.operation)
            # self.ctx.update_qubit(iq, oq)

        return gateOp

    # TODO: Finish implementation
    def swap_gates(self, op1: xOperation, op2: xOperation) -> None:
        """Swap two operations in the IR.

        Args:
            op1 (xdsl.ir.Operation): First operation for the swap
            op2 (xdsl.ir.Operation): Second operation for the swap
        """
        if not (op1 in _gate_like_ops and op2 in _gate_like_ops):
            raise TransformError(f"Can only swap gates. Got {op1}, {op2}")

        if set(op1.results) | set(op2.operands):
            pass
        elif set(op1.operands) | set(op2.results):
            op1, op2 = op2, op1
        else:
            raise TransformError("Cannot swap operations that are not SSA neighbours.")

        # Walk the IR forwards or backwards from the operation with less wires to check if there are
        # any ops that use the same wires as op1 or op2
        n_vals1 = [r for r in op1.results if isinstance(r.type, quantum.QubitType)]
        n_vals2 = [o for o in op2.operands if isinstance(o.type, quantum.QubitType)]
        # If op1 has less results than op2 has operands, do forward traversal from op1
        # Else, do backward traversal from op2.
        if n_vals1 <= n_vals2:
            pass
        else:
            pass

        # Update uses for the swap
        new_op2 = type(op2).create(
            operands=(),
            result_types=(),
            properties=op2.properties,
            attributes=op2.attributes,
            successors=op2.successors,
            regions=op2.regions,
        )
        self.insert_op(new_op2, insertion_point=InsertPoint.before(op1))

        # Reorder the block so that the linear order is valid


# pylint: disable=too-few-public-methods
class PLPatternRewriteWalker(PatternRewriteWalker):
    """A ``PatternRewriteWalker`` for traversing and rewriting modules.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriteWalker that uses a custom
    rewriter that contains abstractions for quantum compilation passes."""

    def _process_worklist(self, listener: PatternRewriterListener) -> bool:
        """
        Process the worklist until it is empty.
        Returns true if any modification was done.
        """
        rewriter_has_done_action = False

        # Handle empty worklist
        op = self._worklist.pop()
        if op is None:
            return rewriter_has_done_action

        # Create a rewriter on the first operation
        # Here, we use our custom rewriter instead of the default PatternRewriter.
        rewriter = PLPatternRewriter(op)
        rewriter.extend_from_listener(listener)

        # do/while loop
        while True:
            # Reset the rewriter on `op`
            rewriter.has_done_action = False
            rewriter.current_operation = op
            rewriter.insertion_point = InsertPoint.before(op)
            rewriter.name_hint = None

            # Apply the pattern on the operation
            try:
                self.pattern.match_and_rewrite(op, rewriter)
            except Exception as err:  # pylint: disable=broad-exception-caught
                op.emit_error(
                    f"Error while applying pattern: {err}",
                    underlying_error=err,
                )
            rewriter_has_done_action |= rewriter.has_done_action

            # If the worklist is empty, we are done
            op = self._worklist.pop()
            if op is None:
                return rewriter_has_done_action
