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

# # Tuple of all operations that return qubits
# _ops_returning_qubits = (
#     quantum.CustomOp,
#     quantum.AllocQubitOp,
#     quantum.ExtractOp,
#     quantum.GlobalPhaseOp,
#     quantum.MeasureOp,
#     quantum.MultiRZOp,
#     quantum.QubitUnitaryOp,
#     quantum.SetBasisStateOp,
#     quantum.SetStateOp,
#     mbqc.MeasureInBasisOp,
# )

# # Tuple of all operations that return "out_qubits"
# _out_qubits_ops = (
#     quantum.CustomOp,
#     quantum.MultiRZOp,
#     quantum.QubitUnitaryOp,
#     quantum.SetBasisStateOp,
#     quantum.SetStateOp,
# )

# # Tuple of all operations that return "out_ctrl_qubits"
# _out_ctrl_qubits_ops = (
#     quantum.CustomOp,
#     quantum.GlobalPhaseOp,
#     quantum.MultiRZOp,
#     quantum.QubitUnitaryOp,
# )

# # Tuple of all operations that return "out_qubit"
# _out_qubit_ops = (quantum.MeasureOp, mbqc.MeasureInBasisOp)

# # Tuple of all operations that return "qubit"
# _qubit_ops = (quantum.AllocQubitOp, quantum.ExtractOp)


# class AbstractWire:
#     """A class representing an abstract wire."""

#     id: UUID

#     def __init__(self):
#         # Create a universally unique identifier
#         self.id = uuid4()

#     def __hash__(self):
#         return hash(self.id)

#     def __eq__(self, other):
#         return self.id == other.id


# class RewriteContext:
#     """A container class for state-keeping during rewrites."""

#     _wires = tuple[int]
#     wire_to_qubit_map: dict[int, list[SSAValue]]
#     qubit_to_wire_map: dict[SSAValue, int]
#     shots: int | None

#     def __init__(self, wires: Sequence[int] | None = None):
#         if wires:
#             self._wires = tuple(wires)
#         else:
#             self._wires = ()
#         self.wire_to_qubit_map = {}
#         self.qubit_to_wire_map = {}
#         self.shots = None

#     @property
#     def wires(self) -> tuple[int]:
#         """Wire labels."""
#         return self._wires

#     def update_qubit(self, old_qubit: SSAValue, new_qubit: SSAValue) -> None:
#         """Update a qubit."""
#         wire = self.qubit_to_wire_map[old_qubit]
#         self.wire_to_qubit_map[wire] = new_qubit
#         self.qubit_to_wire_map[new_qubit] = wire
#         self.qubit_to_wire_map.pop(old_qubit, None)

#     def __getitem__(self, val: int | SSAValue) -> int | SSAValue | None:
#         if isinstance(val, SSAValue):
#             return self.qubit_to_wire_map[val]

#         if self._wires and val not in self._wires:
#             raise CompileError(f"{val} is not an available wire.")
#         return self.wire_to_qubit_map.get(val, None)

#     def __setitem__(self, key: int | SSAValue, item: SSAValue | int) -> None:
#         if isinstance(key, SSAValue):
#             old_wire = self.qubit_to_wire_map.pop(key, None)
#             self.wire_to_qubit_map.pop(old_wire, None)
#             self.qubit_to_wire_map[key] = item
#             self.wire_to_qubit_map[item] = key
#         else:
#             old_qubit = self.wire_to_qubit_map.pop(key, None)
#             self.qubit_to_wire_map.pop(old_qubit, None)
#             self.wire_to_qubit_map[key] = item
#             self.qubit_to_wire_map[item] = key

#     def get_static_wire(self, op: quantum.ExtractOp, update=True) -> int | None:
#         """Get the wire label to which a qubit extraction corresponds."""
#         wire = None
#         if (idx_attr := getattr(op, "idx_attr", None)) is not None:
#             wire = idx_attr.value.data

#         else:
#             idx = op.idx
#             if isinstance(idx.owner, arith.ConstantOp):
#                 wire = idx.owner.properties["value"].data
#             elif isinstance(idx.owner, tensor.ExtractOp):
#                 operand = idx.owner.operands[0]
#                 if isinstance(operand.owner, stablehlo.ConstantOp):
#                     wire = operand.owner.properties["value"].get_values()[0]

#         if wire is not None and update:
#             self[wire] = op.qubit
#         return wire

#     def update_from_op(self, op: xOperation):
#         """Update the wire mapping from an operation's outputs"""
#         # pylint: disable=too-many-branches
#         if isinstance(op, quantum.DeviceInitOp):
#             shots = getattr(op, "shots", None)
#             if shots:
#                 shots_owner = shots.owner
#                 if isinstance(shots_owner, arith.ConstantOp):
#                     self.shots = shots_owner.value.data
#                 else:
#                     assert isinstance(shots_owner, tensor.ExtractOp)
#                     assert isinstance(shots_owner.operands[0], BlockArgument)
#                     self.shots = shots
#             shots = 0
#             return

#         for r in op.results:
#             if isinstance(r, quantum.QuregType):
#                 if isinstance(op, quantum.AllocOp):
#                     nqubits = getattr(op, "nqubits_attr", None)
#                     if nqubits:
#                         self.n_qubits = nqubits.data
#                     else:
#                         assert isinstance(op.nqubits.owner, tensor.ExtractOp)
#                         assert isinstance(op.nqubits.owner.operands[0], BlockArgument)
#                         self.n_qubits = op.nqubits

#                 self.qreg = r
#                 # We assume that only one of the results is a QuregType
#                 break

#         if isinstance(op, quantum.ExtractOp):
#             _ = self.get_static_wire(op, update=True)
#             return

#         if isinstance(op, _out_qubit_ops):
#             self.update_qubit(op.in_qubit, op.out_qubit)

#         if isinstance(op, _out_qubits_ops):
#             for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
#                 self.update_qubit(iq, oq)

#         if isinstance(op, _out_ctrl_qubits_ops):
#             for iq, oq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
#                 self.update_qubit(iq, oq)

#         if isinstance(op, (quantum.InsertOp, quantum.DeallocQubitOp)):
#             qubit = op.qubit
#             wire = self.qubit_to_wire_map.pop(qubit, None)
#             _ = self.wire_to_qubit_map.pop(wire, None)


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
        # self.ctx = RewriteContext()

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
    # pylint: disable=too-many-statements
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

    # def insert_observable(self, obs: Operator, insertion_point: InsertPoint) -> xOperation:
    #     """Insert a PL observable into the IR at the provided insertion point."""
    #     if isinstance(obs, ops.Hermitian):
    #         # Create static matrix
    #         mat = obs.matrix()
    #         mat_attr = builtin.DenseIntOrFPElementsAttr.from_list(
    #             builtin.TensorType(
    #                 builtin.ComplexType(builtin.Float64Type()), shape=math.shape(mat)
    #             ),
    #             mat,
    #         )
    #         tensorOp = stablehlo.ConstantOp(value=mat_attr)
    #         self.insert_op(tensorOp, insertion_point=insertion_point)
    #         insertion_point = InsertPoint.after(tensorOp)

    #         in_qubits = tuple(self.ctx[w] for w in obs.wires)
    #         hermitianOp = quantum.HermitianOp(
    #             operands=(tensorOp.results[0], in_qubits), result_types=(quantum.ObservableType(),)
    #         )
    #         self.insert_op(hermitianOp, insertion_point=insertion_point)
    #         return hermitianOp

    #     if isinstance(obs, _named_observables):
    #         in_qubit = self.ctx[obs.wires[0]]
    #         namedObsOp = quantum.NamedObsOp(
    #             in_qubit, quantum.NamedObservableAttr(getattr(quantum.NamedObservable, obs.name))
    #         )
    #         self.insert_op(namedObsOp, insertion_point=InsertPoint)
    #         return namedObsOp

    #     if isinstance(obs, ops.Prod):
    #         operands = []
    #         for o in obs.operands:
    #             cur_obs = self.insert_observable(o, insertion_point)
    #             operands.append(cur_obs.results[0])
    #             insertion_point = InsertPoint.after(cur_obs)
    #         prodOp = quantum.TensorOp(operands=operands, result_types=(quantum.ObservableType(),))
    #         self.insert_op(prodOp, insertion_point=insertion_point)
    #         return prodOp

    #     if isinstance(obs, ops.LinearCombination):
    #         # Create static tensor for coefficients
    #         coeffs = builtin.DenseIntOrFPElementsAttr.from_list(
    #             builtin.TensorType(builtin.Float64Type(), shape=(len(obs.coeffs),)), obs.coeffs
    #         )
    #         tensorOp = stablehlo.ConstantOp(value=coeffs)
    #         self.insert_op(tensorOp, insertion_point=insertion_point)
    #         insertion_point = InsertPoint.after(tensorOp)

    #         _ops = []
    #         for o in obs.ops:
    #             cur_obs = self.insert_observable(o, insertion_point)
    #             _ops.append(cur_obs.results[0])
    #             insertion_point = InsertPoint.after(cur_obs)

    #         hamiltonianOp = quantum.HamiltonianOp(
    #             operands=(tensorOp.results[0], _ops), result_types=(quantum.ObservableType(),)
    #         )
    #         return hamiltonianOp

    #     raise CompileError(
    #         f"The observable {type(obs).__name__} cannot be inserted into the module."
    #     )

    # def _resolve_dynamic_shape(
    #     self, n_qubits: SSAValue | int, insertion_point: InsertPoint
    # ) -> tuple[tuple[int], tuple[SSAValue] | None, InsertPoint]:
    #     """Get dynamic shape and output tensor shape for dynamic shape for observables
    #     when number of qubits is not known at compile time."""

    #     if isinstance(n_qubits, SSAValue):
    #         # If number of qubits is not known, we indicate the dynamic shape, and
    #         # set the shape of the resulting tensor to (-1,), indicating that the shape is
    #         # not known at compile time.
    #         tensor_shape = (-1,)
    #         # Create dynamic shape, which is (2**num_qubits,) or (1 << num_qubits,)
    #         const1Op = self.insert_constant(1, insertion_point=insertion_point)
    #         leftShiftOp = arith.ShLIOp(const1Op, n_qubits)
    #         self.insert_op(leftShiftOp, insertion_point=InsertPoint.after(const1Op))
    #         insertion_point = InsertPoint.after(leftShiftOp)
    #         dynamic_shape = (leftShiftOp.results[0],)
    #     else:
    #         tensor_shape = (2**n_qubits,)
    #         dynamic_shape = None

    #     return tensor_shape, dynamic_shape, insertion_point

    # def insert_measurement(
    #     self,
    #     mp: measurements.MeasurementProcess,
    #     qreg: quantum.QuregType,
    #     insertion_point: InsertPoint,
    # ) -> None:  # pylint: disable=too-many-statements
    #     """Insert a PL measurement into the IR at the provided insertion point.

    #     .. note::

    #         This method assumes that there are no dangling qubits.
    #     """
    #     # pylint: disable=too-many-statements

    #     # Create and insert an ObservableType to be used by the xDSL measurement op.
    #     # After inserting, update the insertion point to be after the op(s) created in
    #     # the below branches.
    #     if mp.mv:
    #         raise CompileError(
    #             "Inserting measurements that collect statistics on mid-circuit measurements "
    #             "is currently not supported."
    #         )

    #     if mp.obs:
    #         obs = self.insert_observable(mp.obs, insertion_point)
    #         n_qubits = len(mp.obs.wires)

    #         insertion_point = InsertPoint.after(obs)

    #     elif mp.wires:
    #         obs = quantum.ComputationalBasisOp(
    #             operands=(tuple(self.ctx[w] for w in mp.wires), None),
    #             result_types=(quantum.ObservableType()),
    #         )
    #         n_qubits = len(mp.wires)
    #         self.insert_op(obs, insertion_point=insertion_point)

    #         insertion_point = InsertPoint.after(obs)

    #     else:
    #         # Measurement on all wires
    #         obs = quantum.ComputationalBasisOp(
    #             operands=((), self.ctx.qreg), result_types=(quantum.ObservableType(),)
    #         )
    #         self.insert_op(obs, insertion_point=insertion_point)

    #         if not self.ctx.n_qubits:
    #             numQubitsOp = quantum.NumQubitsOp()
    #             n_qubits = numQubitsOp.results[0]
    #             self.ctx.n_qubits = n_qubits
    #             self.insert_op(numQubitsOp, InsertPoint.after(obs))

    #             insertion_point = InsertPoint.after(numQubitsOp)
    #         else:
    #             insertion_point = InsertPoint.after(obs)

    #     # Create the measurement xDSL operation
    #     match type(mp):
    #         case measurements.ExpectationMP:
    #             measurementOp = quantum.ExpvalOp(obs=obs)

    #         case measurements.VarianceMP:
    #             measurementOp = quantum.VarianceOp(obs=obs)

    #         case measurements.StateMP:
    #             # For now, we assume that there is no input MemRefType
    #             tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
    #                 n_qubits, insertion_point
    #             )
    #             measurementOp = quantum.StateOp(
    #                 operands=(obs, dynamic_shape, None),
    #                 result_types=(
    #                     builtin.TensorType(
    #                         element_type=builtin.ComplexType(builtin.Float64Type()),
    #                         shape=tensor_shape,
    #                     )
    #                 ),
    #             )

    #         case measurements.ProbabilityMP:
    #             # For now, we assume that there is no input MemRefType
    #             tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
    #                 n_qubits, insertion_point
    #             )
    #             measurementOp = quantum.ProbsOp(
    #                 operands=(obs, dynamic_shape, None, None),
    #                 result_types=(
    #                     builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
    #                 ),
    #             )

    #         case measurements.SampleMP:
    #             #  n_qubits or shots may not be known at compile time
    #             _iter = (self.ctx.shots, n_qubits)
    #             tensor_shape = tuple(-1 if isinstance(i, SSAValue) else i for i in _iter)
    #             # We only insert values into dynamic_shape that are unknown at compile time
    #             dynamic_shape = tuple(i for i in _iter if isinstance(i, SSAValue))
    #             if isinstance(n_qubits, int) and n_qubits == 1:
    #                 tensor_shape = (tensor_shape[0],)

    #             measurementOp = quantum.SampleOp(
    #                 operands=(obs, dynamic_shape, None),
    #                 result_types=(
    #                     builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
    #                 ),
    #             )

    #         case measurements.CountsMP:
    #             # For now, we assume that there are no input MemRefTypes
    #             tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
    #                 n_qubits, insertion_point
    #             )
    #             measurementOp = quantum.CountsOp(
    #                 operands=(obs, dynamic_shape, None, None),
    #                 result_types=(
    #                     builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
    #                     builtin.TensorType(element_type=builtin.i64, shape=tensor_shape),
    #                 ),
    #             )

    #         case _:
    #             raise CompileError(
    #                 f"The measurement {type(mp).__name__} cannot be supported into the module."
    #             )

    #     self.insert_op(measurementOp, insertion_point=insertion_point)


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
