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
from xdsl.dialects import arith, builtin, scf, stablehlo, tensor
from xdsl.ir import BlockArgument
from xdsl.ir import Operation as xOperation
from xdsl.ir import Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriterListener, PatternRewriteWalker
from xdsl.rewriter import InsertPoint

from pennylane import math, measurements, ops
from pennylane.exceptions import CompileError
from pennylane.operation import Operator

from ...dialects import mbqc, quantum

# Tuple of all operations that return qubits
_ops_returning_qubits = (
    quantum.CustomOp,
    quantum.AllocQubitOp,
    quantum.ExtractOp,
    quantum.GlobalPhaseOp,
    quantum.MeasureOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
    mbqc.MeasureInBasisOp,
)

# Tuple of all operations that return "out_qubits"
_out_qubits_ops = (
    quantum.CustomOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
)

# Tuple of all operations that return "out_ctrl_qubits"
_out_ctrl_qubits_ops = (
    quantum.CustomOp,
    quantum.GlobalPhaseOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
)

# Tuple of all operations that return "out_qubit"
_out_qubit_ops = (quantum.MeasureOp, mbqc.MeasureInBasisOp)

# Tuple of all operations that return "qubit"
_qubit_ops = (quantum.AllocQubitOp, quantum.ExtractOp)


class StateManagement:
    """A container class for managing wire mapping."""

    _wires = tuple[int]
    wire_to_qubit_map: dict[int, SSAValue]
    qubit_to_wire_map: dict[SSAValue, int]
    qreg: quantum.QuregType | None
    n_qubits: int | None
    shots: int | None

    def __init__(self, wires: Sequence[int] | None = None):
        if wires:
            self._wires = tuple(wires)
            self.n_qubits = len(wires)
        else:
            self._wires = ()
            self.n_qubits = None
        self.wire_to_qubit_map = {}
        self.qubit_to_wire_map = {}
        self.qreg = None
        self.shots = None

    @property
    def wires(self) -> tuple[int]:
        """Wire labels."""
        return self._wires

    def update_qubit(self, old_qubit: SSAValue, new_qubit: SSAValue) -> None:
        """Update a qubit."""
        wire = self.qubit_to_wire_map[old_qubit]
        self.wire_to_qubit_map[wire] = new_qubit
        self.qubit_to_wire_map[new_qubit] = wire
        self.qubit_to_wire_map.pop(old_qubit, None)

    def __getitem__(self, val: int | SSAValue) -> int | SSAValue | None:
        if isinstance(val, SSAValue):
            return self.qubit_to_wire_map[val]

        if self._wires and val not in self._wires:
            raise CompileError(f"{val} is not an available wire.")
        return self.wire_to_qubit_map.get(val, None)

    def __setitem__(self, key: int | SSAValue, item: SSAValue | int) -> None:
        if isinstance(key, SSAValue):
            old_wire = self.qubit_to_wire_map.pop(key, None)
            self.wire_to_qubit_map.pop(old_wire, None)
            self.qubit_to_wire_map[key] = item
            self.wire_to_qubit_map[item] = key
        else:
            old_qubit = self.wire_to_qubit_map.pop(key, None)
            self.qubit_to_wire_map.pop(old_qubit, None)
            self.wire_to_qubit_map[key] = item
            self.qubit_to_wire_map[item] = key

    def get_static_wire(self, op: quantum.ExtractOp, update=True) -> int | None:
        """Get the wire label to which a qubit extraction corresponds."""
        wire = None
        if (idx_attr := getattr(op, "idx_attr", None)) is not None:
            wire = idx_attr.value.data

        else:
            idx = op.idx
            if isinstance(idx.owner, arith.ConstantOp):
                wire = idx.owner.properties["value"].data
            elif isinstance(idx.owner, tensor.ExtractOp):
                operand = idx.owner.operands[0]
                if isinstance(operand.owner, stablehlo.ConstantOp):
                    wire = operand.owner.properties["value"].get_values()[0]

        if wire is not None and update:
            self[wire] = op.qubit
        return wire

    def update_from_op(self, op: xOperation):
        """Update the wire mapping from an operation's outputs"""
        # pylint: disable=too-many-branches
        if isinstance(op, quantum.DeviceInitOp):
            shots = getattr(op, "shots", None)
            if shots:
                shots_owner = shots.owner
                if isinstance(shots_owner, arith.ConstantOp):
                    self.shots = shots_owner.value.data
                else:
                    assert isinstance(shots_owner, tensor.ExtractOp)
                    assert isinstance(shots_owner.operands[0], BlockArgument)
                    self.shots = shots
            shots = 0
            return

        for r in op.results:
            if isinstance(r, quantum.QuregType):
                if isinstance(op, quantum.AllocOp):
                    nqubits = getattr(op, "nqubits_attr", None)
                    if nqubits:
                        self.n_qubits = nqubits.data
                    else:
                        assert isinstance(op.nqubits.owner, tensor.ExtractOp)
                        assert isinstance(op.nqubits.owner.operands[0], BlockArgument)
                        self.n_qubits = op.nqubits

                self.qreg = r
                # We assume that only one of the results is a QuregType
                break

        if isinstance(op, quantum.ExtractOp):
            _ = self.get_static_wire(op, update=True)
            return

        if isinstance(op, _out_qubit_ops):
            self.update_qubit(op.in_qubit, op.out_qubit)

        if isinstance(op, _out_qubits_ops):
            for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, _out_ctrl_qubits_ops):
            for iq, oq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, (quantum.InsertOp, quantum.DeallocQubitOp)):
            qubit = op.qubit
            wire = self.qubit_to_wire_map.pop(qubit, None)
            _ = self.wire_to_qubit_map.pop(wire, None)


def _get_bfs_out_qubits(op):
    out_qubits = ()
    if not isinstance(op, _ops_returning_qubits):
        return out_qubits

    if isinstance(op, _out_qubits_ops):
        out_qubits += tuple(op.out_qubits)
    if isinstance(op, _out_ctrl_qubits_ops):
        out_qubits += tuple(op.out_ctrl_qubits)
    if isinstance(op, _out_qubit_ops):
        out_qubits += (op.out_qubit,)
    if isinstance(op, _qubit_ops):
        out_qubits += (op.qubit,)

    return out_qubits


_named_observables = (ops.PauliX, ops.PauliY, ops.PauliZ, ops.Identity, ops.Hadamard)


# TODO: Integration StateManagement with rewriting


class PLPatternRewriter(PatternRewriter):
    """A ``PatternRewriter`` with abstractions for quantum compilation passes.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriter`` that exposes
    methods to abstract away low-level pattern-rewriting details relevant to
    quantum compilation passes.
    """

    def __init__(self, current_operation: xOperation):
        super().__init__(current_operation)
        self.wire_manager = StateManagement()

    def erase_quantum_gate_op(self, op: xOperation, update_qubits: bool = True) -> None:
        """Erase a quantum gate.

        Safely erase a quantum gate from the module being transformed. This method automatically
        handles and pre-processing required before safely erasing an operation. To erase quantum
        gates, which include ``CustomOp``, ``MultiRZOp``, and ``QubitUnitaryOp``, it is recommended
        to use this method instead of ``erase_op``.

        Args:
            op (xdsl.ir.Operation): The operation to erase
        """
        if not isinstance(op, (quantum.CustomOp, quantum.MultiRZOp, quantum.QubitUnitaryOp)):
            return

        # We can also use the following code to perform the same task:
        # self.replace_op(op, (), op.in_qubits + op.in_ctrl_qubits)

        for iq, oq in zip(
            op.in_qubits + op.in_ctrl_qubits, op.out_qubits + op.out_ctrl_qubits, strict=True
        ):
            self.replace_all_uses_with(oq, iq)
            if update_qubits:
                self.wire_manager.update_qubit(oq, iq)

        self.erase_op(op)

    def create_scalar_constant(self, cst: Number, insertion_point: InsertPoint) -> xOperation:
        """Create a scalar ConstantOp and insert it into the IR. The corresponding SSA value is returned."""
        data = [cst]
        if isinstance(cst, float):
            elem_type = builtin.Float64Type()
        elif isinstance(cst, complex):
            elem_type = builtin.ComplexType()
        elif isinstance(cst, bool):
            elem_type = builtin.IntegerType(1)
        else:
            elem_type = builtin.IntegerType(64)

        type_ = builtin.TensorType(elem_type, [1])
        constAttr = builtin.DenseIntOrFPElementsAttr.from_list(type_, data)
        constantOp = arith.ConstantOp(constAttr)
        indexOp = arith.ConstantOp.from_int_and_width(0, 64)
        extractOp = tensor.ExtractOp(
            tensor=constantOp.result, indices=indexOp.result, result_type=elem_type
        )

        self.insert_op(constantOp, insertion_point)
        self.insert_op(indexOp, insertion_point)
        self.insert_op(extractOp, insertion_point)

        return extractOp

    def iter_qubit_successors(self, op: xOperation, traversal_type="bfs"):
        """Iterator function to do a breadth-first traversal over the output qubits
        of an operation. First returned value is the original operation."""
        if traversal_type not in ("bfs", "dfs"):
            raise CompileError(
                f"Unrecognized traversal type {traversal_type}. Valid types are 'bfs' and 'dfs'"
            )
        pop_idx = 0 if traversal_type == "bfs" else -1
        op_queue = [op]

        while op_queue:
            cur_op = op_queue.pop(pop_idx)
            out_qubits = _get_bfs_out_qubits(op)
            for q in out_qubits:
                for use in q.uses:
                    use_op = use.operation
                    if use_op not in op_queue:
                        op_queue.append(use_op)

            yield cur_op

    def _get_gate_base(self, gate: Operator):
        """Get the base op of a gate."""
        if not isinstance(gate, ops.SymbolicOp):
            return gate, (), (), False

        if isinstance(gate, ops.Controlled):
            base_gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate.base)
            ctrl_wires = tuple(gate.control_wires) + tuple(ctrl_wires)
            ctrl_vals = tuple(gate.control_values) + tuple(ctrl_vals)
            return base_gate, ctrl_wires, ctrl_vals, adjoint

        if isinstance(gate, ops.Adjoint):
            base_gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate.base)
            adjoint = adjoint ^ True
            return base_gate, tuple(ctrl_wires), tuple(ctrl_vals), adjoint

        raise CompileError(f"The gate {type(gate).__name__} cannot be inserted into the module.")

    def insert_gate(
        self, gate: Operator, insertion_point: InsertPoint, params: Sequence[SSAValue] | None = None
    ) -> xOperation:
        """Insert a PL gate into the IR at the provided insertion point."""
        # TODO: Add support for StatePrep, BasisState
        gate, ctrl_wires, ctrl_vals, adjoint = self._get_gate_base(gate)
        op_args = {"adjoint": adjoint}

        if isinstance(gate, ops.QubitUnitary):
            # Create static matrix
            if not params:
                mat = gate.matrix()
                mat_attr = builtin.DenseIntOrFPElementsAttr.from_list(
                    builtin.TensorType(
                        builtin.ComplexType(builtin.Float64Type()), shape=math.shape(mat)
                    ),
                    mat,
                )
                tensorOp = stablehlo.ConstantOp(value=mat_attr)
                self.insert_op(tensorOp, insertion_point=insertion_point)
                insertion_point = InsertPoint.after(tensorOp)
                params = [tensorOp.results[0]]

        # Create static parameters
        elif not params:
            params = []
            for d in gate.data:
                constOp = self.create_scalar_constant(d, insertion_point)
                params.append(constOp.results[0])
                insertion_point = InsertPoint.after(constOp)

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
            case _:
                op_class = quantum.CustomOp
                op_args["gate_name"] = gate.name
                op_args["params"] = params

        # Add qubits/control qubits to args. GlobalPhaseOp does not take qubits, only
        # control qubits
        if not isinstance(gate, ops.GlobalPhase):
            op_args["in_qubits"] = tuple(self.wire_manager[w] for w in gate.wires)
        op_args["in_ctrl_qubits"] = (
            tuple(self.wire_manager[w] for w in ctrl_wires) if ctrl_wires else None
        )
        in_ctrl_values = None

        # Add ctrl values to args
        if ctrl_vals:
            true_cst = None
            false_cst = None
            if any(ctrl_vals):
                true_cst = self.create_scalar_constant(True, insertion_point=insertion_point)
                insertion_point = InsertPoint.after(true_cst)
            if not all(ctrl_vals):
                false_cst = self.create_scalar_constant(False, insertion_point=insertion_point)
                insertion_point = InsertPoint.after(false_cst)
            in_ctrl_values = tuple(
                true_cst.results[0] if v else false_cst.results[0] for v in ctrl_vals
            )
        op_args["in_ctrl_values"] = in_ctrl_values

        gateOp = op_class(**op_args)
        self.insert_op(gateOp, insertion_point=insertion_point)

        # Use getattr for in/out_qubits because GlobalPhaseOp does not have in/out_qubits
        for iq, oq in zip(
            getattr(gateOp, "in_qubits", ()) + tuple(gateOp.in_ctrl_qubits),
            getattr(gateOp, "out_qubits", ()) + tuple(gateOp.out_ctrl_qubits),
            strict=True,
        ):
            iq.replace_by_if(oq, lambda use: use.operation != gateOp)
            self.notify_op_modified(gateOp)
            self.wire_manager.update_qubit(iq, oq)

        return gateOp

    def insert_observable(self, obs: Operator, insertion_point: InsertPoint) -> xOperation:
        """Insert a PL observable into the IR at the provided insertion point."""
        if isinstance(obs, ops.Hermitian):
            # Create static matrix
            mat = obs.matrix()
            mat_attr = builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(
                    builtin.ComplexType(builtin.Float64Type()), shape=math.shape(mat)
                ),
                mat,
            )
            tensorOp = stablehlo.ConstantOp(value=mat_attr)
            self.insert_op(tensorOp, insertion_point=insertion_point)
            insertion_point = InsertPoint.after(tensorOp)

            in_qubits = tuple(self.wire_manager[w] for w in obs.wires)
            hermitianOp = quantum.HermitianOp(
                operands=(tensorOp.results[0], in_qubits), result_types=(quantum.ObservableType(),)
            )
            self.insert_op(hermitianOp, insertion_point=insertion_point)
            return hermitianOp

        if isinstance(obs, _named_observables):
            in_qubit = self.wire_manager[obs.wires[0]]
            namedObsOp = quantum.NamedObsOp(
                in_qubit, quantum.NamedObservableAttr(getattr(quantum.NamedObservable, obs.name))
            )
            self.insert_op(namedObsOp, insertion_point=InsertPoint)
            return namedObsOp

        if isinstance(obs, ops.Prod):
            operands = []
            for o in obs.operands:
                cur_obs = self.insert_observable(o, insertion_point)
                operands.append(cur_obs.results[0])
                insertion_point = InsertPoint.after(cur_obs)
            prodOp = quantum.TensorOp(operands=operands, result_types=(quantum.ObservableType(),))
            self.insert_op(prodOp, insertion_point=insertion_point)
            return prodOp

        if isinstance(obs, ops.LinearCombination):
            # Create static tensor for coefficients
            coeffs = builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(builtin.Float64Type(), shape=(len(obs.coeffs),)), obs.coeffs
            )
            tensorOp = stablehlo.ConstantOp(value=coeffs)
            self.insert_op(tensorOp, insertion_point=insertion_point)
            insertion_point = InsertPoint.after(tensorOp)

            _ops = []
            for o in obs.ops:
                cur_obs = self.insert_observable(o, insertion_point)
                _ops.append(cur_obs.results[0])
                insertion_point = InsertPoint.after(cur_obs)

            hamiltonianOp = quantum.HamiltonianOp(
                operands=(tensorOp.results[0], _ops), result_types=(quantum.ObservableType(),)
            )
            return hamiltonianOp

        raise CompileError(
            f"The observable {type(obs).__name__} cannot be inserted into the module."
        )

    def _resolve_dynamic_shape(
        self, n_qubits: SSAValue | int, insertion_point: InsertPoint
    ) -> tuple[tuple[int], tuple[SSAValue] | None, InsertPoint]:
        """Get dynamic shape and output tensor shape for dynamic shape for observables
        when number of qubits is not known at compile time."""

        if isinstance(n_qubits, SSAValue):
            # If number of qubits is not known, we indicate the dynamic shape, and
            # set the shape of the resulting tensor to (-1,), indicating that the shape is
            # not known at compile time.
            tensor_shape = (-1,)
            # Create dynamic shape, which is (2**num_qubits,) or (1 << num_qubits,)
            const1Op = self.create_scalar_constant(1, insertion_point=insertion_point)
            leftShiftOp = arith.ShLIOp(const1Op, n_qubits)
            self.insert_op(leftShiftOp, insertion_point=InsertPoint.after(const1Op))
            insertion_point = InsertPoint.after(leftShiftOp)
            dynamic_shape = (leftShiftOp.results[0],)
        else:
            tensor_shape = (2**n_qubits,)
            dynamic_shape = None

        return tensor_shape, dynamic_shape, insertion_point

    def insert_measurement(
        self, mp: measurements.MeasurementProcess, insertion_point: InsertPoint
    ) -> None:  # pylint: disable=too-many-statements
        """Insert a PL measurement into the IR at the provided insertion point."""
        # pylint: disable=too-many-statements

        # Create and insert an ObservableType to be used by the xDSL measurement op.
        # After inserting, update the insertion point to be after the op(s) created in
        # the below branches.
        if mp.mv:
            raise CompileError(
                "Inserting measurements that collect statistics on mid-circuit measurements "
                "is currently not supported."
            )

        if mp.obs:
            obs = self.insert_observable(mp.obs, insertion_point)
            n_qubits = len(mp.obs.wires)

            insertion_point = InsertPoint.after(obs)

        elif mp.wires:
            obs = quantum.ComputationalBasisOp(
                operands=(tuple(self.wire_manager[w] for w in mp.wires), None),
                result_types=(quantum.ObservableType()),
            )
            n_qubits = len(mp.wires)
            self.insert_op(obs, insertion_point=insertion_point)

            insertion_point = InsertPoint.after(obs)

        else:
            # Measurement on all wires
            obs = quantum.ComputationalBasisOp(
                operands=((), self.wire_manager.qreg), result_types=(quantum.ObservableType(),)
            )
            self.insert_op(obs, insertion_point=insertion_point)

            if not self.wire_manager.n_qubits:
                numQubitsOp = quantum.NumQubitsOp()
                n_qubits = numQubitsOp.results[0]
                self.wire_manager.n_qubits = n_qubits
                self.insert_op(numQubitsOp, InsertPoint.after(obs))

                insertion_point = InsertPoint.after(numQubitsOp)
            else:
                insertion_point = InsertPoint.after(obs)

        # Create the measurement xDSL operation
        match type(mp):
            case measurements.ExpectationMP:
                measurementOp = quantum.ExpvalOp(obs=obs)

            case measurements.VarianceMP:
                measurementOp = quantum.VarianceOp(obs=obs)

            case measurements.StateMP:
                # For now, we assume that there is no input MemRefType
                tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
                    n_qubits, insertion_point
                )
                measurementOp = quantum.StateOp(
                    operands=(obs, dynamic_shape, None),
                    result_types=(
                        builtin.TensorType(
                            element_type=builtin.ComplexType(builtin.Float64Type()),
                            shape=tensor_shape,
                        )
                    ),
                )

            case measurements.ProbabilityMP:
                # For now, we assume that there is no input MemRefType
                tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
                    n_qubits, insertion_point
                )
                measurementOp = quantum.ProbsOp(
                    operands=(obs, dynamic_shape, None, None),
                    result_types=(
                        builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
                        builtin.TensorType(element_type=builtin.i64, shape=tensor_shape),
                    ),
                )

            case measurements.SampleMP:
                #  n_qubits or shots may not be known at compile time
                _iter = (self.wire_manager.shots, n_qubits)
                tensor_shape = tuple(-1 if isinstance(i, SSAValue) else i for i in _iter)
                # We only insert values into dynamic_shape that are unknown at compile time
                dynamic_shape = tuple(i for i in _iter if isinstance(i, SSAValue))

                measurementOp = quantum.SampleOp(
                    operands=(obs, dynamic_shape, None),
                    result_types=(
                        builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
                    ),
                )

            case measurements.CountsMP:
                # For now, we assume that there are no input MemRefTypes
                tensor_shape, dynamic_shape, insertion_point = self._resolve_dynamic_shape(
                    n_qubits, insertion_point
                )
                measurementOp = quantum.CountsOp(
                    operands=(obs, dynamic_shape, None, None),
                    result_types=(
                        builtin.TensorType(element_type=builtin.Float64Type(), shape=tensor_shape),
                        builtin.TensorType(element_type=builtin.i64, shape=tensor_shape),
                    ),
                )

            case _:
                raise CompileError(
                    f"The measurement {type(mp).__name__} cannot be supported into the module."
                )

        self.insert_op(measurementOp, insertion_point=insertion_point)

    def insert_mid_measure(
        self, mcm: measurements.MidMeasureMP, insertion_point: InsertPoint
    ) -> None:
        """Insert a PL measurement into the IR at the provided insertion point."""
        in_qubit = self.wire_manager[mcm.wires[0]]
        midMeasureOp = quantum.MeasureOp(in_qubit=in_qubit, postselect=mcm.postselect)
        self.insert_op(midMeasureOp, insertion_point=insertion_point)
        in_qubit.replace_by_if(lambda use: use.operation != midMeasureOp)
        self.notify_op_modified(midMeasureOp)

        # If reseting, we need to insert a conditional statement that applies a PauliX
        # if we measured |1>
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
            midMeasureOp.out_qubit.replace_by_if(lambda use: use.operation != ifOp)
            self.notify_op_modified(ifOp)


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
            # pylint: disable=attribute-defined-outside-init
            rewriter.has_done_action = False
            rewriter.current_operation = op
            rewriter.insertion_point = InsertPoint.before(op)

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
