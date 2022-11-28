from functools import reduce
from string import ascii_letters as ABC
from os import getenv

import numpy as np

from pennylane.tape import QuantumScript
from pennylane.operation import Tensor, Operation

from pennylane import (
    BasisState,
    QubitStateVector,
    Projector,
    Rot,
)

# Explicitly pull in the compiled module and RPATH'd dependencies before use
from .lightning_qubit_ops import StateVectorC128, MeasuresC128, adjoint_diff
from ._serialize import _serialize_observables, _serialize_ops, create_ops_list
from ..device_interface.abstract_device_driver import AbstractDeviceDriver

class LightningQubitSimulator(AbstractDeviceDriver):
    """

    Current Restrictions:
    * No batching

    * No support for state preparation yet
    * No sampling yet
    * restricted measurement types

    Preprocessing restrictions:
    * Quantum Script wires must be adjacent integers starting from zero
    * All operations must have matrices

    """

    name = "LightningQubitSimulator"

    def __init__(self,):
        self.array = None
        self.statevector = None
        self.measure_ops = None
        self.wires = None

    def execute(self, qs: QuantumScript, dtype=np.complex128):
        if not (self.array or self.statevector):
            num_indices = len(qs.wires)
            self.wires = range(num_indices)
            self.array = np.zeros(2**num_indices, dtype=dtype)
            self.array[0] = 1.0+0j
            self.statevector = StateVectorC128(self.array)

        for op in qs._ops:
            op_callable = getattr(self.statevector, op.name)
            op_callable(op.wires, op.inverse, op.parameters)

        if len(qs.measurements) == 1:
            return self.measure(qs.measurements[0])
        return tuple(self.measure(m) for m in qs.measurements)

    def measure(self, measurementprocess):
        mp_type = measurementprocess.return_type.value

        mp_map = {"probs": self.probability, "expval": self.expval, "state": lambda state, mp: self.state}
        if mp_type in mp_map:
            return np.array(mp_map[mp_type](measurementprocess))
        return state

    def expval(self, measurementprocess):
        if not self.measure_ops:
            self.measure_ops = MeasuresC128(self.statevector)

        if isinstance(measurementprocess.obs, Tensor):
            mat = measurementprocess.obs.matrix()
            return self.measure_ops.expval(mat, measurementprocess.obs.wires)

        return self.measure_ops.expval(measurementprocess.obs.name, measurementprocess.obs.wires)

    def probability(self, measurementprocess):
        return self.measure_ops.probs(self.wires)

    def gradient_prep(self, qs: QuantumScript):
        if not self.wires:
            num_indices = len(qs.wires)
            self.wires = range(num_indices)

        obs_serialized = _serialize_observables(qs, self.wires, use_csingle=False)
        ops_serialized, use_sp = _serialize_ops(qs, self.wires)
        ops_serialized = create_ops_list(*ops_serialized)

        # We need to filter out indices in trainable_params which do not
        # correspond to operators.
        trainable_params = sorted(qs.trainable_params)

        if len(trainable_params) == 0:
            return None

        tp_shift = []
        record_tp_rows = []
        all_params = 0

        for op_idx, tp in enumerate(trainable_params):
            op, _ = qs.get_operation(
                op_idx
            )  # get op_idx-th operator among differentiable operators
            if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
                # We now just ignore non-op or state preps
                tp_shift.append(tp)
                record_tp_rows.append(all_params)
            all_params += 1

        if use_sp:
            # When the first element of the tape is state preparation. Still, I am not sure
            # whether there must be only one state preparation...
            tp_shift = [i - 1 for i in tp_shift]

        output = {
            "obs_serialized" : obs_serialized, 
            "ops_serialized" : ops_serialized, 
            "tp_shift" : tp_shift,
            "record_tp_rows" : record_tp_rows,
            "all_params" : all_params
        }
        return output


    def adjoint_jacobian(self, qs: QuantumScript, use_device_state=False):
        processed_data = self.gradient_prep(qs)

        if not processed_data:  # training_params is empty
            return np.array([], dtype=self.array.dtype)

        trainable_params = processed_data["tp_shift"]

        # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        # numbers of observables, enabling choice between compute time and memory use.
        requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        jac = adjoint_diff.adjoint_jacobian(
                self.statevector,
                processed_data["obs_serialized"],
                processed_data["ops_serialized"],
                trainable_params,
        )
        jac = np.array(jac)
        jac = jac.reshape(-1, len(trainable_params))
        jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        jac_r[:, processed_data["record_tp_rows"]] = jac
        return jac_r
