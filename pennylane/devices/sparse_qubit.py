# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The default.qubit device is PennyLane's standard qubit-based device.
"""

import concurrent.futures
import logging
from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Optional, Union

import numpy as np

import pennylane as qml
from pennylane.logging import debug_logger, debug_logger_init
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.ops.op_math.condition import Conditional
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumScriptOrBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.typing import PostprocessingFn, Result, ResultBatch

from . import Device
from .execution_config import DefaultExecutionConfig, ExecutionConfig
from .modifiers import simulator_tracking, single_tape_support
from .preprocess import (
    decompose,
    mid_circuit_measurements,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_observables,
)
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_jvp, adjoint_vjp
from .qubit.sampling import jax_random_split
from .qubit.simulate import get_final_state, measure_final_state, simulate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import sparse

@simulator_tracking
@single_tape_support
class SparseQubit(Device):

    @property
    def name(self):
        """The name of the device."""
        return "sparse.qubit"

    def execute(self, circuits, execution_config=DefaultExecutionConfig):

        tape = circuits[0]

        def func(obj):
            for op_name in ["X", "CRZ", "CRX", "CRY", "Hadamard", "CNOT", "Toffoli", "PhaseShift", "PauliX", "PauliY",
                            "PauliZ", "SWAP", "RY"]:
                if op_name in obj.name:
                    return True
            return False

        tape = qml.devices.preprocess.decompose(tape, stopping_condition = func, max_expansion=5)[0][0]

        wires = tape.wires



        state = SparseState(len(wires))

        """
        shape = tuple([2 for _ in range(len(wires))])
        print(shape)

        coords = [tuple([0 for _ in range(len(wires))]),]
        print(coords)
        data = [1.0]

        coords = np.array(coords).T
        state = sparse.COO(coords, data, shape=shape)

        print("hola", state[0,0])
        """
        for gate in tape.operations:

            control_wires = gate.control_wires

            if len(control_wires) != 0:
                gate = gate.base

            wires = gate.wires

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.apply(gate)
                return qml.state()

            wires_copy = [w for w in wires].copy()
            wires_copy.sort()
            matrix = qml.matrix(circuit, wire_order=wires_copy)()

            new_basis_states = []
            already_modified = []
            for basis in state.coefs_dic.copy():

                if len(control_wires) > 0:
                    for bit in [basis[int(k)] for k in control_wires]:
                        if bit == "0":
                            break
                    if bit == "0":
                        continue

                if not basis in already_modified:

                    if not basis in new_basis_states:
                        new_basis_states.append(basis)

                    semilla = "".join(
                        [bit if len(qml.wires.Wires(i).intersection(wires)) == 0 else "_" for i, bit in
                         enumerate(basis)])
                    # la semilla es básicamente coger basis y eliminarle el bit en la posición op.wires

                    basis_to_modify = []
                    for i in range(2 ** len(wires)):

                        str_bin_i = bin(2 ** len(wires) + i)[-len(wires):]

                        semilla_aux = semilla
                        for char_bin_i in str_bin_i:
                            semilla_aux = semilla_aux.replace("_", char_bin_i, 1)

                        if not semilla_aux in state.coefs_dic.keys():
                            state.coefs_dic[semilla_aux] = 0

                        basis_to_modify.append(semilla_aux)
                        # estos son los estados que tenemos que modificar por cada basis_state

                    already_modified += basis_to_modify

                    my_array = np.array([state.coefs_dic[base] for base in basis_to_modify])
                    new_array = matrix @ my_array.T
                    for item, base in zip(new_array, basis_to_modify):
                        state.coefs_dic[base] = item

            epsilon = 1e-4  # Puedes ajustar este valor según tus necesidades

            claves_a_eliminar = [clave for clave, valor in state.coefs_dic.items() if np.abs(valor) < epsilon]

            for clave in claves_a_eliminar:
                del state.coefs_dic[clave]

        if "probs" in str(circuits[0].measurements[0]):
            for basis in state.coefs_dic:
                state.coefs_dic[basis] =  abs(state.coefs_dic[basis]) ** 2

            prob_wires = [int(i) for i in tape.measurements[0].wires]

            result = {}
            for bitstring, coef in state.coefs_dic.items():
                new_key = ''.join(bitstring[pos] for pos in prob_wires)
                if new_key in result:
                    result[new_key] += coef
                else:
                    result[new_key] = coef

            return (SparseState(len(result), result.values(), result.keys()),)

        return (state,)


    def __init__(self, wires=None, shots=None) -> None:
        super().__init__(wires=wires, shots=shots)
        self._debugger = None


class SparseState:

    def __init__(self, n_wires, coefs = None, basis_states = None, round = 4):
        self.round = round # redondeo en __repr__
        if coefs is None and  basis_states is None:
            coefs, basis_states = [1], ["0" * n_wires]

        self.n_wires = n_wires
        self.coefs_dic = {state:coef for state, coef in zip(basis_states, coefs)}

    def __repr__(self):
        return "".join([f"+ ({np.round(self.coefs_dic[state],self.round)})|{state}⟩\n" for state in self.coefs_dic])





