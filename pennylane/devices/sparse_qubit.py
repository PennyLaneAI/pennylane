import numpy as np
import pennylane as qml
from pennylane.devices import Device
from pennylane.devices.modifiers import simulator_tracking, single_tape_support


@simulator_tracking
@single_tape_support
class SparseQubit(Device):

    @property
    def name(self):
        return "sparse.qubit"

    def execute(self, circuits, execution_config=None):

        tape = circuits[0]

        def func(obj):
            for op_name in [
                "X",
                "CRZ",
                "CRX",
                "CRY",
                "Hadamard",
                "CNOT",
                "Toffoli",
                "PhaseShift",
                "PauliX",
                "PauliY",
                "PauliZ",
                "SWAP",
                "RY",
            ]:
                if op_name in obj.name:
                    return True
            return False

        tape = qml.devices.preprocess.decompose(tape, stopping_condition=func)[0][0]

        # Fixed ordering of wires -> position in the bitstring.
        all_wires = tape.wires
        wire_to_pos = {w: i for i, w in enumerate(all_wires)}

        state = SparseState(len(all_wires))

        # Relative pruning threshold: drop terms whose probability is negligible
        # compared to the total norm, instead of a fixed absolute cutoff.
        rel_epsilon = 1e-12

        for gate in tape.operations:

            # Apply every gate as a dense matrix over ALL of its wires (controls
            # included). The previous "peel the controls and skip |0> branches"
            # optimization was only correct for a single, cleanly-exposed control
            # layer; it silently produced wrong results for adjoint-wrapped
            # controlled ops (e.g. Adjoint(MultiControlledX)) and for control
            # values other than all-ones. Building the full matrix over the whole
            # wire set is unconditionally correct and still sparse in the state.
            wires = gate.wires

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.apply(gate)
                return qml.state()

            # Positions in the bitstring occupied by this gate's wires.
            target_positions = sorted(wire_to_pos[w] for w in wires)

            # The matrix MUST be built in the SAME order in which the bitstring
            # "_" slots are filled below, i.e. by ascending bitstring position.
            # Sorting by wire *label* instead (the old behaviour) silently swaps
            # control/target whenever a gate's wires are not already in ascending
            # position order (e.g. CNOT([1, 0])).
            ordered_wires = [all_wires[p] for p in target_positions]
            matrix = qml.matrix(circuit, wire_order=ordered_wires)()

            target_positions = set(target_positions)

            already_modified = []
            for basis in state.coefs_dic.copy():

                if basis in already_modified:
                    continue

                # Build a template with "_" at the target-wire positions.
                semilla = "".join(
                    "_" if pos in target_positions else bit for pos, bit in enumerate(basis)
                )

                basis_to_modify = []
                for i in range(2 ** len(wires)):
                    str_bin_i = bin(2 ** len(wires) + i)[-len(wires) :]

                    semilla_aux = semilla
                    for char_bin_i in str_bin_i:
                        semilla_aux = semilla_aux.replace("_", char_bin_i, 1)

                    if semilla_aux not in state.coefs_dic:
                        state.coefs_dic[semilla_aux] = 0

                    basis_to_modify.append(semilla_aux)

                already_modified += basis_to_modify

                my_array = np.array([state.coefs_dic[base] for base in basis_to_modify])
                new_array = matrix @ my_array.T
                for item, base in zip(new_array, basis_to_modify):
                    state.coefs_dic[base] = item

            # Relative pruning: threshold scales with the state's norm.
            total_norm_sq = sum(np.abs(v) ** 2 for v in state.coefs_dic.values())
            cutoff = rel_epsilon * total_norm_sq
            claves_a_eliminar = [k for k, v in state.coefs_dic.items() if np.abs(v) ** 2 < cutoff]
            for clave in claves_a_eliminar:
                del state.coefs_dic[clave]

        if isinstance(tape.measurements[0], qml.measurements.ProbabilityMP):
            for basis in state.coefs_dic:
                state.coefs_dic[basis] = abs(state.coefs_dic[basis]) ** 2

            prob_positions = [wire_to_pos[w] for w in tape.measurements[0].wires]

            result = {}
            for bitstring, coef in state.coefs_dic.items():
                new_key = "".join(bitstring[pos] for pos in prob_positions)
                result[new_key] = result.get(new_key, 0) + coef

            return (SparseState(len(result), result.values(), result.keys()),)

        return (state,)

    def __init__(self, wires=None, shots=None) -> None:
        super().__init__(wires=wires, shots=shots)
        self._debugger = None


class SparseState:

    def __init__(self, n_wires, coefs=None, basis_states=None, round=4):
        self.round = round
        if coefs is None and basis_states is None:
            coefs, basis_states = [1], ["0" * n_wires]

        self.n_wires = n_wires
        self.coefs_dic = {state: coef for state, coef in zip(basis_states, coefs)}

    def __repr__(self):
        return "".join(
            [
                f"+ ({np.round(self.coefs_dic[state],self.round)})|{state}⟩\n"
                for state in self.coefs_dic
            ]
        )
