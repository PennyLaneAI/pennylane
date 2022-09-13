# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Contains the dataset generation pipelines.
"""

import dill
import zstd

import os

os.environ["OMP_NUM_THREADS"] = "8"

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.grouping import pauli_word_to_string, string_to_pauli_word

from pathlib import Path
import re

from tqdm.auto import tqdm

import functools as ft
import warnings

from .spin import SpinSystem, IsingModel, HeisenbergModel, FermiHubbardModel

import pennylane as qml
import numpy as np  # not pennylane numpy, otherwise the tensor type causes problems with QuSpin variable named "static"

import itertools as it
import scipy as sp

from pathlib import Path
from tqdm import tqdm

import quspin
from quspin.tools.lanczos import lanczos_full, lin_comb_Q_T


class DataPipeline:
    """Main Pipeline Class"""

    def __init__(self) -> None:
        filepath = None

    @staticmethod
    def read_data(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    @staticmethod
    def write_data(data, filepath, protocol=4):
        """Writing the data to a file"""
        pickled_data = dill.dumps(data, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as file:
            file.write(compressed_pickle)

    @staticmethod
    def write_data_seperated(data, filepath, protocol=4):
        """Writing the data to a file"""
        for (key, val) in data.values():
            pickled_data = dill.dumps(val, protocol=protocol)  # returns data as a bytes object
            compressed_pickle = zstd.compress(pickled_data)
            with open(filepath + f"_{key}.dat", "wb") as file:
                file.write(compressed_pickle)

    @staticmethod
    def append_data(filepath):
        """Append the data to a file"""
        try:
            with open(filepath, "rb") as file:
                compressed_pickle = file.read()
            depressed_pickle = zstd.decompress(compressed_pickle)
            data = dill.loads(depressed_pickle)
        except:
            data = {}
        return


class ChemDataPipeline(DataPipeline):
    """Quantum Chemistry Data Pipeline Class"""

    def __init__(self):
        self.tapered_hf_state = None
        self.opt_asd_params = None
        self.opt_ag_params = None
        self.opt_tap_params = None
        super().__init__()

    @staticmethod
    def read_xyz(path):
        """Reads a molecule from its xyz file"""
        if path.split(".")[-1] != "xyz":
            raise NotImplementedError("Currently supports only xyz files")
        with open(path, "r") as f:
            lines = f.readlines()
            num_atoms = int(lines[0])
            mol_name = lines[1].strip("\n")
            symbols = []
            geometry = []
            for line in lines[2:]:
                data = re.split(" |\t", line)  # multiple delimiters
                symbols.append(data[0])
                geometry.append(pnp.array(list(map(float, data[1:]))))
        return num_atoms, mol_name, symbols, pnp.array(geometry)

    @staticmethod
    def is_commuting_obs(ham1, ham2):
        """Check for commutivity between two observables"""
        commute = True
        for op1 in ham1.ops:
            for op2 in ham2.ops:
                if not qml.grouping.is_commuting(op1, op2):
                    commute = False
                    break
        return commute

    def run_adaptive_vqe(self, singles, doubles, hf_state, ham, classical_energy):
        """Runs VQE routine implementing AdaptiveGivens template"""

        def circuit_1(params, wires, excitations):
            qml.BasisState(hf_state, wires=wires)
            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                else:
                    qml.SingleExcitation(params[i], wires=excitation)

        def circuit_2(params, wires, excitations, gates_select, params_select):
            qml.BasisState(hf_state, wires=wires)
            for i, gate in enumerate(gates_select):
                if len(gate) == 4:
                    qml.DoubleExcitation(params_select[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params_select[i], wires=gate)
            for i, gate in enumerate(excitations):
                if len(gate) == 4:
                    qml.DoubleExcitation(params[i], wires=gate)
                elif len(gate) == 2:
                    qml.SingleExcitation(params[i], wires=gate)

        excitations = singles + doubles
        qubits = len(ham.wires)

        dev = qml.device("lightning.qubit", wires=ham.wires, batch_obs=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def cost_fn_1(param, excitations):
            circuit_1(param, wires=ham.wires, excitations=excitations)
            return qml.expval(ham)

        circuit_gradient = qml.grad(cost_fn_1, argnum=0)
        params = pnp.array([0.0] * len(doubles))
        grads = circuit_gradient(params, excitations=doubles)

        doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
        doubles_indice = [i for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]

        opt = qml.AdamOptimizer(stepsize=0.01)
        params_doubles = pnp.zeros(len(doubles_select), requires_grad=True)
        params_doubles, prev_energy = opt.step_and_cost(
            cost_fn_1, params_doubles, excitations=doubles_select
        )
        for n in tqdm(range(100), leave=False):
            params_doubles, energy = opt.step_and_cost(
                cost_fn_1, params_doubles, excitations=doubles_select
            )
            if pnp.abs(energy - prev_energy) <= 1e-6:
                break

        @qml.qnode(dev, diff_method="parameter-shift")
        def cost_fn_2(param, excitations, gates_select, params_select):
            circuit_2(
                param,
                wires=ham.wires,
                excitations=excitations,
                gates_select=gates_select,
                params_select=params_select,
            )
            return qml.expval(ham)

        circuit_gradient = qml.grad(cost_fn_2, argnum=0)
        params = pnp.array([0.0] * len(singles))
        grads = circuit_gradient(
            params, excitations=singles, gates_select=doubles_select, params_select=params_doubles
        )

        singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
        singles_indice = [len(doubles) + i for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]

        gates_select = doubles_select + singles_select
        if self.opt_ag_params is None:
            params = pnp.zeros(len(gates_select), requires_grad=True)
        else:
            params = self.opt_ag_params.copy()

        opt = qml.AdamOptimizer(stepsize=0.01)
        params, prev_energy = opt.step_and_cost(cost_fn_1, params, excitations=gates_select)
        for n in (pbar := tqdm(range(1000), position=0, leave=True)):
            params, energy = opt.step_and_cost(cost_fn_1, params, excitations=gates_select)
            chem_acc = pnp.round(pnp.abs(classical_energy - energy), 5)
            if pnp.abs(chem_acc) <= 1e-3 or pnp.abs(energy - prev_energy) <= 1e-6:
                break
            pbar.set_description(f"AdaptGiv: Chem. Acc.  = {chem_acc} Hartree\t")
        self.opt_ag_params = params.copy()
        circuit = ft.partial(circuit_1, excitations=gates_select)

        all_params = pnp.zeros(len(doubles + singles), requires_grad=True)
        for idx, exc_itx in enumerate(doubles_indice + singles_indice):
            all_params[exc_itx] = params[idx]
        self.opt_asd_params = all_params.copy()

        return params, circuit, energy

    def run_tapered_vqe(
        self, singles, doubles, tapering_data, tapered_hf_state, tapered_ham, classical_energy
    ):
        """Runs VQE routine implementing taperedVQE template"""
        global opt_tap_params

        generators, paulix_ops, opt_sector = tapering_data
        singles_tapered, doubles_tapered = [], []
        for excitation in singles:
            op_coeffs, op_terms = qml.qchem.observable_hf.jordan_wigner(excitation)
            excitation_op = qml.Hamiltonian(op_coeffs, op_terms)
            excitation_tapered_op = qml.taper(excitation_op, generators, paulix_ops, opt_sector)
            singles_tapered.append(excitation_tapered_op)

        for excitation in doubles:
            op_coeffs, op_terms = qml.qchem.observable_hf.jordan_wigner(excitation)
            excitation_op = qml.Hamiltonian(op_coeffs, op_terms)
            excitation_tapered_op = qml.Hamiltonian(
                excitation_tapered_op.coeffs, excitation_tapered_op.ops
            )
            excitation_tapered_op = qml.taper(excitation_op, generators, paulix_ops, opt_sector)
            doubles_tapered.append(excitation_tapered_op)

        commuting_excitations = []
        for excitation in doubles_tapered + singles_tapered:
            commute = True
            for generator in generators:
                if not ChemDataPipeline.is_commuting_obs(generator, excitation):
                    commute = False
                    break
            if commute:
                commuting_excitations.append(excitation)
        if not commuting_excitations:
            warnings.warn(
                "Tapered excitations doesn't commute, results might suffer", RuntimeWarning
            )
            commuting_excitations = doubles_tapered + singles_tapered

        dev = qml.device("lightning.qubit", wires=tapered_ham.wires, batch_obs=True)

        def circuit(params, wires, excitations):
            qml.BasisState(tapered_hf_state, wires=tapered_ham.wires)
            for idx, excitation in enumerate(excitations):
                for itx, op in enumerate(excitation.ops):
                    if op.label() != "I":
                        qml.PauliRot(
                            params[idx][itx], qml.grouping.pauli_word_to_string(op), op.wires
                        )

        @qml.qnode(dev)
        def cost_fn(param, excitations):
            circuit(param, wires=tapered_ham.wires, excitations=commuting_excitations)
            return qml.expval(tapered_ham)

        circuit_gradient = qml.grad(cost_fn, argnum=0)
        if opt_tap_params is None:
            param_len = sum([len(excitation.ops) for excitation in commuting_excitations])
            params = pnp.random.rand(len(commuting_excitations), param_len, requires_grad=True)
        else:
            params = opt_tap_params.copy()
        grads = circuit_gradient(params, excitations=commuting_excitations)

        opt = qml.AdamOptimizer(stepsize=0.01)
        params, prev_energy = opt.step_and_cost(cost_fn, params, excitations=commuting_excitations)
        for n in (pbar := tqdm(range(1000), position=0, leave=True)):
            params, energy = opt.step_and_cost(cost_fn, params, excitations=commuting_excitations)
            chem_acc = pnp.round(pnp.abs(classical_energy - energy), 5)
            if pnp.abs(chem_acc) <= 1e-3 or pnp.abs(energy - prev_energy) <= 1e-6:
                break
            pbar.set_description(f"TapExc: Chem. Acc. = {chem_acc} Hartree\t")
        circuit = ft.partial(circuit, excitations=commuting_excitations)
        opt_tap_params = params.copy()
        return params, circuit, energy

    @staticmethod
    def convert_ham_obs(hamiltonian, wire_map):
        hamil_dict = {}
        coeffs, ops = hamiltonian.terms()
        for coeff, op in zip(coeffs, ops):
            hamil_dict.update({qml.grouping.pauli_word_to_string(op, wire_map): coeff})
        return hamil_dict

    @staticmethod
    def convert_ham_dict(hamil_dict, wire_map):
        coeffs, ops = [], []
        for key, val in hamil_dict.items():
            coeffs.append(val)
            ops.append(qml.grouping.string_to_pauli_word(key, wire_map))
        return qml.Hamiltonian(coeffs, ops)

    def pipeline(
        self,
        molname,
        symbols,
        geometry,
        charge,
        basis_name,
        descriptor,
        update_keys=[],
        skip_keys=[],
        filename="",
        prog_bar=None,
    ):
        """Implements the data generation pipeline"""

        if prog_bar is None:
            raise ValueError("Please initialize progress bar for verbose output")
        prog_bar.set_description(f"Molecule Generation")
        mol = qml.qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)
        path = f"data/qchem/{molname}/{mol.basis_name}/{geometry}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = f"{path}{molname}_{mol.basis_name}_{geometry}_full.dat"
        print(filename)
        f = self.append_data(filename)
        present_keys = list(f.keys())
        f["molecule"] = mol
        wire_map = None

        if "hamiltonian" not in skip_keys and (
            "hamiltonian" not in present_keys or "hamiltonian" in update_keys
        ):
            prog_bar.set_description(f"Hamiltonian Generation")
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
                symbols, geometry, charge=charge, basis=basis_name
            )
            sparse_ham = qml.utils.sparse_hamiltonian(hamiltonian)
            sparse_hamiltonian = qml.SparseHamiltonian(sparse_ham, hamiltonian.wires)
            # classical_energy = qml.eigvals(sparse_hamiltonian)[0]
            wire_map = {wire: idx for idx, wire in enumerate(hamiltonian.wires)}
            f["hamiltonian"] = hamiltonian  # self.convert_ham_obs(hamiltonian, wire_map)
            f["ham_wire_map"] = wire_map
            # f["fci_energy"] = classical_energy
            f["sparse_hamiltonian"] = sparse_ham  # iltonian
            f["meas_groupings"] = qml.grouping.optimize_measurements(
                hamiltonian.ops, hamiltonian.coeffs
            )
            self.write_data(f, filename)
        else:
            wire_map = f["ham_wire_map"]
            hamiltonian = self.convert_ham_dict(f["hamiltonian"], wire_map)
            # classical_energy = f["fci_energy"]
            sparse_hamiltonian = qml.SparseHamiltonian(f["sparse_hamiltonian"], hamiltonian.wires)

        if "symmetries" not in skip_keys and (
            "symmetries" not in present_keys or "symmetries" in update_keys
        ):
            prog_bar.set_description(f"Symmetry Generation")
            hamiltonian = self.convert_ham_dict(f["hamiltonian"], wire_map)
            generators = qml.symmetry_generators(hamiltonian)
            paulixops = qml.paulix_ops(generators, len(hamiltonian.wires))
            paulix_sector = qml.qchem.optimal_sector(hamiltonian, generators, mol.n_electrons)
            f["symmetries"] = generators
            f["paulix_ops"] = paulixops
            f["optimal_sector"] = paulix_sector
            self.write_data(f, filename)
        else:
            generators = f["symmetries"]
            paulixops = f["paulix_ops"]
            paulix_sector = f["optimal_sector"]

        if "hamiltonian" not in skip_keys and (
            "hamiltonian" not in present_keys or "hamiltonian" in update_keys
        ):
            prog_bar.set_description(f"AuxillaryObs Generation")
            observables = [
                hamiltonian,
                qml.qchem.dipole_moment(mol)()[2],
                qml.qchem.particle_number(len(hamiltonian.wires)),
                qml.qchem.spin2(mol.n_electrons, len(hamiltonian.wires)),
                qml.qchem.spinz(len(hamiltonian.wires)),
            ]
            hf_state = pnp.where(pnp.arange(len(hamiltonian.wires)) < mol.n_electrons, 1, 0)
            f["num_op"] = observables[1]
            f["dipole_op"] = observables[2]
            f["spin2_op"] = observables[3]
            f["spinz_op"] = observables[4]
            f["hf_state"] = hf_state
            if charge:
                (eigvals, eigvecs), index = pnp.linalg.eigh(sparse_ham.todense()), 1
                for index in range(1, len(eigvals)):
                    eigvec = eigvecs[:, index]
                    num_part = pnp.conjugate(eigvec.T) @ qml.matrix(f["num_op"]) @ eigvec
                    if int(pnp.round(num_part[0][0].real)) == mol.n_electrons:
                        break
                classical_energy = eigvals[index]
            else:
                classical_energy = qml.eigvals(sparse_hamiltonian)[0]
            f["fci_energy"] = classical_energy
            self.write_data(f, filename)
        else:
            observables = [hamiltonian, f["dipole_op"], f["num_op"], f["spin2_op"], f["spinz_op"]]
            hf_state = f["hf_state"]
            classical_energy = f["fci_energy"]
        print(classical_energy, f["fci_energy"])
        singles, doubles = qml.qchem.excitations(mol.n_electrons, len(hamiltonian.wires))

        if "symmetries" not in skip_keys and (
            "symmetries" not in present_keys or "symmetries" in update_keys
        ):
            prog_bar.set_description(f"TaperingObs Generation")
            if self.tapered_hf_state is None:
                self.tapered_hf_state = qml.qchem.taper_hf(
                    generators, paulixops, paulix_sector, mol.n_electrons, len(hamiltonian.wires)
                )
            tapered_obs = [
                qml.taper(observable, generators, paulixops, paulix_sector)
                for observable in observables
            ]
            coeffs, ops = tapered_obs[0].terms()
            wire_map = {wire: itx for itx, wire in enumerate(tapered_obs[0].wires.tolist())}
            ops = [string_to_pauli_word(pauli_word_to_string(op, wire_map=wire_map)) for op in ops]
            tapered_obs[0] = qml.Hamiltonian(coeffs=coeffs, observables=ops)

            f["tapered_hamiltonian"] = tapered_obs[0]
            f["tapered_dipole_op"] = tapered_obs[1]
            f["tapered_num_op"] = tapered_obs[2]
            f["tapered_spin2_op"] = tapered_obs[3]
            f["tapered_spinz_op"] = tapered_obs[4]
            f["tapered_hf_state"] = self.tapered_hf_state
            self.write_data(f, filename)
        else:
            tapered_obs = [
                f["tapered_hamiltonian"],
                f["tapered_dipole_op"],
                f["tapered_num_op"],
                f["tapered_spin2_op"],
                f["tapered_spinz_op"],
            ]
            self.tapered_hf_state = f["tapered_hf_state"]

        prog_bar.set_description(f"VQE Execution")
        if "vqe_exec" not in skip_keys and (
            "vqe_exec" not in present_keys or "vqe_exec" in update_keys
        ):
            params, circuit, energy = self.run_adaptive_vqe(
                singles, doubles, hf_state, sparse_hamiltonian, classical_energy
            )
            # f["adaptive_givens"] = True
            f["vqe_energy"] = energy
            f["vqe_params"] = params
            f["vqe_circuit"] = circuit
        elif "vqe_exec" in skip_keys:
            # f["adaptive_givens"] = False
            f["vqe_energy"] = None
            f["vqe_params"] = None
            f["vqe_circuit"] = None

        # if 'excitations_tapered' not in skip_keys and ('excitations_tapered' not in present_keys or 'excitations_tapered' in update_keys):
        #     sparse_ham = qml.utils.sparse_hamiltonian(tapered_obs[0])
        #     sparse_hamiltonian = qml.SparseHamiltonian(sparse_ham, tapered_obs[0].wires)
        #     params, circuit, energy = self.run_tapered_vqe(singles, doubles,
        #                                     (generators, paulixops, paulix_sector),
        #                                     self.tapered_hf_state, sparse_hamiltonian,
        #                                     classical_energy)
        #     f["excitations_tapered"] = True
        #     f["excitations_tapered_params"] = params
        #     f["excitations_tapered_circuit"] = circuit
        #     f["excitations_tapered_energy"] = energy
        # elif 'excitations_tapered' in skip_keys:
        #     f["excitations_tapered"] = False
        #     f["excitations_tapered_params"] = None
        #     f["excitations_tapered_circuit"] = None
        #     f["excitations_tapered_energy"] = None
        prog_bar.set_description("")
        dataset = qml.qdata.ChemDataset(
            molecule=f["molecule"],
            hamiltonian=f["hamiltonian"],
            meas_groupings=f["meas_groupings"],
            symmetries=f["symmetries"],
            paulix_ops=f["paulix_ops"],
            optimal_sector=f["optimal_sector"],
            dipole_op=f["dipole_op"],
            num_op=f["num_op"],
            spin2_op=f["spin2_op"],
            spinz_op=f["spinz_op"],
            tapered_hamiltonian=f["tapered_hamiltonian"],
            tapered_dipole_op=f["tapered_dipole_op"],
            tapered_num_op=f["tapered_num_op"],
            tapered_spin2_op=f["tapered_spin2_op"],
            tapered_spinz_op=f["tapered_spinz_op"],
            vqe_params=f["vqe_params"],
            vqe_circuit=f["vqe_circuit"],
            vqe_energy=f["vqe_energy"],
            fci_energy=f["fci_energy"],
        )
        self.write_data(dataset, filename)
        self.write_data_seperated(f, f"{path}{molname}_{mol.basis_name}_{geometry}")


class SpinDataPipeline(DataPipeline):
    """Quantum Spins Data Pipeline Class"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def getGroundStateData(ham):
        # in a next iteration this function should estimate the requirements to evaluate
        # the hamiltonian and then choose an appropriate method
        # E,V=ham.eigh()
        # return np.min(E),V[np.argmin(E)]
        v0 = np.random.normal(0, 1, size=ham.Ns)
        v0 /= np.linalg.norm(v0)
        m_GS = 10  # Krylov subspace dimension
        # Lanczos finds the largest-magnitude eigenvalues:
        E, V, Q_T = lanczos_full(ham, v0, m_GS, full_ortho=True)
        psi_GS_lanczos = lin_comb_Q_T(V[:, 0], Q_T)
        psi_GS_lanczos /= np.linalg.norm(psi_GS_lanczos)

        return np.min(E), psi_GS_lanczos

    @staticmethod
    def corr_function(i, j):
        ops = []
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            if i != j:
                ops.append(op(i) @ op(j))
            else:
                ops.append(qml.Identity(i))
        return ops

    @staticmethod
    def circuit(psi, observables):
        psi = psi / np.linalg.norm(psi)  # normalize the state
        qml.QubitStateVector(psi, wires=range(int(np.log2(len(psi)))))
        return [qml.expval(o) for o in observables]

    @staticmethod
    def build_exact_corrmat(coups, corrs, circuit, psi, num_qubits):
        corr_mat_exact = np.zeros((num_qubits, num_qubits))
        for idx, (i, j) in enumerate(coups):
            corr = corrs[idx]
            if i == j:
                corr_mat_exact[i][j] = 1.0
            else:
                corr_mat_exact[i][j] = (
                    np.sum(np.array([circuit(psi, observables=[o]) for o in corr]).T) / 3
                )
                corr_mat_exact[j][i] = corr_mat_exact[i][j]
        return corr_mat_exact

    @staticmethod
    def gen_class_shadow(circ_template, circuit_params, num_shadows, num_qubits):
        # prepare the complete set of available Pauli operators
        unitary_ops = [qml.PauliX, qml.PauliY, qml.PauliZ]
        # sample random Pauli measurements uniformly
        unitary_ensmb = np.random.randint(0, 3, size=(num_shadows, num_qubits), dtype=int)

        outcomes = np.zeros((num_shadows, num_qubits))
        for ns in range(num_shadows):
            # for each snapshot, extract the Pauli basis measurement to be performed
            meas_obs = [unitary_ops[unitary_ensmb[ns, i]](i) for i in range(num_qubits)]
            # perform single shot randomized Pauli measuremnt for each qubit
            outcomes[ns, :] = circ_template(circuit_params, observables=meas_obs)

        return outcomes, unitary_ensmb

    def pipeline(self, sysname, spinsys, filename=""):
        """Implements the data generation pipeline"""

        path = (
            f"data/qspin/{sysname}/{spinsys.periodicity}/{spinsys.lattice_name}/{spinsys.layout}/"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

        if filename:
            filepath = filename
        else:
            filepath = f"{path}{sysname}_{spinsys.periodicity}_{spinsys.lattice_name}_{spinsys.layout}_full.dat"

        f = {}

        f["hamiltonians"] = spinsys.build_hamiltonian()
        f["parameters"] = spinsys.params
        f["phase_labels"] = spinsys.build_phaselabels()
        f["order_parameters"] = {}

        print(f"GroundState Calculation")
        groundstate_data = np.array(
            [
                list(self.getGroundStateData(f["hamiltonians"][i]))
                for i in range(spinsys.num_systems)
            ],
            dtype=object,
        )
        f["ground_energies"] = groundstate_data[:, 0].astype("complex128")
        f["ground_states"] = groundstate_data[:, 1]

        if type != "Fermi-Hubbard" and type != "fermi-hubbard":
            dev_exact = qml.device("default.qubit", wires=spinsys.num_sites)  # for exact simulation
        else:
            dev_exact = qml.device("default.qubit", wires=2 * spinsys.num_sites)
        circuit_exact = qml.QNode(self.circuit, dev_exact)

        coups = list(it.product(range(spinsys.num_sites), repeat=2))
        corrs = [self.corr_function(i, j) for i, j in coups]

        # expval_exact = []
        # print(f"CorrelationMat Calculation") #double-check this section
        # for x in tqdm(groundstate_data[:, 1]):
        #     expval_exact.append(self.build_exact_corrmat(coups, corrs, circuit_exact, x, spinsys.num_sites))# for x in data
        # f['correlation_matrix'] = expval_exact

        if spinsys.type != "FermiHubbard":
            dev_oshot = qml.device("default.qubit", wires=spinsys.num_sites, shots=1)
        else:
            print("hello")
            dev_oshot = qml.device("default.qubit", wires=2 * spinsys.num_sites, shots=1)
        circuit_oshot = qml.QNode(self.circuit, dev_oshot)

        shadows = []
        print(f"ClassShadow Calculation")
        for x in tqdm(groundstate_data[:, 1]):
            shadows.append(
                self.gen_class_shadow(circuit_oshot, x, 100, spinsys.num_sites)
            )  # for x in data
        f["classical_shadows"] = shadows

        self.write_data(f, filepath)
        dataset = qml.qdata.ChemDataset(
            spin_system=f["spin_system"],
            hamiltonian=f["hamiltonian"],
            parameters=f["parameters"],
            phase_labels=f["phase_labels"],
            order_parameters=f["order_parameters"],
            ground_energies=f["ground_energies"],
            ground_states=f["ground_states"],
            classical_shadows=f["classical_shadows"],
        )
        self.write_data(dataset, filename)
        self.write_data_seperated(
            f, f"{path}{sysname}_{spinsys.periodicity}_{spinsys.lattice_name}_{spinsys.layout}"
        )
