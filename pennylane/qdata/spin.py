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
Contains the spin class.
"""
import numpy as np
import networkx as nx
import quspin
from tqdm.auto import tqdm
import pennylane as qml


class SpinSystem:
    """Spin System Class"""

    def __init__(self, num_systems, type, periodicity, lattice_name, lattice, layout) -> None:
        self.num_systems = num_systems
        self.type = type
        if self.type not in ["Ising", "Heisenberg", "FermiHubbard", "BoseHubbard"]:
            raise NotImplementedError(
                "Only Ising, Heisenberg and FermiHubbard models are supported currently."
            )
        self.periodicity = periodicity
        self.lattice_name = lattice_name
        self.lattice = lattice
        self.layout = layout
        self.build_lattice_structure()
        self.num_sites = len(self.nodes.values())

    def build_lattice_structure(self):
        node_list = {node: idx for idx, node in enumerate(self.lattice.nodes.keys())}
        edge_list = []
        for nodes in self.lattice.edges.keys():
            edge_list.append((node_list[nodes[0]], node_list[nodes[1]]))
        self.nodes, self.edges = node_list, edge_list

    def getGroundStateData(self, ham):
        sp_ham = qml.utils.sparse_hamiltonian(ham)
        eigvals, eigvecs = sp.linalg.eigh(
            sp_ham.todense()
        )  # this is a slower method, outputs correct values
        psi0 = eigvecs[:, np.argmin(eigvals)]
        return min(eigvals), psi0


class IsingModel(SpinSystem):
    """Ising System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, J=1) -> None:
        super().__init__(num_systems, "Ising", periodicity, lattice, layout)
        self.J = J
        self.h = np.linspace(0, 2 * self.J, self.num_systems)
        self.params = (self.J, self.h)
        self.hamiltonian = self.build_hamiltonian()
        self.phase_labels = self.build_phaselabels()

    def build_hamiltonian(self):
        hamils = []
        self.basis = quspin.basis.spin_basis_general(len(self.nodes.values()))
        J_zz = [[self.J, j, k] for (j, k) in self.edges]
        for i in tqdm(range(self.num_systems)):
            h_z = [[self.h[i], j] for j in self.nodes.values()]
            static = [["zz", J_zz], ["x", h_z]]
            dynamic = []
            hamils.append(
                quspin.operators.hamiltonian(
                    static, dynamic, basis=self.basis, dtype=np.float64, check_symm=False
                )
            )
        return hamils

    def build_phaselabels(self):
        labels = np.array([-1 if self.h[i] < 1 else 1 for i in range(self.num_systems)])
        return labels

    def build_corr_func(self):
        pass


class HeisenbergModel(SpinSystem):
    """Heisenberg System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, J=1) -> None:
        super().__init__(num_systems, "Heisenberg", periodicity, lattice, layout)
        self.J = J
        self.delta = np.linspace(-2, 2, self.num_systems)
        self.params = (self.J, self.delta)
        self.hamiltonian = self.build_hamiltonian()
        self.phase_labels = self.build_phaselabels()

    def build_hamiltonian(self):
        hamils = []
        self.basis = quspin.basis.spin_basis_general(len(self.nodes.values()))
        J_ij = [[self.J, j, k] for (j, k) in self.edges]
        for i in tqdm(range(self.num_systems)):
            h_z = [[self.delta[i], j] for j in self.nodes.values()]
            static = [["xx", J_ij], ["yy", J_ij], ["zz", J_ij]]
            dynamic = []
            hamils.append(
                quspin.operators.hamiltonian(
                    static, dynamic, basis=self.basis, dtype=np.float64, check_symm=False
                )
            )
        return hamils

    def build_phaselabels(self):
        labels = []
        for i in range(self.num_systems):
            if self.delta[i] < -1:
                labels.append(0)
            elif self.delta[i] > 1:
                labels.append(2)
            else:
                labels.append(1)
        return labels

    def build_corr_func(self):
        pass


class FermiHubbardModel(SpinSystem):
    """FermiHubbard System Class"""

    def __init__(self, num_systems, layout, lattice, periodicity, U=1) -> None:
        super().__init__(num_systems, "FermiHubbard", periodicity, lattice, layout)
        self.U = U
        self.J = np.linspace(self.U, 2 * self.U, self.num_systems)
        self.mu = 0
        self.params = (self.U, self.J, self.mu)
        self.hamiltonian = self.build_hamiltonian()
        self.phase_labels = self.build_phaselabels()

    def build_hamiltonian(self):
        hamils = []
        self.basis = quspin.basis.spinful_fermion_basis_general(len(self.nodes))
        interaction = [[self.U, j, j] for j in self.nodes.values()]
        potential = [[-self.mu, j] for j in self.nodes.values()]
        for i in tqdm(range(self.num_systems)):
            hopping_left = [[-self.J[i], j, k] for (j, k) in self.edges] + [
                [-self.J[i], k, j] for (j, k) in self.edges
            ]
            hopping_right = [[self.J[i], k, j] for (j, k) in self.edges] + [
                [self.J[i], k, j] for (j, k) in self.edges
            ]
            static = [
                ["+-|", hopping_left],  # spin up hops to left
                ["-+|", hopping_right],  # spin up hops to right
                ["|+-", hopping_left],  # spin down hopes to left
                ["|-+", hopping_right],  # spin up hops to right
                ["n|", potential],  # onsite potenial, spin up
                ["|n", potential],  # onsite potential, spin down
                ["n|n", interaction],
            ]  # spin up-spin down interaction
            dynamic = []
            hamils.append(
                quspin.operators.hamiltonian(
                    static,
                    dynamic,
                    basis=self.basis,
                    dtype=np.float64,
                    check_symm=False,
                    check_herm=False,
                    check_pcon=False,
                )
            )
        return hamils

    def build_phaselabels(self):
        labels = []
        for i in range(self.num_systems):
            if self.U / self.J[i] < -1:
                labels.append(0)
            elif self.U / self.J[i] > 1:
                labels.append(2)
            else:
                labels.append(1)
        return labels

    def build_corr_func(self):
        pass
