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
This module contains the functions needed for computing the dipole moment.
"""

import pennylane as qp

from .openfermion_pyscf import _import_pyscf


def molecular_dipole(  # pylint: disable=too-many-arguments
    molecule,
    method="openfermion",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    cutoff=1.0e-16,
):
    r"""Generate the dipole moment operator for a molecule in the Pauli basis.

    The dipole operator in the second-quantized form is

    .. math::

        \hat{D} = -\sum_{pq} d_{pq} [\hat{c}_{p\uparrow}^\dagger \hat{c}_{q\uparrow} +
        \hat{c}_{p\downarrow}^\dagger \hat{c}_{q\downarrow}] -
        \hat{D}_\mathrm{c} + \hat{D}_\mathrm{n},

    where the matrix elements :math:`d_{pq}` are given by the integral of the position operator
    :math:`\hat{{\bf r}}` over molecular orbitals :math:`\phi`

    .. math::

        d_{pq} = \int \phi_p^*(r) \hat{{\bf r}} \phi_q(r) dr,

    and :math:`\hat{c}^{\dagger}` and :math:`\hat{c}` are the creation and annihilation operators,
    respectively. The contribution of the core orbitals and nuclei are denoted by
    :math:`\hat{D}_\mathrm{c}` and :math:`\hat{D}_\mathrm{n}`, respectively, which are computed as

    .. math::
        \hat{D}_\mathrm{c} = 2 \sum_{i=1}^{N_\mathrm{core}} d_{ii} \quad \text{and} \quad
        \hat{D}_\mathrm{n} = \sum_{i=1}^{N_\mathrm{atoms}} Z_i {\bf R}_i,

    where :math:`Z_i` and :math:`{\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    The fermionic dipole operator is then transformed to the qubit basis, which gives

    .. math::

        \hat{D} = \sum_{j} c_j P_j,

    where :math:`c_j` is a numerical coefficient and :math:`P_j` is a tensor product of
    single-qubit Pauli operators :math:`X, Y, Z, I`. The qubit observables corresponding
    to the components :math:`\hat{D}_x`, :math:`\hat{D}_y`, and :math:`\hat{D}_z` of the
    dipole operator are then computed separately.

    Args:
        molecule (~qchem.molecule.Molecule): The molecule object
        method (str): Quantum chemistry method used to solve the
            mean field electronic structure problem. The ``"openfermion"`` backend uses the
            OpenFermion-PySCF plugin and requires ``openfermionpyscf`` to be installed.
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): Transformation used to map the fermionic Hamiltonian to the qubit Hamiltonian.
            Input values can be ``'jordan_wigner'``, ``'parity'`` or ``'bravyi_kitaev'``.
        outpath (str): Path to the directory containing output files
        wires (Wires or Sequence[int] or dict[int, int]): Custom wire mapping used to convert the
            qubit operator to
            an observable measurable in a Pennylane ansatz.
            For types ``Wires``/``list``/``tuple``, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.
        cutoff (float): Cutoff value for including the matrix elements
            :math:`\langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle`. The matrix elements
            with absolute value less than ``cutoff`` are neglected.

    Returns:
        list[pennylane.Hamiltonian]: The qubit observables corresponding to the components
        :math:`\hat{D}_x`, :math:`\hat{D}_y` and :math:`\hat{D}_z` of the dipole operator.

    **Example**

    >>> symbols = ["H", "H", "H"]
    >>> coordinates = np.array([[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]])
    >>> mol = qp.qchem.Molecule(symbols, coordinates, charge=1)
    >>> dipole_obs = qp.qchem.molecular_dipole(mol, method="openfermion")
    >>> np.round(qp.matrix(dipole_obs[0], wire_order=range(6))[0, 0].real, 6)
    np.float64(2.869)
    """

    method = method.strip().lower()
    if method == "dhf":
        raise ValueError(
            "The differentiable Hartree-Fock solver (method='dhf') has been removed. "
            "Use method='openfermion' (the default)."
        )
    if method != "openfermion":
        raise ValueError("Only the 'openfermion' backend is supported.")

    mapping = mapping.strip().lower()
    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"'{mapping}' is not supported."
            f"Please set the mapping to 'jordan_wigner', 'parity' or 'bravyi_kitaev'."
        )

    symbols = molecule.symbols
    coordinates = molecule.coordinates

    if qp.math.shape(coordinates)[0] == len(symbols) * 3:
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_hf = coordinates.flatten()
    else:
        raise ValueError(
            "The shape of the coordinates does not match the number of atoms in the molecule."
        )

    if molecule.mult != 1:
        raise ValueError(
            "Open-shell systems are not supported. Change the charge or spin multiplicity of the molecule."
        )

    pyscf = _import_pyscf()
    mol = pyscf.gto.M(
        atom=list(zip(symbols, geometry_hf.reshape(-1, 3), strict=True)),
        unit="Bohr",
        basis=molecule.basis_name,
        charge=molecule.charge,
        spin=molecule.mult - 1,
        verbose=0,
    )
    core, active = qp.qchem.active_space(
        mol.nelectron, mol.nao, molecule.mult, active_electrons, active_orbitals
    )

    dip = qp.qchem.dipole_of(
        symbols,
        geometry_hf,
        molecule.name,
        molecule.charge,
        molecule.mult,
        molecule.basis_name,
        package="pyscf",
        core=core,
        active=active,
        mapping=mapping,
        cutoff=cutoff,
        outpath=outpath,
        wires=wires,
    )

    return dip
