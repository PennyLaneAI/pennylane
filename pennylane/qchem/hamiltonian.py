# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for computing the molecular Hamiltonian.
"""

from functools import singledispatch

from scipy.constants import angstrom, physical_constants

import pennylane as qp

from .atomic_numbers import atomic_numbers
from .molecule import Molecule

BOHR_TO_ANG = physical_constants["Bohr radius"][0] / angstrom


def molecular_hamiltonian(*args, **kwargs):
    """molecular_hamiltonian(molecule, method="pyscf", active_electrons=None, active_orbitals=None,\
    mapping="jordan_wigner", outpath=".", wires=None, convert_tol=1e12)
    Generate the qubit Hamiltonian of a molecule.

    This function drives the construction of the second-quantized electronic Hamiltonian
    of a molecule and its transformation to the basis of Pauli matrices.

    The net charge of the molecule can be given to simulate cationic/anionic systems. Also, the
    spin multiplicity can be input to determine the number of unpaired electrons occupying the HF
    orbitals as illustrated in the left panel of the figure below.

    The basis of Gaussian-type *atomic* orbitals used to represent the *molecular* orbitals can be
    specified to go beyond the minimum basis approximation.

    An active space can be defined for a given number of *active electrons* occupying a reduced set
    of *active orbitals* as sketched in the right panel of the figure below.

    |

    .. figure:: ../../_static/qchem/fig_mult_active_space.png
        :align: center
        :width: 90%

    |

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        method (str): Quantum chemistry method used to solve the
            mean field electronic structure problem. Available options are ``method="pyscf"``
            to use the PySCF package (requires ``pyscf`` to be installed), or ``method="openfermion"`` to
            use the OpenFermion-PySCF plugin (this requires ``openfermionpyscf`` to be installed).
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): transformation used to map the fermionic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
        wires (Wires or Sequence[int] or dict[int, int]): Custom wire mapping for connecting to
            Pennylane ansatz.
            For types ``Wires``/``list``/``tuple``, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.
        convert_tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the Hamiltonian coefficients created by openfermion.
            Coefficients with imaginary part less than 2.22e-16*tol are considered to be real.


    Returns:
        tuple[pennylane.Operator, int]: the fermionic-to-qubit transformed  Hamiltonian
        and the number of qubits

    .. note::
        The ``molecular_hamiltonian`` function accepts a ``Molecule`` object as its first argument.
        Look at the `Usage Details` for more details on the old interface.

        The ``molecular_hamiltonian`` function is not currently compatible with :func:`~.qjit` and ``jax.jit``.

    **Example**

    >>> symbols = ['H', 'H']
    >>> coordinates = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    >>> molecule = qp.qchem.Molecule(symbols, coordinates)
    >>> H, qubits = qp.qchem.molecular_hamiltonian(molecule)
    >>> print(qubits)
    4

    .. details::
        :title: Usage Details

        The old interface for this method involved passing molecular information as separate arguments:

            ``molecular_hamiltonian``\\ (`symbols, coordinates, name='molecule', charge=0, mult=1, basis='sto-3g',`
            `method='pyscf', active_electrons=None, active_orbitals=None, mapping='jordan_wigner', outpath='.',`
            `wires=None, convert_tol=1e12`)

        Molecule-based Arguments:
          - **symbols** (list[str]): symbols of the atomic species in the molecule
          - **coordinates** (array[float]): atomic positions in Cartesian coordinates.
            The atomic coordinates must be in atomic units and can be given as either a 1D array of
            size ``3*N``, or a 2D array of shape ``(N, 3)`` where ``N`` is the number of atoms.
            name (str): name of the molecule
          - **charge** (int): Net charge of the molecule. If not specified a neutral system is assumed.
          - **mult** (int): Spin multiplicity :math:`\\mathrm{mult}=N_\\mathrm{unpaired} + 1` for :math:`N_\\mathrm{unpaired}`
            unpaired electrons occupying the HF orbitals. Possible values of ``mult`` are :math:`1, 2, 3, \\ldots`.
            If not specified, a closed-shell HF state is assumed.
          - **basis** (str): atomic basis set used to represent the molecular orbitals

        Therefore, a molecular Hamiltonian had to be constructed in the following manner:

        .. code-block:: python

            from pennylane import qchem

            symbols = ["H", "H"]
            geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

            H, qubit = qchem.molecular_hamiltonian(symbols, geometry, charge=0)

        As part of the new interface, we are shifting towards extracting all the molecular information
        from the :class:`~.qchem.molecule.Molecule` within the ``molecular_hamiltonian`` method.

    """

    if len(args) != 0:
        return _molecular_hamiltonian_dispatch(*args, **kwargs)

    method = kwargs.pop("symbols", None) or kwargs.pop("molecule", None)
    if method is not None:
        return _molecular_hamiltonian_dispatch(method, **kwargs)

    raise NotImplementedError(
        "The provided arguments do not contain information about symbols in the molecule. "
        "Please provide that information in the form of a molecule object or as a list of symbols."
    )


@singledispatch
def _molecular_hamiltonian_dispatch(*args, **kwargs):
    r"""Generate the qubit Hamiltonian of a molecule."""
    raise NotImplementedError(
        "molecular_hamiltonian supports only list or molecule object types. "
        "Please provide one of them."
    )


@_molecular_hamiltonian_dispatch.register(Molecule)
def _(  # pylint: disable=too-many-arguments
    molecule,
    method="pyscf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    convert_tol=1e12,
):
    return _molecular_hamiltonian(
        molecule.symbols,
        molecule.coordinates,
        molecule.name,
        molecule.charge,
        molecule.mult,
        molecule.basis_name,
        method,
        active_electrons,
        active_orbitals,
        mapping,
        outpath,
        wires,
        convert_tol,
    )


@_molecular_hamiltonian_dispatch.register(list)
def _(  # pylint: disable=too-many-arguments
    symbols,
    coordinates,
    unit="bohr",
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    method="pyscf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    convert_tol=1e12,
):

    if (coord_unit := unit.strip().lower()) not in ("angstrom", "bohr"):
        raise ValueError(
            f"The provided unit '{unit}' is not supported. "
            f"Please set 'unit' to 'bohr' or 'angstrom'."
        )

    if coord_unit == "angstrom":
        coordinates = coordinates / BOHR_TO_ANG

    return _molecular_hamiltonian(
        symbols,
        coordinates=coordinates,
        name=name,
        charge=charge,
        mult=mult,
        basis=basis,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping=mapping,
        outpath=outpath,
        wires=wires,
        convert_tol=convert_tol,
    )


def _molecular_hamiltonian(  # pylint: disable=too-many-arguments
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    method="pyscf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    convert_tol=1e12,
):
    r"""Generate the qubit Hamiltonian of a molecule."""
    # pylint: disable=protected-access

    method = method.strip().lower()
    if method == "dhf":
        raise ValueError(
            "The differentiable Hartree-Fock solver (method='dhf') has been removed. "
            "Use method='pyscf' (the default) or method='openfermion'."
        )
    if method not in ["pyscf", "openfermion"]:
        raise ValueError("Only 'pyscf' and 'openfermion' backends are supported.")

    mapping = mapping.strip().lower()
    if mapping not in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        raise ValueError(
            f"'{mapping}' is not supported."
            f"Please set the mapping to 'jordan_wigner', 'parity' or 'bravyi_kitaev'."
        )

    geometry_hf = coordinates.flatten()

    wires_map = None

    if wires:
        wires_new = qp.qchem.convert._process_wires(wires)
        wires_map = dict(zip(range(len(wires_new)), list(wires_new.labels), strict=True))

    if method == "pyscf":
        n_electrons = sum(atomic_numbers[s] for s in symbols) - charge

        if n_electrons % 2 == 1 or mult != 1:
            raise ValueError(
                "Open-shell systems are not supported for the requested backend. Use "
                "method = 'openfermion' or change the charge or spin multiplicity of the molecule."
            )

        core_constant, one_mo, two_mo = qp.qchem.openfermion_pyscf._pyscf_integrals(
            symbols,
            geometry_hf,
            charge,
            mult,
            basis,
            active_electrons,
            active_orbitals,
        )

        hf = qp.qchem.fermionic_observable(core_constant, one_mo, two_mo)

        qubits = len(hf.wires)

        if mapping == "jordan_wigner":
            h_pl = qp.jordan_wigner(hf, wire_map=wires_map, tol=1.0e-10)
        elif mapping == "parity":
            h_pl = qp.parity_transform(hf, qubits, wire_map=wires_map, tol=1.0e-10)
        elif mapping == "bravyi_kitaev":
            h_pl = qp.bravyi_kitaev(hf, qubits, wire_map=wires_map, tol=1.0e-10)

        h_pl = h_pl.simplify()

        return h_pl, len(h_pl.wires)

    h_pl = qp.qchem.openfermion_pyscf._openfermion_hamiltonian(
        symbols,
        geometry_hf,
        name,
        charge,
        mult,
        basis,
        active_electrons,
        active_orbitals,
        mapping,
        outpath,
        wires,
        convert_tol,
    )

    return h_pl, len(h_pl.wires)


def molecular_integrals(molecule, active_electrons=None, active_orbitals=None):
    r"""Compute the one- and two-electron integrals of a molecule in the molecular orbital basis.

    The integrals are obtained from a Hartree-Fock calculation with PySCF and define the
    second-quantized electronic Hamiltonian

    .. math::

        H = E_\text{core} + \sum_{pq} h_{pq} c_p^{\dagger} c_q
        + \frac{1}{2} \sum_{pqrs} h_{pqrs} c_p^{\dagger} c_q^{\dagger} c_r c_s,

    where :math:`E_\text{core}` is the core constant, :math:`h_{pq}` and :math:`h_{pqrs}` are the
    one- and two-electron integrals, and :math:`c^{\dagger}` and :math:`c` are the creation and
    annihilation operators. The two-electron integrals use the physicists' notation expected by
    :func:`~.qchem.fermionic_observable` and :func:`~.qchem.basis_rotation`.
    For :func:`~.qchem.factorize`, convert the tensor to chemists' notation with
    ``np.swapaxes(two, 1, 3)``.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        active_electrons (int): Number of active electrons. If not specified, all electrons are
            considered to be active.
        active_orbitals (int): Number of active spatial orbitals. If not specified, all orbitals
            are considered to be active.

    Returns:
        tuple[array[float], array[float], array[float]]: the core constant and the one- and
        two-electron integrals

    .. note::

        This function requires PySCF, which can be installed with ``pip install pyscf``.
        The integrals are not differentiable with respect to the molecular parameters.

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    >>> mol = qp.qchem.Molecule(symbols, geometry)
    >>> core, one, two = qp.qchem.molecular_integrals(mol)
    >>> core
    array([0.71428571])
    >>> np.round(np.diag(one), 6)
    array([-1.252797, -0.475602])
    >>> two.shape
    (2, 2, 2, 2)
    >>> factors, _, _ = qp.qchem.factorize(np.swapaxes(two, 1, 3))
    >>> factors.shape
    (3, 2, 2)
    """
    # pylint: disable-next=protected-access
    return qp.qchem.openfermion_pyscf._pyscf_integrals(
        molecule.symbols,
        qp.math.toarray(molecule.coordinates).flatten(),
        molecule.charge,
        molecule.mult,
        molecule.basis_name,
        active_electrons,
        active_orbitals,
    )
