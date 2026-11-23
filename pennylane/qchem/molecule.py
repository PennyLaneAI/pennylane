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
"""Molecular metadata for external quantum chemistry backends."""

import collections
from dataclasses import dataclass, field

from scipy.constants import angstrom, physical_constants

from pennylane.typing import TensorLike

from .atomic_numbers import atomic_numbers

BOHR_TO_ANG = physical_constants["Bohr radius"][0] / angstrom


@dataclass(eq=False)
class Molecule:  # pylint: disable=too-many-instance-attributes
    r"""Store molecular information for quantum chemistry calculations.

    Args:
        symbols (list[str]): symbols of the atomic species
        coordinates (array[float]): Cartesian coordinates, shaped ``(N, 3)`` or ``(3*N,)``
        charge (int): net charge of the molecule
        mult (int): spin multiplicity
        basis_name (str): atomic basis set understood by the selected backend
        name (str): name of the molecule
        unit (str): coordinate unit, ``"bohr"`` or ``"angstrom"``

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    >>> mol = qp.qchem.Molecule(symbols, geometry)
    >>> mol.n_electrons
    2
    """

    symbols: list[str]
    coordinates: TensorLike
    charge: int = 0
    mult: int = 1
    basis_name: str = "sto-3g"
    name: str = "molecule"
    unit: str = "bohr"
    nuclear_charges: list[int] = field(init=False)
    n_electrons: int = field(init=False)

    def __post_init__(self):
        if set(self.symbols) - set(atomic_numbers):
            raise ValueError(
                f"Atoms in {set(self.symbols) - set(atomic_numbers)} are not supported."
            )

        self.unit = self.unit.strip().lower()
        self.basis_name = self.basis_name.lower()

        if self.unit not in ("angstrom", "bohr"):
            raise ValueError(
                f"The provided unit '{self.unit}' is not supported. "
                f"Please set 'unit' to 'bohr' or 'angstrom'."
            )
        if self.unit == "angstrom":
            self.coordinates = self.coordinates / BOHR_TO_ANG

        self.nuclear_charges = [atomic_numbers[symbol] for symbol in self.symbols]
        self.n_electrons = sum(self.nuclear_charges) - self.charge

    def __repr__(self):
        elements, counter, flags = set(self.symbols), collections.Counter(self.symbols), []
        if counter["C"]:
            flags = ["C", "H"] if counter["H"] else ["C"]
        ordered_elems = flags + list(sorted(elements.difference(set(flags))))
        formula = "".join([x + str(counter[x]) if counter[x] > 1 else x for x in ordered_elems])
        return (
            f"<Molecule = {formula}, Charge: {self.charge}, "
            f"Basis: {self.basis_name.upper()}, Electrons: {self.n_electrons}>"
        )
