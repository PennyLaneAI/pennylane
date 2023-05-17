# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains AttributeType definition for ``pennylane.qchem.Molecule``."""

from typing import Tuple, Type

import numpy as np

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.qchem import Molecule


class DatasetMolecule(AttributeType[ZarrGroup, Molecule, Molecule]):
    """Attribute type for ``pennylane.qchem.Molecule``."""

    type_id = "molecule"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Molecule], ...]:
        return (Molecule,)

    def zarr_to_value(self, bind: ZarrGroup) -> Molecule:
        return Molecule(
            symbols=list(bind["symbols"].asstr()),
            coordinates=np.array(bind["coordinates"]),
            charge=int(bind["charge"][()]),
            mult=int(bind["mult"][()]),
            basis_name=bind["basis_name"].asstr()[()],
            l=np.array(bind["l"]),
            alpha=np.array(bind["alpha"]),
            coeff=np.array(bind["coeff"]),
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Molecule) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        bind["symbols"] = value.symbols
        bind["coordinates"] = value.coordinates
        bind["charge"] = value.charge
        bind["mult"] = value.mult
        bind["basis_name"] = value.basis_name
        bind["l"] = value.l
        bind["alpha"] = value.alpha
        bind["coeff"] = value.coeff

        return bind
