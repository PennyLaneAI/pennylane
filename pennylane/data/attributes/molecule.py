from typing import Tuple, Type

import numpy as np

from pennylane.data.base._zarr import zarr
from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.qchem import Molecule
from pennylane.qchem.molecule import Molecule


class DatasetMolecule(AttributeType[ZarrGroup, Molecule, Molecule]):
    type_id = "molecule"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Molecule], ...]:
        return (Molecule,)

    def zarr_to_value(self, bind: ZarrGroup) -> Molecule:
        return Molecule(
            symbols=list(bind["symbols"]),
            coordinates=np.array(bind["coordinates"]),
            charge=int(bind["charge"][()]),
            mult=int(bind["mult"][()]),
            basis_name=bind["basis_name"][()],
            l=np.array(bind["l"]),
            alpha=np.array(bind["alpha"]),
            coeff=np.array(bind["coeff"]),
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Molecule) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        bind.array("symbols", value.symbols, dtype=str)
        bind["coordinates"] = value.coordinates
        bind["charge"] = value.charge
        bind["mult"] = value.mult
        bind["basis_name"] = value.basis_name
        bind["l"] = value.l
        bind["alpha"] = value.alpha
        bind["coeff"] = value.coeff

        return bind
