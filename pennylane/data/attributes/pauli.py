import json
from typing import Tuple, Type

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import HDF5Array, HDF5Group
from pennylane.pauli import PauliSentence, PauliWord


class DatasetPauliWord(AttributeType[HDF5Array, PauliWord, PauliWord]):
    type_id = "pauli_word"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[PauliWord]]:
        return (PauliWord,)

    def hdf5_to_value(self, bind: HDF5Array) -> PauliWord:
        return PauliWord(json.loads(bind.asstr()[()]))

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: PauliWord) -> HDF5Array:
        bind_parent[key] = json.dumps(dict(value.items()))

        return bind_parent[key]


class DatasetPauliSentence(AttributeType[HDF5Group, PauliSentence, PauliSentence]):
    type_id = "pauli_sentence"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[PauliSentence]]:
        return (PauliSentence,)

    def hdf5_to_value(self, bind: HDF5Group) -> PauliSentence:
        return PauliSentence(
            (PauliWord(json.loads(word)), coeff)
            for word, coeff in zip(bind["words"], bind["coeffs"])
        )

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: PauliWord) -> HDF5Group:
        bind = bind_parent.create_group(key)

        bind["coeffs"] = list(value.values())
        bind["words"] = [json.dumps(word) for word in value.keys()]

        return bind
