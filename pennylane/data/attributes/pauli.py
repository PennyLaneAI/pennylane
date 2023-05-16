from pennylane.data.base.attribute import AttributeType
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.data.base.typing_util import ZarrArray, ZarrGroup
from typing import Tuple, Type
import json
from pennylane.pauli.pauli_arithmetic import PauliWord


class DatasetPauliWord(AttributeType[ZarrArray, PauliWord, PauliWord]):
    type_id = "pauli_word"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[PauliWord]]:
        return (PauliWord,)

    def zarr_to_value(self, bind: ZarrArray) -> PauliWord:
        return PauliWord(json.loads(bind.asstr()[()]))

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: PauliWord) -> ZarrArray:
        bind_parent[key] = json.dumps(dict(value.items()))

        return bind_parent[key]


class DatasetPauliSentence(AttributeType[ZarrGroup, PauliSentence, PauliSentence]):
    type_id = "pauli_sentence"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[PauliSentence]]:
        return (PauliSentence,)

    def zarr_to_value(self, bind: ZarrGroup) -> PauliSentence:
        return PauliSentence(
            (PauliWord(json.loads(word)), coeff)
            for word, coeff in zip(bind["words"], bind["coeffs"])
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: PauliWord) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        bind["coeffs"] = list(value.values())
        bind["words"] = [json.dumps(word) for word in value.keys()]

        return bind
