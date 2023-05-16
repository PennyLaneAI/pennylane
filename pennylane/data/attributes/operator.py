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
"""Contains AttributeType definition for subclasses of ``pennylane.operation.Operator``."""

from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar
import json

from pennylane.data.base.attribute import AttributeType
from pennylane.data.attributes.array import DatasetArray
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.operation import Operator
from pennylane import Hamiltonian
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word


@lru_cache(1)
def _get_all_operator_classes() -> Tuple[Type[Operator], ...]:
    """This function returns a tuple of every subclass of
    ``pennylane.operation.Operator``."""
    acc = set()

    def rec(cls):
        for subcls in cls.__subclasses__():
            if subcls not in acc:
                acc.add(subcls)
                rec(subcls)

    rec(Operator)

    return tuple(acc - {Hamiltonian})


@lru_cache(1)
def _operator_name_to_class_dict() -> Dict[str, Type[Operator]]:
    """Returns a dictionary mapping the type name of each ``pennylane.operation.Operator``
    class to the class."""

    op_classes = _get_all_operator_classes()

    return {op.__qualname__: op for op in op_classes}


Op = TypeVar("Op", bound=Operator)


class DatasetOperator(Generic[Op], AttributeType[ZarrGroup, Op, Op]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "operator"

    def __post_init__(self, value: Op, info):
        """Save the class name of the operator ``value`` into the
        attribute info."""
        super().__post_init__(value, info)
        self.info["operator_class"] = type(value).__qualname__

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Operator], ...]:
        return _get_all_operator_classes()

    def zarr_to_value(self, bind: ZarrGroup) -> Op:
        mapper = AttributeTypeMapper(bind)

        op_cls = _operator_name_to_class_dict()[mapper.info["operator_class"]]
        op = object.__new__(op_cls)

        for attr_name, attr in mapper.items():
            if attr_name == "data":
                setattr(op, attr_name, list(attr.bind))
            else:
                setattr(op, attr_name, attr.copy_value())

        return op

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Op) -> ZarrGroup:
        bind = bind_parent.create_group(key)
        mapper = AttributeTypeMapper(bind)

        for attr_name, attr in vars(value).items():
            if attr_name == "data":
                mapper.set_item(attr_name, attr, None, require_type=DatasetArray)
            else:
                mapper[attr_name] = attr

        return bind


class DatasetHamiltonian(AttributeType[ZarrGroup, Hamiltonian, Hamiltonian]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "hamiltonian"

    def __post_init__(self, value: Hamiltonian, info):
        """Save the class name of the operator ``value`` into the
        attribute info."""
        super().__post_init__(value, info)
        self.info["operator_class"] = type(value).__qualname__

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        wire_map = {w: i for i, w in enumerate(json.loads(bind["wires"][()]))}

        ops = [string_to_pauli_word(pauli_string, wire_map) for pauli_string in bind["ops"]]
        coeffs = list(bind["coeffs"])

        return Hamiltonian(coeffs, ops)

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        coeffs, ops = value.terms()
        wire_map = {w: i for i, w in enumerate(value.wires)}

        bind.array("ops", data=[pauli_word_to_string(op, wire_map) for op in ops], dtype=str)
        bind.array("coeffs", data=coeffs)
        bind.array("wires", data=json.dumps([w for w in value.wires]), dtype=str)

        return bind
