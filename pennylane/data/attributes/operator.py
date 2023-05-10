from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import ZarrGroup, get_type_str
from pennylane.operation import Operator
from copy import deepcopy


@lru_cache(1)
def _get_all_operator_classes() -> Tuple[Type[Operator], ...]:
    acc = set()

    def rec(cls):
        for subcls in cls.__subclasses__():
            if subcls not in acc:
                acc.add(subcls)
                rec(subcls)

    rec(Operator)

    return tuple(acc)


@lru_cache(1)
def _operator_name_to_class_dict() -> Dict[str, Type[Operator]]:
    op_classes = _get_all_operator_classes()

    return {get_type_str(op): op for op in op_classes}


Op = TypeVar("Op", bound=Operator)


class _DatasetOperatorBase(Generic[Op], AttributeType[ZarrGroup, Op, Op], abstract=True):
    def __post_init__(self, value: Op, info):
        self.info["operator_class"] = get_type_str(type(value))

        super().__post_init__(value, info)


class DatasetOperator(Generic[Op], _DatasetOperatorBase[Op]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` classes
    whose ``__init__()`` method meets the following conditions:

        - The ``params``
        - it must accept ``wires`` and ``id`` as keyword arguments,
        and any additional keywrod arguments must be hyperparameters, or can excluded
        from the constructor without loss of information
    """

    type_id = "operator"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Operator], ...]:
        return _get_all_operator_classes()

    def zarr_to_value(self, bind: ZarrGroup) -> Op:
        mapper = AttributeTypeMapper(bind)

        op_cls = _operator_name_to_class_dict()[mapper.info["operator_class"]]
        op = object.__new__(op_cls)

        for attr_name, attr in mapper.items():
            setattr(op, attr_name, attr.copy_value())

        return op

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Op) -> ZarrGroup:
        bind = bind_parent.create_group(key)
        mapper = AttributeTypeMapper(bind)

        for attr_name, attr in vars(value).items():
            mapper[attr_name] = attr

        return bind
