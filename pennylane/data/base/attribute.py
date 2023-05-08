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
"""Contains the base class for Dataset attribute types, and a class for
attribute metadata."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping
from numbers import Number
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)
import itertools
from pennylane.data.base._zarr import zarr
from pennylane.data.base.typing_util import (
    UNSET,
    T,
    Zarr,
    ZarrAny,
    ZarrGroup,
    get_type_str,
    resolve_special_type,
)


class AttributeInfo(Generic[T], MutableMapping[str, Any]):
    """Contains metadata that may be assigned to a dataset
    attribute. Is stored in the Zarr object's ``attrs`` dict.

    Attributes:
        attrs_bind: The Zarr attrs dict that this instance is bound to,
            or any mutable mapping
        py_type: Type annotation for this attribute
        doc: Documentation for this attribute
        meta: Extra metdata to attach to this attribute. Must be
            json serializable.
    """

    attrs_namespace: ClassVar[str] = "qml.data."

    attrs_bind: MutableMapping[str, Any]

    doc: Optional[str]

    @overload
    def __init__(
        self,
        attrs_bind: Optional[MutableMapping[str, Any]],
        *,
        doc: Optional[str] = None,
        py_type: Optional[str] = None,
    ):
        ...

    @overload
    def __init__(self):
        ...

    def __init__(self, attrs_bind: Optional[MutableMapping[str, Any]] = None, **kwargs: Any):
        object.__setattr__(self, "attrs_bind", attrs_bind or {})

        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, attrs_bind: MutableMapping[str, Any], clobber: bool = False):
        for k in self.raw_keys():
            if k not in attrs_bind or clobber:
                attrs_bind[k] = self.attrs_bind[k]

    def load(self, attrs_bind: MutableMapping[str, Any], clobber: bool = False):
        AttributeInfo(attrs_bind).save(self.attrs_bind, clobber=clobber)

    @property
    def py_type(self) -> Optional[str]:
        return self.get("py_type")

    @py_type.setter
    def py_type(self, type_: Union[str, type]):
        self["py_type"] = get_type_str(type_)

    def __len__(self) -> int:
        return self.get("__len__", 0)

    def _update_len(self, inc: int):
        self.attrs_bind[f"{self.attrs_namespace}__len__"] = self.__len__() + inc

    def __setitem__(self, __name: str, __value: Any):
        key = f"{self.attrs_namespace}{__name}"
        exists = key in self.attrs_bind
        self.attrs_bind[key] = __value
        if not exists:
            self._update_len(1)

    def __getitem__(self, __name: str) -> Any:
        try:
            return self.attrs_bind[f"{self.attrs_namespace}{__name}"]
        except KeyError as exc:
            raise KeyError(__name) from exc

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__class__.__dict__:
            object.__setattr__(self, __name, __value)
        else:
            self[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self[__name]
        except KeyError:
            return None

    def __delitem__(self, __name: str):
        del self.attrs_bind[f"{self.attrs_namespace}{__name}"]
        self._update_len(-1)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain.from_iterable(
            key.split(self.attrs_namespace, maxsplit=1)[1:2] for key in self.raw_keys()
        )

    def raw_keys(self) -> Iterator[str]:
        return (key for key in self.attrs_bind if key.startswith(self.attrs_namespace))


class AttributeType(ABC, Generic[Zarr, T]):
    """
    The AttributeType class provides an interface for converting Python objects to and from a Zarr
    array or Group. It uses the registry pattern to maintain a mapping of codec IDs to
    Codecs, and Python types to compatible Codecs.

    Attributes:
        type_id: Unique identifier for this Codec class. Must be declared
            in subclasses.
        registry: Maps codec ids to compatible Codec classes
        type_to_default_codec_id: Maps types to their default Codec classes
    """

    type_id: ClassVar[str]

    Self = TypeVar("Self", bound="AttributeType")

    _parent: ZarrGroup
    _name: Optional[str] = None
    _key: Optional[str] = None

    def __init__(
        self,
        value: Optional[T] = None,
        info: Optional[AttributeInfo] = None,
        *,
        parent: Optional[ZarrGroup] = None,
        key: Optional[str] = None,
    ) -> None:
        if parent is not None:
            if key is None:
                raise TypeError("'key' argument must be provided for 'parent'.")
            self._parent = parent
            self._key = key
        else:
            self._parent = zarr.group()
            self._key = "_"

        if self.key not in self._parent:
            if value is None:
                value = self.default_value()
                if value is UNSET:
                    raise TypeError("'value' not provided and attribute does not exist in parent.")

            info = info or AttributeInfo()
            info.py_type = type(value)

            self.set_value(value, info)

        self._check_bind(self.bind)

    @classmethod
    def consumes_types(cls) -> Iterable[type]:
        """
        Returns an iterable of types for which this should be the default
        codec. If a value of one of these types is assigned to a Dataset
        without specifying a `type_id`, this type will be used.
        """
        return ()

    def default_value(self) -> Union[T, Literal[UNSET]]:
        """Returns a valid default value for this type, or ``UNSET`` if this type
        must be initialized with a value."""
        return UNSET

    @abstractmethod
    def zarr_to_value(self, bind: Zarr) -> T:
        """Parses bind into Python object."""

    @abstractmethod
    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: T) -> Zarr:
        """Converts value into a Zarr Array or Group under bind_parent[key]."""

    @property
    def key(self) -> str:
        """The bound name of this attribute under ``parent``."""
        return self._key or ""

    @property
    def info(self) -> AttributeInfo:
        """Returns the ``AttributeInfo`` for this attribute."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> Zarr:
        """Returns the Zarr object that contains this attribute's
        data."""
        return cast(Zarr, self._parent[self.key])

    @property
    def bind_parent(self) -> ZarrGroup:
        """Returns the Zarr group that contains this attribute."""
        return self._parent

    def get_value(self) -> T:
        """Loads the mapped value from ``bind``."""
        return self.zarr_to_value(self.bind)

    def set_value(self, value: T, info: Optional[AttributeInfo]) -> Zarr:
        """Converts ``value`` into Zarr format and sets the attribute info."""
        if info is None:
            info = AttributeInfo()
            info.load(self.bind.attrs)

        info["type_id"] = self.type_id

        new_bind = self.value_to_zarr(self.bind_parent, self.key, value)
        info.save(new_bind.attrs)

        return new_bind

    def _set_parent(self, parent: ZarrGroup, key: str):
        """Copies this attribute's data into ``parent``, under ``key``."""
        zarr.convenience.copy(source=self.bind, dest=parent, name=key)
        self._parent = parent
        self._key = key

    def _check_bind(self, bind: Zarr):
        """
        Checks that ``bind.attrs`` contains the type_id corresponding to
        this type.
        """
        existing_type_id = self.info.get("type_id")
        if existing_type_id is not None and existing_type_id != self.type_id:
            raise TypeError(
                f"zarr {type(bind).__qualname__} is bound to another type {existing_type_id}"
            )

    def __str__(self) -> str:
        return str(self.get_value())

    def __repr__(self) -> str:
        return object.__repr__(self)

    def __eq__(self, __value: object) -> bool:
        return self.get_value() == __value

    __registry: dict[str, type["AttributeType"]] = {}
    __type_consumer_registry: dict[type, type["AttributeType"]] = {}

    registry: Mapping[str, type["AttributeType"]] = MappingProxyType(__registry)
    type_consumer_registry: Mapping[type, type["AttributeType"]] = MappingProxyType(
        __type_consumer_registry
    )

    def __init_subclass__(cls) -> None:  # pylint: disable=arguments-differ
        existing_type = AttributeType.__registry.get(cls.type_id)
        if existing_type is not None:
            raise TypeError(
                f"AttributeType with type_id '{cls.type_id}' already exists: {existing_type}"
            )

        AttributeType.__registry[cls.type_id] = cls  # type: ignore

        for type_ in cls.consumes_types():
            existing_type = AttributeType.type_consumer_registry.get(type_)
            if existing_type is not None:
                warnings.warn(
                    f"Conflicting default types: Both '{cls}' and '{existing_type}' consume type '{type_}'. '{type_}' will now be consumed by '{cls}'"
                )
            AttributeType.__type_consumer_registry[type_] = cls

        return super().__init_subclass__()


def get_attribute_type(zobj: Zarr) -> type[AttributeType[Zarr, Any]]:
    """
    Returns the ``AttributeType`` of the dataset attribute contained
    in ``zobj``.
    """
    type_id = zobj.attrs[f"{AttributeInfo.attrs_namespace}type_id"]

    return AttributeType.registry[type_id]


def _get_type(type_or_obj: Union[object, type]) -> type:
    """Given an object or an object type, returns the underlying class.

    Examples:
        >>> _get_type(list)
        <class 'list'>
        >>> _get_type(List[int])
        <class 'list'>
        >>> _get_type([])
        <class 'list'>
    """

    # First, check if this is a special type, e.g a parametrized
    # generic like List[int]
    special_args = resolve_special_type(type_or_obj)
    if special_args is not None:
        type_, _ = special_args
    elif isinstance(type_or_obj, type):
        type_ = type_or_obj
    else:
        type_ = type(type_or_obj)

    return type_


def match_obj_type(type_or_obj: Union[T, type[T]]) -> type[AttributeType[ZarrAny, T]]:
    """
    Returns an ``AttributeType`` that can accept an object of type ``type_or_obj``
    as a value.

    Args:
        type_or_obj: A type or an object

    Returns:
        AttributeType that can accept ``type_or_obj`` (or an object of that
            type) as a value.

    Raises:
        TypeError, if no AttributeType can accept an object of that type
    """

    type_ = _get_type(type_or_obj)
    ret = AttributeType.registry["array"]

    if type_ in AttributeType.type_consumer_registry:
        ret = AttributeType.type_consumer_registry[type_]
    elif issubclass(type_, Number):
        ret = AttributeType.registry["scalar"]
    elif hasattr(type_, "__array__"):
        ret = AttributeType.registry["array"]

    return ret
