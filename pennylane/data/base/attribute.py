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

import itertools
import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Number
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pennylane.data.base._hdf5 import h5py
from pennylane.data.base.typing_util import (
    HDF5,
    UNSET,
    HDF5Any,
    HDF5Group,
    T,
    get_type_str,
    resolve_special_type,
)


class AttributeInfo(MutableMapping):
    """Contains metadata that may be assigned to a dataset
    attribute. Is stored in the HDF5 object's ``attrs`` dict.

    Attributes:
        attrs_namespace: Keys for this class's attributes will
            be prefixed with this string in ``attrs_bind``.
        attrs_bind: The HDF5 attrs dict that this instance is bound to,
            or any mutable mapping
        py_type: Type annotation for this attribute
        doc: Documentation for this attribute
        **kwargs: Extra metadata to include. Must be a string, number
            or numpy array
    """

    attrs_namespace: ClassVar[str] = "qml.data."
    attrs_bind: typing.MutableMapping[str, Any]

    doc: Optional[str]

    @overload
    def __init__(  # overload to specify known keyword args
        self,
        attrs_bind: Optional[typing.MutableMapping[str, Any]] = None,
        *,
        doc: Optional[str] = None,
        py_type: Optional[str] = None,
        **kwargs: Any,
    ):
        ...

    @overload
    def __init__(self):  # need at least two overloads when using @overload
        ...

    def __init__(self, attrs_bind: Optional[typing.MutableMapping[str, Any]] = None, **kwargs: Any):
        object.__setattr__(self, "attrs_bind", attrs_bind if attrs_bind is not None else {})

        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, info: "AttributeInfo") -> None:
        """Inserts the values set in this instance into ``info``."""
        for k, v in self.items():
            info[k] = v

    def load(self, info: "AttributeInfo"):
        """Inserts the values set in ``info`` into this instance."""
        info.save(self)

    @property
    def py_type(self) -> Optional[str]:
        """String representation of this attribute's python type."""
        return self.get("py_type")

    @py_type.setter
    def py_type(self, type_: Union[str, Type]):
        self["py_type"] = get_type_str(type_)

    def __len__(self) -> int:
        return self.get("__len__", 0)

    def _update_len(self, inc: int):
        self.attrs_bind[f"{self.attrs_namespace}__len__"] = len(self) + inc

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

    def __delitem__(self, __name: str) -> None:
        del self.attrs_bind[f"{self.attrs_namespace}{__name}"]
        self._update_len(-1)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain.from_iterable(
            key.split(self.attrs_namespace, maxsplit=1)[1:2] for key in self.attrs_bind
        )


# Type variable for 'value' argument to 'AttributeType.__init__()'
InitValueType = TypeVar("InitValueType")


class AttributeType(ABC, Generic[HDF5, T, InitValueType]):
    """
    The AttributeType class provides an interface for converting Python objects to and from a HDF5
    array or Group. It uses the registry pattern to maintain a mapping of type_id to
    AttributeType, and Python types to compatible AttributeTypes.

    Attributes:
        type_id: Unique identifier for this AttributeType class. Must be declared
            in subclasses.
        registry: Maps type_ids to their AttributeType classes
        type_consumer_registry: Maps types to their default AttributeType
    """

    type_id: ClassVar[str]

    Self = TypeVar("Self", bound="AttributeType")

    @overload
    def __init__(
        self,
        value: Union[InitValueType, Literal[UNSET]] = UNSET,
        info: Optional[AttributeInfo] = None,
        *,
        parent_and_key: Optional[Tuple[HDF5Group, str]] = None,
    ):
        """Initialize a new dataset attribute from ``value``.

        Args:
            value: Value that will be stored in dataset attribute.
            info: Metadata to attach to attribute.
            parent_and_key: A 2-tuple specifying the HDF5 group that will contain
                this attribute, and its key. If None, attribute will be stored in-memory.
        """

    @overload
    def __init__(self, *, bind: HDF5):
        """Load previously persisted dataset attribute from ``bind``.

        If ``bind`` contains an attribute of a different type, or does not
        contain a dataset attribute, a ``TypeError` will be raised.

        Args:
            bind: HDF5 object from which existing attribute will be loaded.
        """

    def __init__(
        self,
        value: Union[InitValueType, Literal[UNSET]] = UNSET,
        info: Optional[AttributeInfo] = None,
        *,
        bind: Optional[HDF5] = None,
        parent_and_key: Optional[Tuple[HDF5Group, str]] = None,
    ) -> None:
        """
        Initialize a new dataset attribute, or load from an existing
        hdf5 object.

        This constructor can be called two ways: value initialization
        or bind initialization.

        Value initialization creates the attribute with specified ``value`` in
        a new HDF5 object, with optional ``info`` attached. The attribute can
        be created in an existing HDF5 group by passing the ``parent_and_key``
        argument.

        Bind initialization loads an attribute that was previously persisted
        in HDF5 object ``bind``.

        Note that if ``bind`` is provided, all other arguments will be ignored.

        Args:
            value: Value to initialize attribute to
            info: Metadata to attach to attribute
            bind: HDF5 object from which existing attribute will be loaded
            parent_and_key: A 2-tuple specifying the HDF5 group that will contain
                this attribute, and its key.
        """
        if bind is not None:
            self._bind = bind
            self._check_bind()
            return

        if parent_and_key is not None:
            parent, key = parent_and_key
        else:
            parent, key = h5py.group(), "_"

        if value is UNSET:
            value = self.default_value()
            if value is UNSET:
                raise TypeError("__init__() missing 1 required positional argument: 'value'")

        self._bind = self._set_value(value, info, parent, key)
        self._check_bind()
        self.__post_init__(value, info)

    @property
    def info(self) -> AttributeInfo:
        """Returns the ``AttributeInfo`` for this attribute."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> HDF5:
        """Returns the HDF5 object that contains this attribute's
        data."""
        return self._bind

    def default_value(self) -> Union[InitValueType, Literal[UNSET]]:
        """Returns a valid default value for this type, or ``UNSET`` if this type
        must be initialized with a value."""
        return UNSET

    def __post_init__(self, value: InitValueType, info: Optional[AttributeInfo]) -> None:
        """Called after __init__(), only during value initialization. Can be implemented
        in subclasses to implement additional initialization"""

    @classmethod
    def py_type(cls, value_type: Type[InitValueType]) -> str:
        """Determines the ``py_type`` of an attribute during value initialization,
        if it was not provided in the ``info`` argument. This method returns
        ``f"{value_type.__module__}.{value_type.__name__}``.
        """
        return get_type_str(value_type)

    @classmethod
    def consumes_types(cls) -> typing.Iterable[type]:
        """
        Returns an iterable of types for which this should be the default
        codec. If a value of one of these types is assigned to a Dataset
        without specifying a `type_id`, this type will be used.
        """
        return ()

    @abstractmethod
    def hdf5_to_value(self, bind: HDF5) -> T:
        """Parses bind into Python object."""

    @abstractmethod
    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: InitValueType) -> HDF5:
        """Converts value into a HDF5 Array or Group under bind_parent[key]."""

    def get_value(self) -> T:
        """Parses the mapped value from ``bind``."""
        return self.hdf5_to_value(self.bind)

    def copy_value(self) -> T:
        """Parse the mapped value from ``bind``, and also perform a 'deep-copy'
        of any nested values contained in ``bind``."""
        return self.get_value()

    def _set_value(
        self, value: InitValueType, info: Optional[AttributeInfo], parent: HDF5Group, key: str
    ) -> HDF5:
        """Converts ``value`` into HDF5 format and sets the attribute info."""
        if info is None:
            info = AttributeInfo()

        info["type_id"] = self.type_id
        if info.py_type is None:
            info.py_type = self.py_type(type(value))

        new_bind = self.value_to_hdf5(parent, key, value)
        new_info = AttributeInfo(new_bind.attrs)
        info.save(new_info)

        return new_bind

    def _set_parent(self, parent: HDF5Group, key: str):
        """Copies this attribute's data into ``parent``, under ``key``."""
        h5py.copy(source=self.bind, dest=parent, key=key, if_exists="replace")

    def _check_bind(self):
        """
        Checks that ``bind.attrs`` contains the type_id corresponding to
        this type.
        """
        existing_type_id = self.info.get("type_id")
        if existing_type_id is None:
            raise TypeError(
                f"HDF5 '{type(self.bind).__qualname__}' does not contain a dataset attribute."
            )
        if existing_type_id != self.type_id:
            raise TypeError(
                f"HDF5 '{type(self.bind).__qualname__}' is bound to another attribute type {existing_type_id}"
            )

    def __str__(self) -> str:
        return str(self.get_value())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.get_value())})"

    def __eq__(self, __value: object) -> bool:
        return self.get_value() == __value

    __registry: typing.Mapping[str, Type["AttributeType"]] = {}
    __type_consumer_registry: typing.Mapping[type, Type["AttributeType"]] = {}

    registry: typing.Mapping[str, Type["AttributeType"]] = MappingProxyType(__registry)
    type_consumer_registry: typing.Mapping[type, Type["AttributeType"]] = MappingProxyType(
        __type_consumer_registry
    )

    def __init_subclass__(  # pylint: disable=arguments-differ
        cls, *, abstract: bool = False
    ) -> None:
        if abstract:
            return super().__init_subclass__()

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

    def __copy__(self: Self) -> Self:
        impl_group = h5py.group()
        h5py.copy(self.bind, impl_group, "_")

        return type(self)(bind=impl_group["_"])

    def __deepcopy__(self: Self, memo) -> Self:
        return self.__copy__()


def get_attribute_type(zobj: HDF5) -> Type[AttributeType[HDF5, Any, Any]]:
    """
    Returns the ``AttributeType`` of the dataset attribute contained
    in ``zobj``.
    """
    type_id = zobj.attrs[f"{AttributeInfo.attrs_namespace}type_id"]

    return AttributeType.registry[type_id]


def _get_type(type_or_obj: Union[object, Type]) -> Type:
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


def match_obj_type(type_or_obj: Union[T, Type[T]]) -> Type[AttributeType[HDF5Any, T, T]]:
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
    elif issubclass(type_, Sequence):
        ret = AttributeType.registry["list"]
    elif issubclass(type_, Mapping):
        ret = AttributeType.registry["dict"]

    return ret
