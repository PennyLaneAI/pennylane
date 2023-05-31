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
"""
Contains the :class:`~pennylane.data.DatasetBase` class, and `qml.data.Attribute` class
for declaratively defining dataset classes.
"""

import json
import typing
from dataclasses import InitVar, dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_origin,
)

# pylint doesn't think this exists
from typing_extensions import dataclass_transform  # pylint: disable=no-name-in-module

from pennylane.data.base._hdf5 import h5py
from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.mapper import MapperMixin, match_obj_type
from pennylane.data.base.typing_util import UNSET, HDF5Any, HDF5Group, T


@dataclass
class Attribute(Generic[T]):
    """
    The Attribute class is used to declaratively define the
    attributes of a Dataset subclass, in a similar way to
    dataclasses. This class should not be used directly when
    declaring attributes, use the ``attribute()`` function instead.

    Attributes:
        attribute_type: The ``AttributeType`` class for this attribute
        info: Attribute info
    """

    attribute_type: Type[AttributeType[HDF5Any, T, Any]]
    info: AttributeInfo


def attribute(  # pylint: disable=too-many-arguments, unused-argument
    attribute_type: Union[Type[AttributeType[HDF5Any, T, Any]], Literal[UNSET]] = UNSET,
    doc: Optional[str] = None,
    py_type: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Used to define fields on a declarative Dataset.

    Args:
        attribute_type: ``AttributeType`` class for this attribute. If not provided,
            type may be derived from the type annotation on the class.
        doc: Documentation for the attribute
        py_type: Type annotation or string describing this object's type. If not
            provided, the annotation on the class will be used
    """

    return Attribute(
        cast(Type[AttributeType[HDF5Any, T, T]], attribute_type),
        AttributeInfo(doc=doc, py_type=py_type),
    )


@dataclass_transform(
    order_default=False,
    eq_default=False,
    field_specifiers=(attribute,),
    kw_only_default=True,
)
class _DatasetTransform:  # pylint: disable=too-few-public-methods
    """This base class that tells the type system that ``Dataset`` behaves like a dataclass.
    See: https://peps.python.org/pep-0681/
    """


class Dataset(MapperMixin, _DatasetTransform):
    """
    Base class for Datasets.

    Attributes:
        fields: A mapping of attribute names to their ``Attribute`` information. Note that
            this contains attributes declared on the class, not attributes added to
            an instance. Use ``attrs`` to view all attributes on an instance.
        bind: The HDF5 group that contains this dataset's attributes
        params: The parameters of this Dataset
    """

    Self = TypeVar("Self", bound="Dataset")

    type_id = "dataset"
    category_id: ClassVar[str]

    fields: ClassVar[typing.Mapping[str, Attribute]] = MappingProxyType({})

    bind: HDF5Group = attribute(  # type: ignore
        kw_only=False,
        default=None,
    )
    params: typing.Mapping[str, str] = attribute(kw_only=False, default=None)  # type: ignore
    validate: InitVar[bool] = attribute(default=False, kw_only=False)

    def __init__(
        self,
        bind: Optional[HDF5Group] = None,
        params: Optional[typing.Mapping[str, str]] = None,
        validate: bool = False,
        **attrs: Any,
    ):
        """
        Load a dataset from a HDF5 Group or initialize a new Dataset.

        Args:
            bind: The HDF5 group that contains this dataset. If None, a new
                group will be created in memory. Any attributes that already exist
                in ``bind`` will be loaded into this dataset.
            validate: If ``True``, all declared fields of this dataset must
                be provided in ``attrs`` or contained in ``bind``.
            **attrs: Attributes to add to this dataset.
        """
        if isinstance(bind, (h5py.Group, h5py.File)):
            self._bind = bind
        else:
            self._bind = h5py.group()

        self._init_bind()
        if params is not None:
            self.info["params"] = json.dumps(params)

        for name, attr in attrs.items():
            setattr(self, name, attr)

        if validate:
            self._validate_attrs()

    @property
    def category(self) -> Optional[str]:
        """Returns the category ID of this dataset."""
        return self.info.get("category_id")

    @property
    def info(self) -> AttributeInfo:
        """Return attribute info associated with this dataset."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> HDF5Group:  # pylint: disable=function-redefined
        """Return the HDF5 group that contains this dataset."""
        return self._bind

    @cached_property
    def params(self) -> typing.Mapping[str, str]:  # pylint: disable=function-redefined
        """Returns this dataset's parameters."""
        return json.loads(self.info.get("params", "{}"))

    @property
    def attrs(self) -> typing.Mapping[str, AttributeType]:
        """Returns all attributes of this Dataset."""
        return self._mapper.view()

    def list_attributes(self) -> List[str]:
        """Returns a list of this dataset's attributes."""
        return list(self.attrs.keys())

    @classmethod
    def open(
        cls,
        filepath: Union[str, Path],
        mode: Literal["w", "w-", "a", "r"] = "r",
        validate: bool = False,
    ) -> "Dataset":
        """Open existing dataset or create a new one file at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist). Default is "r".
        Returns:
            Dataset object from file
        """
        if mode in ("w-", "w"):
            validate = False

        f = h5py.File(filepath, mode)

        return cls(bind=f, validate=validate)

    def read(
        self,
        filepath: Union[str, Path],
        assign_to: Optional[str] = None,
        *,
        overwrite_attrs: bool = False,
    ) -> None:
        """Load dataset from HDF5 file at filepath. Can also accept an S3 URL.

        Args:
            filepath: Path to file containing dataset
            assign_to: Attribute name to which the contents of the file should be assigned.
                If this is ``None`` (the default value), the file's attributes will be assigned
                to the current dataset
            overwrite_attrs: Whether to overwrite attributes that already exist in this
                dataset.
        """
        # TODO: better error message when overwriting attribute fails
        zgrp = h5py.open_group(filepath, mode="r")

        if assign_to:
            h5py.copy(zgrp, self.bind, assign_to)
        else:
            if_exists = "overwrite" if overwrite_attrs else "raise"
            h5py.copy_all(zgrp, self.bind, if_exists=if_exists)

    def write(self, filepath: Union[str, Path], mode: Literal["w", "w-", "a"] = "w-") -> None:
        """Write dataset to HDF5 file at filepath. Can also accept an S3 URL.

        Args:
            filepath: Path of hdf5 file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), and "a" (append existing,
                create if doesn't exist). Default is "w-".
        """
        # TODO: better error message for 'ContainsGroupError'
        with h5py.open_group(filepath, mode=mode) as grp:
            h5py.copy_all(self.bind, grp)
            grp.attrs.update(self.bind.attrs)

    def _validate_attrs(self) -> None:
        """Validates that ``attrs`` matched the delcared fields of this
        dataset."""
        missing = []
        for name in self.fields:
            if name not in self.attrs:
                missing.append(name)

        if missing:
            missing_args = ", ".join(f"'{arg}'" for arg in missing)
            raise TypeError(
                f"__init__() missing {len(missing)} required keyword argument(s): {missing_args}"
            )

    def _init_bind(self):
        if self.bind.file.mode == "r+":
            if "type_id" not in self.info:
                self.info["type_id"] = self.type_id
            if "category_id" not in self.info and (getattr(self, "category_id", None) is not None):
                self.info["category_id"] = self.category_id

    def __setattr__(self, __name: str, __value: Union[Any, AttributeType]) -> None:
        if __name.startswith("_") or __name in type(self).__dict__:
            object.__setattr__(self, __name, __value)
            return

        if __name in self.fields:
            field = self.fields[__name]
            self._mapper.set_item(__name, __value, field.info, field.attribute_type)
        else:
            self._mapper[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self._mapper[__name].get_value()
        except KeyError as exc:
            if __name in self.fields:
                return UNSET

            raise AttributeError(f"'{type(self)}' object has no attribute '{__name}'") from exc

    def __delattr__(self, __name: str) -> None:
        try:
            del self._mapper[__name]
        except KeyError as exc:
            raise AttributeError(f"'{type(self)}' object has no attribute '{__name}'") from exc

    def __repr__(self) -> str:
        repr_items = ", ".join(
            f"{name}: {value}"
            for name, value in {**self.params, "attributes": (str(list(self.attrs)))}
        )

        return f"<{type(self).__name__} = {repr_items}>"

    def __init_subclass__(cls) -> None:
        """Initializes the ``fields`` dict of a Dataset subclass using
        the declared ``Attributes`` and their type annotations."""

        super().__init_subclass__()

        fields = {}
        reserved_names = ("params", "bind", "validate")

        for name, annotated_type in cls.__annotations__.items():
            if (
                name in reserved_names
                or isinstance(annotated_type, InitVar)
                or get_origin(annotated_type) is ClassVar
            ):
                continue

            attr = getattr(cls, name, attribute())
            try:
                delattr(cls, name)
            except AttributeError:
                pass

            attr.info.py_type = annotated_type
            if attr.attribute_type is UNSET:
                attr.attribute_type = match_obj_type(annotated_type)

            fields[name] = attr

        cls.fields = MappingProxyType(fields)


class _DatasetAttributeType(AttributeType[HDF5Group, Dataset, Dataset]):
    """Attribute type for loading and saving datasets as attributes of
    datasets, or elements of collection types."""

    type_id = "dataset"

    def hdf5_to_value(self, bind: HDF5Group) -> Dataset:
        return Dataset(bind)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Dataset) -> HDF5Group:
        h5py.copy(value.bind, bind_parent, key)

        return bind_parent[key]
