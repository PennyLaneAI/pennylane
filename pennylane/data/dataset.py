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
Contains the :class:`~pennylane.data.Dataset` class, and `qml.data.Attribute` class
for declaratively defining dataset classes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Generic, Literal, Optional, Union, cast

from pennylane.data.base._zarr import zarr
from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.mapper import MapperMixin, match_obj_type
from pennylane.data.base.typing_util import UNSET, T, ZarrAny, ZarrGroup


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
        default: Default value for this attribute, if not provided
            to __init__()
        default_factory: Zero-argument callable that returns
            a default value
    """

    attribute_type: type[AttributeType[ZarrAny, T]]
    info: AttributeInfo
    default: Union[Literal[UNSET], T, None]
    default_factory: Optional[Callable[[], T]] = None


def attribute(
    default: Union[Literal[UNSET], T, None] = UNSET,
    attribute_type: Union[type[AttributeType[ZarrAny, T]], Literal[UNSET]] = UNSET,
    doc: Optional[str] = None,
    py_type: Optional[Any] = None,
    default_factory: Optional[Callable[[], T]] = None,
    **extra: Any,
) -> Attribute[T]:
    """Used to define fields on a declarative Dataclass.

    Args:
        default: See ``Attribute.default``
        attribute_type: ``AttributeType`` class for this attribute. If not provided,
            type may be derived from the type annotation on the class.
        doc: Documentation for the attribute
        py_type: Type annotation or string describing this object's type. If not
            provided, the annotation on the class will be used
        default_factory: See ``Attribute.default_factory``
    """

    return Attribute(
        cast(type[AttributeType[ZarrAny, T]], attribute_type),
        AttributeInfo(doc=doc, py_type=py_type, extra=extra),
        default=default,
        default_factory=default_factory,
    )


class Dataset(MapperMixin):
    type_id = "dataset"

    fields: ClassVar[Mapping[str, Attribute]] = MappingProxyType({})

    def __init__(
        self,
        bind: Optional[Union[str, Path, ZarrGroup]] = None,
        **attrs: Any,
    ):
        """
        Load a dataset from a Zarr Group or initialize a new Dataset.

        Args:
            bind: The Zarr group, or path to zarr file, that will contain this dataset.
                If None, the dataset will be stored in memory. Any attributes that
                already exist in ``bind`` will be loaded into this dataset.
            **attrs: Attributes to add to this dataset.
        """
        if bind is None:
            self._bind = zarr.group()
            self._bind.attrs["type_id"] = self.type_id
        elif isinstance(bind, zarr.Group):
            self._bind = bind
        else:
            self._bind = zarr.open_group(path=bind)

        self._validate_arguments(attrs)
        for name, attr in attrs.items():
            setattr(self, name, attr)

    @property
    def bind(self) -> ZarrGroup:
        """The Zarr group that contains this dataset's attributes."""
        return self._bind

    @property
    def attrs(self) -> Mapping[str, AttributeType]:
        """Returns all attributes of this Dataset."""
        return self._mapper.view()

    def list_attributes(self) -> list[str]:
        return list(self.attrs.keys())

    def read(
        self,
        filepath: Union[str, Path],
        assign_to: Optional[str] = None,
        *,
        overwrite_attrs: bool = False,
    ):
        """Load dataset from Zarr file at filepath. Can also accept an S3 URL.

        Args:
            filepath: Path to file containing dataset
            assign_to: Attribute name to which the contents of the file should be assigned.
                If this is ``None`` (the default value), the file's attributes will assigned
                to the current dataset
            overwrite_attrs: Whether to overwrite attributes that already exist in this
                dataset.
        """
        zgrp = zarr.open_group(filepath, mode="r")

        if assign_to:
            zarr.convenience.copy(zgrp, self.bind, assign_to)
        else:
            if_exists = "replace" if overwrite_attrs else "raise"
            zarr.convenience.copy_all(zgrp, self.bind, if_exists=if_exists)

    def write(self, filepath: Union[str, Path], mode: Literal["w", "w-", "a"] = "w-"):
        """Write dataset to Zarr file at filepath. Can also accept an S3 URL.

        Args:
            filepath: Path of zarr file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), and "a" (append existing,
                create if doesn't exist). Default is "w-".
        """
        zgrp = zarr.open_group(path=filepath, mode=mode)
        zarr.convenience.copy_all(self.bind, zgrp)

    def _validate_arguments(self, args: dict[str, Any]):
        """Validates arguments to __init__() based on the declared
        fields of this dataset."""
        missing = []
        for name, field in self.fields.items():
            if name not in args:
                if field.default_factory is not None:
                    args[name] = field.default_factory()
                elif field.default != UNSET:
                    args[name] = field.default
                else:
                    missing.append(name)

        if missing:
            missing_args = ", ".join(f"'{arg}'" for arg in missing)
            raise TypeError(
                f"__init__() missing {len(missing)} required keyword argument(s): {missing_args}"
            )

    def __setattr__(self, __name: str, __value: Union[Any, AttributeType]) -> None:
        if __name.startswith("_"):
            return super().__setattr__(__name, __value)

        self._mapper[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self._mapper[__name].get_value()
        except KeyError as exc:
            raise AttributeError(name=__name, obj=self) from exc

    def __init_subclass__(cls, **kwargs) -> None:
        """Initializes the ``fields`` dict of a Dataset subclass using
        the declared ``Attributes`` and their type annotations."""
        ignore = {"fields", "type_id"}
        fields = {}

        for name, annotated_type in cls.__annotations__.items():
            if name in ignore or name.startswith("_"):
                continue

            attr = getattr(cls, name, None)
            if attr is None:
                attr = attribute()
            else:
                delattr(cls, name)

            if not isinstance(attr, Attribute):
                attr = attribute(default=attr)

            attr.info.py_type = annotated_type
            if attr.attribute_type is UNSET:
                attr.attribute_type = match_obj_type(annotated_type)

            fields[name] = attr

        cls.fields = MappingProxyType(fields)
