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

import typing
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_origin,
)

from typing_extensions import dataclass_transform

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
    """Used to define fields on a declarative Dataclass.

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
    order_default=False, eq_default=False, field_specifiers=(attribute,), kw_only_default=True
)
class _DatasetTransform:  # pylint: disable=too-few-public-methods
    """This base class that tells the type system that ``Dataset`` behaves like a dataclass.
    See: https://peps.python.org/pep-0681/
    """


class Dataset(AttributeType[HDF5Group, "Dataset", "Dataset"], MapperMixin, _DatasetTransform):
    """
    Base class for Datasets.

    Attributes:
        fields: A mapping of attribute names to their ``Attribute`` information. Note that
            this contains attributes declared on the class, not attributes added to
            an instance. Use ``attrs`` to view all attributes on an instance.
        bind: The HDF5 group that contains this dataset.
    """

    Self = TypeVar("Self", bound="Dataset")

    type_id = "dataset"

    fields: ClassVar[typing.Mapping[str, Attribute]] = MappingProxyType({})

    bind: HDF5Group = attribute(default=None, kw_only=False)  # type: ignore

    def __init__(
        self,
        bind: Optional[Union[HDF5Group, Tuple[HDF5Group, str]]] = None,
        info: Optional[AttributeInfo] = None,
        **attrs: Any,
    ):
        """
        Load a dataset from a HDF5 Group or initialize a new Dataset.

        Args:
            bind: The HDF5 group, or path to hdf5 file, that will contain this dataset.
                If None, the dataset will be stored in memory. Any attributes that
                already exist in ``bind`` will be loaded into this dataset.
            **attrs: Attributes to add to this dataset.
        """
        if isinstance(bind, (h5py.Group, h5py.File)):
            super().__init__(value=None, info=info, bind=bind)  # type: ignore
        else:
            super().__init__(value=None, info=info, parent_and_key=bind)

        self._validate_arguments(attrs)
        for name, attr in attrs.items():
            setattr(self, name, attr)

    @classmethod
    def consumes_types(cls) -> Tuple[Type["Dataset"]]:
        return (cls,)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value) -> HDF5Group:
        return bind_parent.create_group(key)

    def hdf5_to_value(self, bind: HDF5Group) -> "Dataset":
        return self

    @property
    def bind(self) -> HDF5Group:
        """The HDF5 group that contains this dataset's attributes."""
        return self._bind

    @property
    def attrs(self) -> typing.Mapping[str, AttributeType]:
        """Returns all attributes of this Dataset."""
        return self._mapper.view()

    def list_attributes(self) -> List[str]:
        """Returns a list of this dataset's attributes."""
        return list(self.attrs.keys())

    @classmethod
    def open(
        cls: Type[Self], filepath: Union[str, Path], mode: Literal["w", "w-", "a", "r"] = "r"
    ) -> Self:
        """Open existing dataset or create a new one file at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist). Default is "r".
        Returns:
            Dataset object from file
        """
        f = h5py.File(filepath, mode)

        return cls(bind=f)

    def read(
        self,
        filepath: Union[str, Path],
        assign_to: Optional[str] = None,
        *,
        overwrite_attrs: bool = False,
    ):
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

    def write(self, filepath: Union[str, Path], mode: Literal["w", "w-", "a"] = "w-"):
        """Write dataset to HDF5 file at filepath. Can also accept an S3 URL.

        Args:
            filepath: Path of hdf5 file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), and "a" (append existing,
                create if doesn't exist). Default is "w-".
        """
        # TODO: better error message for 'ContainsGroupError'

        with h5py.open_group(filepath, mode=mode) as zgrp:
            h5py.copy_all(self.bind, zgrp)
            zgrp.attrs.update(self.bind.attrs)

    def _validate_arguments(self, args: Dict[str, Any]):
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
            super().__setattr__(__name, __value)
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
            raise AttributeError(f"'{type(self)}' object has no attribute '{__name}'") from exc

    def __delattr__(self, __name: str) -> None:
        try:
            del self._mapper[__name]
        except KeyError as exc:
            raise AttributeError(f"'{type(self)}' object has no attribute '{__name}'") from exc

    def __repr__(self) -> str:
        attrs_repr = ", ".join(
            (f"{attr_name}={repr(attr.get_value())}" for attr_name, attr in self.attrs.items())
        )

        return f"Dataset({attrs_repr})"

    def __init_subclass__(cls, **kwargs) -> None:
        """Initializes the ``fields`` dict of a Dataset subclass using
        the declared ``Attributes`` and their type annotations."""
        super().__init_subclass__(**kwargs)

        fields = {}

        for name, annotated_type in cls.__annotations__.items():
            if (
                name.startswith("_")
                or isinstance(annotated_type, InitVar)
                or get_origin(annotated_type) is ClassVar
            ):
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
