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
Contains the :class:`~pennylane.data.Dataset` base class, and `qml.data.Attribute` class
for declaratively defining dataset classes.
"""

import typing
from dataclasses import InitVar, dataclass
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
    Tuple,
)

# pylint doesn't think this exists
from typing_extensions import dataclass_transform  # pylint: disable=no-name-in-module

from pennylane.data.base import hdf5
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group, h5py
from pennylane.data.base.mapper import MapperMixin, match_obj_type
from pennylane.data.base.typing_util import UNSET, T


@dataclass
class Field(Generic[T]):
    """
    The Attribute class is used to declaratively define the
    attributes of a Dataset subclass, in a similar way to
    dataclasses. This class should not be used directly when
    declaring attributes, use the ``attribute()`` function instead.

    Attributes:
        attribute_type: The ``DatasetAttribute`` class for this attribute
        info: Attribute info
    """

    attribute_type: Type[DatasetAttribute[HDF5Any, T, Any]]
    info: AttributeInfo


def field(  # pylint: disable=too-many-arguments, unused-argument
    attribute_type: Union[Type[DatasetAttribute[HDF5Any, T, Any]], Literal[UNSET]] = UNSET,
    doc: Optional[str] = None,
    py_type: Optional[Any] = None,
    is_param: bool = False,
    **kwargs,
) -> Any:
    """Used to define fields on a declarative Dataset.

    Args:
        attribute_type: ``DatasetAttribute`` class for this attribute. If not provided,
            type may be derived from the type annotation on the class.
        doc: Documentation for the attribute
        py_type: Type annotation or string describing this object's type. If not
            provided, the annotation on the class will be used
        kwargs: Extra arguments to ``AttributeInfo``
    """

    return Field(
        cast(Type[DatasetAttribute[HDF5Any, T, T]], attribute_type),
        AttributeInfo(doc=doc, py_type=py_type, is_param=is_param, **kwargs),
    )


class _InitArg:  # pylint: disable=too-few-public-methods
    """Sentinel value returned by ``_init_arg()``."""


def _init_arg(  # pylint: disable=unused-argument
    default: Any, alias: Optional[str] = None, kw_only: bool = False
) -> Any:
    """This function exists only for the benefit of the type checker. It is used to
    annotate attributes on ``Dataset`` that are not part of the data model, but
    should appear in the generated ``__init__`` method.
    """
    return _InitArg


@dataclass_transform(
    order_default=False, eq_default=False, kw_only_default=True, field_specifiers=(field, _init_arg)
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
    """

    Self = TypeVar("Self", bound="Dataset")

    type_id = "dataset"

    __data_name__: ClassVar[str]
    __params__: ClassVar[Tuple[str, ...]]

    fields: ClassVar[typing.Mapping[str, Field]] = MappingProxyType({})

    bind_: Optional[HDF5Group] = _init_arg(default=None, alias="bind", kw_only=False)
    validate: InitVar[bool] = _init_arg(default=False, kw_only=False)
    data_name_: Optional[str] = _init_arg(default=None, alias="data_name")

    def __init__(
        self,
        bind: Optional[HDF5Group] = None,
        validate: bool = False,
        *,
        data_name: Optional[str] = None,
        params: Optional[Tuple[str, ...]] = None,
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
            data_name: String describing the type of data this datasets contains, e.g
                'qchem' for quantum chemistry. Defaults to the data name defined by
                the class, this is 'generic' for base datasets.
            params: Tuple of names of attributes of this dataset that will serve
                as its parameters
            **attrs: Attributes to add to this dataset.
        """
        if isinstance(bind, (h5py.Group, h5py.File)):
            self._bind = bind
        else:
            self._bind = hdf5.create_group()

        self._init_bind(data_name, params)

        for name in self.fields:
            try:
                attr_value = attrs.pop(name)
                setattr(self, name, attr_value)
            except KeyError:
                pass

        for name, attr in attrs.items():
            setattr(self, name, attr)

        if validate:
            self._validate_attrs()

    @classmethod
    def open(
        cls,
        filepath: Union[str, Path],
        mode: Literal["w", "w-", "a", "r"] = "r",
        copy: bool = False,
    ) -> "Dataset":
        """Open existing dataset or create a new one at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist). Default is "r".
            copy: Whether to load the dataset into memory after opening. If False, and the dataset
                is opened in write mode, any changes made to the dataset will be automatically written
                to the file.

        Returns:
            Dataset object from file
        """
        filepath = Path(filepath).expanduser()

        f = h5py.File(filepath, mode)

        if copy:
            f_copy = hdf5.create_group()
            hdf5.copy_all(f, f_copy)
            f.close()

            f = f_copy

        return cls(f)

    @property
    def data_name(self) -> str:
        """Returns the data name (category) of this dataset."""
        return self.info.get("data_name", self.__data_name__)

    @property
    def params(self) -> typing.Mapping[str, str]:  # pylint: disable=function-redefined
        """Returns this dataset's parameters."""
        return {attr_name: getattr(self, attr_name) for attr_name in self.info.get("params", [])}

    @property
    def info(self) -> AttributeInfo:
        """Return metadata associated with this dataset."""
        return AttributeInfo(self.bind.attrs)

    @property
    def bind(self) -> HDF5Group:  # pylint: disable=function-redefined
        """Return the HDF5 group that contains this dataset."""
        return self._bind

    @property
    def attrs(self) -> typing.Mapping[str, DatasetAttribute]:
        """Returns all attributes of this Dataset."""
        return self._mapper.view()

    @property
    def attr_info(self) -> typing.Mapping[str, AttributeInfo]:
        """Returns a mapping of the ``AttributeInfo`` for each of this dataset's attributes."""
        return MappingProxyType(
            {
                attr_name: AttributeInfo(self.bind[attr_name].attrs)
                for attr_name in self.list_attributes()
            }
        )

    def list_attributes(self) -> List[str]:
        """Returns a list of this dataset's attributes."""
        return list(self.attrs.keys())

    def read(
        self,
        source: Union[str, Path, "Dataset"],
        attributes: Optional[typing.Iterable[str]] = None,
        *,
        overwrite_attrs: bool = False,
    ) -> None:
        """Load dataset from HDF5 file at filepath.

        Args:
            source: Dataset, or path to HDF5 file containing dataset, from which
                to read attributes
            attributes: Optional list of attributes to copy. If None, all attributes
                will be copied.
            overwrite_attrs: Whether to overwrite attributes that already exist in this
                dataset.
        """
        if not isinstance(source, Dataset):
            source = Path(source).expanduser()
            source = Dataset.open(source, mode="r")

        source.write(self, attributes=attributes, overwrite_attrs=overwrite_attrs)

    def write(
        self,
        dest: Union[str, Path, "Dataset"],
        mode: Literal["w", "w-", "a"] = "a",
        attributes: Optional[typing.Iterable[str]] = None,
        *,
        overwrite_attrs: bool = False,
    ) -> None:
        """Write dataset to HDF5 file at filepath. Can also accept an S3 URL.

        Args:
            dest: HDF5 file, or path to HDF5 file containing dataset, to write
                attributes to
            mode: File handling mode, if ``source`` is a file system path. Possible
                values are "w-" (create, fail if file exists), "w" (create, overwrite existing),
                and "a" (append existing, create if doesn't exist). Default is "w-".
            attributes: Optional list of attributes to copy. If None, all attributes
                will be copied.
            overwrite_attrs: Whether to overwrite attributes that already exist in this
                dataset.
        """
        attributes = attributes if attributes is not None else ()
        on_conflict = "overwrite" if overwrite_attrs else "ignore"

        if not isinstance(dest, Dataset):
            dest = Path(dest).expanduser()
            dest = Dataset.open(dest, mode=mode)
            dest.info.update(self.info)

        hdf5.copy_all(self.bind, dest.bind, *attributes, on_conflict=on_conflict)

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

    def _init_bind(self, data_name: Optional[str] = None, params: Optional[Tuple[str, ...]] = None):
        if self.bind.file.mode == "r+":
            if "type_id" not in self.info:
                self.info["type_id"] = self.type_id
            if "data_name" not in self.info:
                self.info["data_name"] = data_name or self.__data_name__
            if "params" not in self.info:
                self.info["params"] = params or self.__params__

    def __setattr__(self, __name: str, __value: Union[Any, DatasetAttribute]) -> None:
        if __name.startswith("_") or __name in type(self).__dict__:
            object.__setattr__(self, __name, __value)
            return

        if __name in self.fields:
            field_ = self.fields[__name]
            self._mapper.set_item(__name, __value, field_.info, field_.attribute_type)
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
            for name, value in {**self.params, "attributes": (str(list(self.attrs)))}.items()
        )

        return f"<{type(self).__name__} = {repr_items}>"

    def __init_subclass__(
        cls, *, data_name: Optional[str] = None, params: Optional[Tuple[str, ...]] = None
    ) -> None:
        """Initializes the ``fields`` dict of a Dataset subclass using
        the declared ``Attributes`` and their type annotations."""

        super().__init_subclass__()

        fields = {}
        if data_name:
            cls.__data_name__ = data_name
        if params:
            cls.__params__ = params

        # get field info from annotated class attributes, e.g:
        # name: int = field(...)
        for name, annotated_type in cls.__annotations__.items():
            if get_origin(annotated_type) is ClassVar:
                continue

            try:
                field_ = getattr(cls, name)
                delattr(cls, name)
            except AttributeError:
                # field only has type annotation
                field_ = field()

            if field_ is _InitArg:
                continue

            field_.info.py_type = annotated_type
            if field_.attribute_type is UNSET:
                field_.attribute_type = match_obj_type(annotated_type)

            fields[name] = field_

        cls.fields = MappingProxyType(fields)

    __data_name__ = "generic"
    __params__ = tuple()


class _DatasetAttributeType(DatasetAttribute[HDF5Group, Dataset, Dataset]):
    """Attribute type for loading and saving datasets as attributes of
    datasets, or elements of collection types."""

    type_id = "dataset"

    def hdf5_to_value(self, bind: HDF5Group) -> Dataset:
        return Dataset(bind)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Dataset) -> HDF5Group:
        hdf5.copy(value.bind, bind_parent, key)

        return bind_parent[key]
