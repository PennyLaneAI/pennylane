# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Contains the dataset class.
"""

from abc import ABC
import os
import dill
import zstd


class Dataset(ABC):
    "The dataset data type. Allows users to create datasets with their own data"
    CFG_ATTRS = {"dtype", "folder", "file_prefix", "file_fstring", "_attributes", "__doc__"}

    def __init__(self, dtype, folder, file_prefix, attributes=None, **kwargs):
        self.dtype = dtype
        self.folder = folder
        self.file_prefix = file_prefix
        self.file_fstring = os.path.join(folder, file_prefix) + "_{}.dat"
        self._attributes = attributes or []
        self.__doc__ = ""
        for key, val in kwargs.items():
            setattr(self, f"{key}", val)

    def __get_filename_for_attribute(self, attribute):
        return self.file_fstring.format(attribute)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        filepath = self.__get_filename_for_attribute(name)
        if os.path.exists(filepath):
            # TODO: setattr?
            return self._read_file(filepath)
        # TODO: download the file here?
        raise AttributeError(
            f"'Dataset' object has no attribute {name} and no matching file was found"
        )

    @staticmethod
    def _read_file(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    def read(self, lazy=True):
        """Loading the dataset from saved files"""
        if self._attributes == ["full"]:
            data = self._read_file(self.__get_filename_for_attribute("full"))
            for attr, val in data.items():
                setattr(self, f"{attr}", None if lazy else val)
            return
        for attr in self._attributes:
            val = None if lazy else self._read_file(self.__get_filename_for_attribute(attr))
            setattr(self, f"{attr}", val)

    @staticmethod
    def _write_file(filepath, data, protocol=4):
        """Writing the data to a file"""
        pickled_data = dill.dumps(data, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as file:
            file.write(compressed_pickle)

    def write(self, filepath, protocol=4):
        """Writing the dataset to a file"""
        dataset = {}
        for key, val in vars(self).items():
            dataset.update({key: val})
        self._write_file(filepath=filepath, data=dataset, protocol=protocol)

    def setattr(self, name, value, update_file=False, protocol=4):
        """Set value of an attribute and its file"""
        setattr(self, name, value)
        if update_file:
            self._write_file(self.__get_filename_for_attribute(name), value, protocol)

    def list_attributes(self):
        """List the attributes saved on the Dataset"""
        return list(set(vars(self)) - self.CFG_ATTRS)

    @staticmethod
    def from_dataset(source, new_folder=None):
        copied = Dataset(source.dtype, new_folder or source.folder, source.file_prefix)
        for key, val in vars(source).items():
            if key not in ["dtype", "folder", "file_prefix", "file_fstring"]:
                copied.setattr(key, val)
        return copied

    def setdocstr(self, docstr, args=None, argtypes=None, argsdocs=None):
        """Build the docstring for the Dataset class"""
        docstring = f"{docstr}\n\n"
        if args and argsdocs and argtypes:
            docstring += "Args:\n"
            for arg, argdoc, argtype in zip(args, argsdocs, argtypes):
                docstring += f"\t{arg} ({argtype}): {argdoc}\n"
            docstring += f"\nReturns:\n\tDataset({self.dtype})"
        self.__doc__ = docstring
