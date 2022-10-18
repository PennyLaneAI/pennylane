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
from glob import glob
import os
import dill
import zstd


class Dataset(ABC):
    "The dataset data type. Allows users to create datasets with their own data"

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, f"{key}", val)

    @property
    def attrs(self):
        return {k: v for k,v in vars(self).items() if k[0] != '_'}

    @staticmethod
    def _read_file(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        return dill.loads(depressed_pickle)

    def read(self, filename=None, foldername=None, lazy=True):
        """Loading the dataset from saved files"""
        if filename:
            files = [os.path.join(foldername, filename)] if foldername else [filename]
        elif foldername:
            files = glob(foldername + "/*.dat")
        else:
            raise ValueError("Must provide one or more of {filename, foldername}")
        for f in files:
            data = self._read_file(f)
            for attr, val in data.items():
                setattr(self, f"{attr}", None if lazy else val)

    def write(self, filepath, protocol=4):
        """Writing the data to a file"""
        pickled_data = dill.dumps(self.attrs, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as file:
            file.write(compressed_pickle)

    def list_attributes(self):
        """List the attributes saved on the Dataset"""
        return list(self.attrs)

    @staticmethod
    def from_dataset(source, new_folder=None):
        copied = Dataset(source._dtype, new_folder or source.folder, source.file_prefix)
        for key, val in source.attrs.items():
            copied.setattr(key, val)
        return copied


class RemoteDataset(Dataset):
    def __init__(self, dtype, folder, attr_prefix):
        self._dtype = dtype
        self._folder = folder.rstrip('/')
        self._prefix = os.path.join(self._folder, attr_prefix)+"_{}.dat"
        self.__doc__ == ""

        for f in glob(self._folder + "/*.dat"):
            data = self._read_file(f)
            for attr, value in data.items():
                setattr(self, f"{attr}", value)

    def __get_filename_for_attribute(self, attribute):
        return self._prefix.format(attribute)

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

    def setdocstr(self, docstr, args=None, argtypes=None, argsdocs=None):
        """Build the docstring for the Dataset class"""
        docstring = f"{docstr}\n\n"
        if args and argsdocs and argtypes:
            docstring += "Args:\n"
            for arg, argdoc, argtype in zip(args, argsdocs, argtypes):
                docstring += f"\t{arg} ({argtype}): {argdoc}\n"
            docstring += f"\nReturns:\n\tDataset({self._dtype})"
        self.__doc__ = docstring
