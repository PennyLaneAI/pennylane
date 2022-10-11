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

    def __init__(self, dfile=None, dtype=None, **kwargs):
        self.dfile = dfile
        self.dtype = dtype
        self.__doc__ = ""
        for key, val in kwargs.items():
            setattr(self, f"{key}", val)

    def __getattribute__(self, name):
        try:
            value = super().__getattribute__(name)
        except AttributeError:
            value = None
        dfile = super().__getattribute__("dfile")
        if value is None and dfile is not None:
            file_path = f"{dfile}_{name}.dat"
            if os.path.exists(file_path):
                value = self._read_file(file_path)
        return value

    @staticmethod
    def _read_file(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    def read(self, filepath=None, lazy=True):
        """Loading the dataset from a saved file"""
        filepath = self.dfile if filepath is None else filepath
        data = self._read_file(filepath)
        for (key, val) in data.items():
            setattr(self, f"{key}", val if not lazy else None)

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
        for (key, val) in self.__dict__.items():
            dataset.update({key: val})
        self._write_file(filepath=filepath, data=dataset, protocol=protocol)

    def setattr(self, name, value, update_file=False, protocol=4):
        """ Set value of an attribute and its file """
        setattr(self, name, value)
        if update_file and os.path.exists(self.dfile):
            self._write_file(f"{self.dfile}_{name}.dat", value, protocol)

    @staticmethod
    def from_dataset(dataset, copy_dfile=False):
        """ Build a dataset from another dataset """
        dataset = Dataset(dtype=dataset.dtype, dfile=dataset.dtype if copy_dfile else None)
        for (key, val) in dataset.__dict__.items():
            if key not in ["dtype", "dfile"]:
                dataset.setattr(key, val)
        return dataset

    def setdocstr(self, docstr, args=None, argtypes=None, argsdocs=None):
        """ Build the docstring for the Dataset class """
        docstring = f"""{docstr}\n\n"""
        if args and argsdocs and argtypes:
            docstring += """Args:\n"""
            for idx in range(len(args)):
                arg, argdoc, argtype = args[idx], argsdocs[idx], argtypes[idx]
                docstring += f"""\t{arg} ({argtype}): {argdoc}\n"""
            docstring += f"""\nReturns:\n\tDataset({self.dtype})"""
        self.__doc__ = docstring
