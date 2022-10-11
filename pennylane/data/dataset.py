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

    def __init__(self, dtype=None, **kwargs):
        self.dtype = dtype
        self.__doc__ = ""
        for key, val in kwargs.items():
            setattr(self, f"{key}", val)

    @staticmethod
    def _read_file(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    def read(self, filepath):
        """Loading the dataset from a saved file"""
        data = self._read_file(filepath)
        for (key, val) in data.items():
            setattr(self, f"{key}", val)

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

    @staticmethod
    def from_dataset(dataset, copy_dtype=False):
        """ Build a dataset from another dataset """
        dataset = Dataset(dtype=dataset.dtype if copy_dtype else None)
        for (key, val) in dataset.__dict__.items():
            if key not in ["dtype"]:
                dataset.setattr(key, val)
        return dataset
