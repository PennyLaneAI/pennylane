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

import dill
import zstd


class Dataset:
    "The dataset data type. Allows users to create datasets with their own data"

    def __init__(self, data_file=None, **kwargs):
        self._data_file = data_file
        for key, val in kwargs.items():
            setattr(self, f"_{key}", val)

    @property
    def data_file(self):
        """property of ChemDataset"""
        return self._data_file

    @data_file.setter
    def data_file(self, value):
        self._data_file = value

    @staticmethod
    def read_data(filepath):
        """Reading the data from a saved file"""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()
        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    @staticmethod
    def write_data(filepath, data, protocol=4):
        """Writing the data to a file"""
        pickled_data = dill.dumps(data, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as file:
            file.write(compressed_pickle)

    def write_dataset(self, filepath, protocol=4):
        """Writing the dataset to a file"""
        dataset = {}
        for (key, val) in self.__dict__.items():
            dataset.update({key: val})
        self.write_data(filepath=filepath, data=dataset, protocol=protocol)

    def from_dataset(self, dataset):
        for (key, val) in dataset.__dict__.items():
            setattr(self, key, val)
            # if getattr(self, key, None):
            #    setattr(self, key, val)
