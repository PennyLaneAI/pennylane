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
Contains the :class:`~pennylane.data.Dataset` class and its associated functions.
"""

from abc import ABC
import dill
import zstd


class Dataset(ABC):
    """Create a dataset object to store a collection of information describing
    a physical system and its evolution. For example, a dataset for an arbitrary
    quantum system could have a Hamiltonian, its ground state,
    and an efficient state-preparation circuit for that state.

    Args:
        dtype (string): the type of the dataset, e.g., `qchem`, `qspin`, etc.
        **kwargs: variable-length keyworded arguments specifying the data to be stored in the dataset

    **Example**

    >>> Hamiltonian = qml.Hamiltonian([1., 1.], [qml.PauliZ(wires=0), qml.PauliZ(wires=1)])
    >>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(Hamiltonian))
    >>> ground_state_energy = np.min(eigvals)
    >>> ground_state = np.transpose(eigvecs)[np.argmin(eigvals)]
    >>> dataset = qml.data.Dataset(Hamiltonian = Hamiltonian, ground_state = ground_state,
            ground_state_energy = ground_state_energy)
    >>> print(dataset.Hamiltonian)
          (1) [Z0]
        + (1) [Z1]
    >>> print(dataset.ground_energy)
    -2.0

    .. details::
        :title: Usage Details

        In addition to creating datasets in memory, we can also store them in
        the disk and then load them as follows. First we create the dataset:

        >>> Hamiltonian = qml.Hamiltonian([1., 1.], [qml.PauliZ(wires=0), qml.PauliZ(wires=1)])
        ...     eigvals, eigvecs = np.linalg.eigh(qml.matrix(Hamiltonian))
        >>> ground_state_energy = np.min(eigvals)
        >>> ground_state = np.transpose(eigvecs)[np.argmin(eigvals)]
        >>> dataset = qml.data.Dataset(Hamiltonian = Hamiltonian, ground_state = ground_state,
        ...     ground_state_energy = ground_state_energy)

        Then to save the dataset to a file, we call :func:`Dataset.write()`:

        >>> dataset.write('./path/to/file/dataset.dat')

        We can then retrieve the data using :func:`Dataset.read()`

        >>> retrieved_data = qml.data.Dataset()
        >>> retrieved_data.read('./path/to/file/dataset.dat')
        >>> print(retrieved_data.Hamiltonian)
          (1) [Z0]
        + (1) [Z1]
        >>> print(dataset.Hamiltonian)
          (1) [Z0]
        + (1) [Z1]
    """

    def __init__(self, dtype=None, **kwargs):
        self.dtype = dtype
        self.__doc__ = ""
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __eq__(self, __o):
        return self.__dict__ == __o.__dict__

    @staticmethod
    def _read_file(filepath):
        """Loads data previously saved with :func:`~pennylane.data.dataset.write`.
        Data is read as a dictionary."""
        with open(filepath, "rb") as file:
            compressed_pickle = file.read()

        depressed_pickle = zstd.decompress(compressed_pickle)
        data = dill.loads(depressed_pickle)
        return data

    def read(self, filepath):
        """Loads data from a saved file to the current dataset.

        Args:
            filepath (string): the desired location and filename to load, e.g. './path/to/file/file_name.dat'.

        **Example**

        >>> new_dataset = qml.data.Dataset(kw1 = 1, kw2 = '2', kw3 = [3])
        >>> new_dataset.read('./path/to/file/file_name.dat')
        """
        data = self._read_file(filepath)
        for (key, val) in data.items():
            setattr(self, f"{key}", val)

    @staticmethod
    def _write_file(filepath, data, protocol=4):
        """Writes the input data to a file as a dictionary."""
        pickled_data = dill.dumps(data, protocol=protocol)  # returns data as a bytes object
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as file:
            file.write(compressed_pickle)

    def write(self, filepath, protocol=4):
        """Writes a dataset to a file as a dictionary.

        Args:
            filepath (string): the desired save location and file name

        **Example**

        >>> new_dataset = qml.data.Dataset(kw1 = 1, kw2 = '2', kw3 = [3])
        >>> new_dataset.write('./path/to/file/file_name.dat')
        """
        dataset = {}
        for (key, val) in self.__dict__.items():
            dataset.update({key: val})
        self._write_file(filepath=filepath, data=dataset, protocol=protocol)

    @classmethod
    def from_dataset(cls, dataset):
        """Builds a dataset from another dataset. Copies the data from another :class:`~pennylane.data.Dataset`.

        Args:
            dataset (Dataset): the dataset to copy

        Returns:
            Dataset: a new dataset containing the same keys and values as the original

        **Example**

            >>> original_dataset = qml.data.Dataset(kw1 = 1, kw2 = '2', kw3 = [3])
            >>> new_dataset = qml.data.Dataset.from_dataset(original_dataset)
            >>> print(vars(original_dataset))
            {'dtype': None, '__doc__': '', 'kw1': 1, 'kw2': '2', 'kw3': [3]}
            >>> print(vars(new_dataset))
            {'dtype': None, '__doc__': '', 'kw1': 1, 'kw2': '2', 'kw3': [3]}
        """
        kwargs = {key: val for (key, val) in dataset.__dict__.items() if key not in ["dtype"]}
        return cls(dtype=dataset.dtype, **kwargs)
