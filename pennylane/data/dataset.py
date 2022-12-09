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
from glob import glob
import os

from pennylane import Hamiltonian
from pennylane.pauli import string_to_pauli_word


def _import_zstd_dill():
    """Import zstd and dill."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import zstd, dill
    except ImportError as Error:
        raise ImportError(
            "This feature requires zstd and dill."
            "They can be installed with:\n\n pip install zstd dill."
        ) from Error

    return zstd, dill


# pylint: disable=too-many-instance-attributes
class Dataset(ABC):
    """Create a dataset object to store a collection of information describing
    a physical system and its evolution. For example, a dataset for an arbitrary
    quantum system could have a Hamiltonian, its ground state,
    and an efficient state-preparation circuit for that state.

    Args:
        *args: For internal use only. These will be ignored if called with ``standard=False``
        standard (bool): For internal use only. See the note below for the behavior when this is set to ``True``
        **kwargs: variable-length keyword arguments specifying the data to be stored in the dataset

    Note on the ``standard`` kwarg:
        A `standard` Dataset uses previously generated, hosted quantum data. This special instance of the
        ``Dataset`` class makes certain assumptions about folder management for downloading the data and
        handling I/O. As such, the ``Dataset`` class should not be instantiated by the users directly with
        ``standard=True``. Instead, they should use :meth:`~load`.


    .. seealso:: :meth:`~load`

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

    _standard_argnames = ["data_name", "data_folder", "attr_prefix", "docstring"]

    def __std_init__(self, data_name, folder, attr_prefix, docstring):
        """Constructor for standardized datasets."""
        self._dtype = data_name
        self._folder = folder.rstrip(os.path.sep)
        self._prefix = os.path.join(self._folder, attr_prefix) + "_{}.dat"
        self._prefix_len = len(attr_prefix) + 1
        self._description = os.path.join(data_name, self._folder.split(data_name)[-1][1:])
        self.__doc__ = docstring

        self._fullfile = self._prefix.format("full")
        if not os.path.exists(self._fullfile):
            self._fullfile = None

        for f in glob(f"{self._folder}{os.path.sep}*.dat"):
            self.read(f, lazy=True)

    def __base_init__(self, **kwargs):
        """Constructor for user-defined datasets."""
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __init__(self, *args, standard=False, **kwargs):
        """Dispatching constructor for direct invocation with kwargs or from qml.data.load()"""
        if standard:
            if len(args) != len(self._standard_argnames):
                raise TypeError(
                    f"Standard datasets expect {len(self._standard_argnames)} arguments: {self._standard_argnames}"
                )
            for name, val in zip(self._standard_argnames, args):
                if not isinstance(val, str):
                    raise ValueError(f"Expected {name} to be a str, got {type(val).__name__}")
            self._is_standard = True
            self.__std_init__(*args)
        else:
            self._is_standard = False
            self.__base_init__(**kwargs)

    def __repr__(self):
        attr_str = (
            str(list(self.attrs))
            if len(self.attrs) < 3
            else f"{str(list(self.attrs)[:2])[:-1]}, ...]"
        )

        std_str = f"description: {self._description}, " if self._is_standard else ""
        return f"<Dataset = {std_str}attributes: {attr_str}>"

    @property
    def attrs(self):
        """Returns attributes of the dataset."""
        return {k: v for k, v in vars(self).items() if k[0] != "_"}

    @staticmethod
    def _read_file(filepath):
        """Read data from a saved file.

        Returns:
            A data value for non-standard datasets or full files, otherwise a dictionary
        """
        with open(filepath, "rb") as f:
            compressed_pickle = f.read()

        zstd, dill = _import_zstd_dill()
        depressed_pickle = zstd.decompress(compressed_pickle)
        return dill.loads(depressed_pickle)

    def read(self, filepath, lazy=False):
        """Loads data from a saved file to the current dataset.

        Args:
            filepath (string): The desired location and filename to load, e.g. './path/to/file/file_name.dat'.
            lazy (bool): Indicates if only the key of the attribute should be saved to the Dataset instance

        **Example**

        >>> new_dataset = qml.data.Dataset(kw1 = 1, kw2 = '2', kw3 = [3])
        >>> new_dataset.read('./path/to/file/file_name.dat')
        """
        # set 'full' for non-standard datasets so they read keys too
        attribute = self.__get_attribute_from_filename(filepath) if self._is_standard else "full"
        if attribute == "full":
            data = self._read_file(filepath)
            for attr, value in data.items():
                setattr(self, f"{attr}", None if lazy else value)
        else:
            data = None if lazy else self._read_file(filepath)
            setattr(self, f"{attribute}", data)

    @staticmethod
    def _write_file(data, filepath, protocol=4):
        """General method to write data to a file."""
        zstd, dill = _import_zstd_dill()
        pickled_data = dill.dumps(data, protocol=protocol)
        compressed_pickle = zstd.compress(pickled_data)
        with open(filepath, "wb") as f:
            f.write(compressed_pickle)

    def write(self, filepath, protocol=4):
        """Writes the dataset to a file as a dictionary.

        Args:
            filepath (string): the desired save location and file name

        **Example**

        >>> new_dataset = qml.data.Dataset(kw1 = 1, kw2 = '2', kw3 = [3])
        >>> new_dataset.write('./path/to/file/file_name.dat')
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        self._write_file(self.attrs, filepath, protocol=protocol)

    def list_attributes(self):
        """List the attributes saved on the Dataset"""
        return list(self.attrs)

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        cls = self.__class__

        if not self._is_standard:
            return cls(**self.attrs)

        copied = cls.__new__(cls)
        copied._is_standard = True
        copied._dtype = self._dtype
        copied._folder = self._folder
        copied._prefix = self._prefix
        copied._prefix_len = self._prefix_len
        copied._fullfile = self._fullfile
        copied.__doc__ = self.__doc__
        for key, val in self.attrs.items():
            setattr(copied, f"{key}", val)

        return copied

    # The methods below are intended for use only by standard Dataset objects
    def __get_filename_for_attribute(self, attribute):
        return self._fullfile or self._prefix.format(attribute)

    def __get_attribute_from_filename(self, filename):
        return os.path.basename(filename)[self._prefix_len : -4]

    @staticmethod
    def __dict_to_hamiltonian(hamil_dict, wire_map):
        coeffs, ops = [], []
        for key, val in hamil_dict.items():
            coeffs.append(val)
            ops.append(string_to_pauli_word(key, wire_map))
        return Hamiltonian(coeffs, ops)

    # TODO: validate this and add to _write_file (if 'hamiltonian' in data))
    # @staticmethod
    # def __hamiltonian_to_dict(hamiltonian, wire_map):
    #     hamil_dict = {}
    #     coeffs, ops = hamiltonian.terms()
    #     for coeff, op in zip(coeffs, ops):
    #         hamil_dict.update({pauli_word_to_string(op, wire_map): coeff})
    #     return hamil_dict

    def __getattribute__(self, name):
        try:
            val = super().__getattribute__(name)
            if val is None and self._is_standard and name in self.attrs:
                raise AttributeError(
                    f"Dataset has a '{name}' attribute, but it is None and no data file was found"
                )
            return val
        except AttributeError:
            if not self._is_standard:
                raise
            filepath = self.__get_filename_for_attribute(name)
            if not os.path.exists(filepath):  # TODO: download the file here?
                raise
            value = self._read_file(filepath)
            if filepath == self._fullfile:
                if name not in value:
                    raise
                value = value[name]
            if name == "hamiltonian":
                value = self.__dict_to_hamiltonian(value["terms"], value["wire_map"])
            setattr(self, name, value)
            return value
