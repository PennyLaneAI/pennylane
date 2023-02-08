# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains the :class:`Configuration` class, which is used to
load, store, save, and modify configuration options for PennyLane and all
supported plugins and devices.
"""

import contextlib
import os

import toml
from appdirs import user_config_dir


class Configuration:
    """Configuration class.

    This class is responsible for loading, saving, and storing PennyLane
    and plugin/device configurations.

    Args:
        name (str): filename of the configuration file.
            This should be a valid TOML file. You may also pass an absolute
            or a relative file path to the configuration file.
    """

    def __init__(self, name):
        # Look for an existing configuration file
        self._config = {}
        self._filepath = None
        self._name = name
        self._user_config_dir = user_config_dir("pennylane", "Xanadu")
        self._env_config_dir = os.environ.get("PENNYLANE_CONF", "")

        # search the current directory the directory under environment
        # variable PENNYLANE_CONF, and default user config directory, in that order.
        directories = [os.curdir, self._env_config_dir, self._user_config_dir, ""]
        for directory in directories:
            with contextlib.suppress(FileNotFoundError):
                self._filepath = os.path.join(directory, self._name)
                self.load(self._filepath)
                break

    def __str__(self):
        if self._config:
            return f"{self._config}"
        return ""

    def __repr__(self):
        return f"PennyLane Configuration <{self._filepath}>"

    @property
    def path(self):
        """Return the path of the loaded configuration file.

        Returns:
            str: If no configuration is loaded, this returns ``None``."""
        return self._filepath

    def load(self, filepath):
        """Load a configuration file.

        Args:
            filepath (str): path to the configuration file.
        """
        with open(filepath, "r", encoding="utf8") as f:
            self._config = toml.load(f)

    def save(self, filepath):
        """Save a configuration file.

        Args:
            filepath (str): path to the configuration file.
        """
        with open(filepath, "w", encoding="utf8") as f:
            toml.dump(self._config, f)

    def __getitem__(self, key):
        keys = key.split(".")
        return self.safe_get(self._config, *keys)

    def __setitem__(self, key, value):
        keys = key.split(".")
        self.safe_set(self._config, value, *keys)

    def __bool__(self):
        return bool(self._config)

    @staticmethod
    def safe_set(dct, value, *keys):
        """Safely set the value of a key from a nested dictionary.

        If any key provided does not exist, a dictionary containing the
        remaining keys is dynamically created and set to the required value.

        Args:
            dct (dict): the dictionary to set the value of.
            value: the value to set. Can be any valid type.
            *keys: each additional argument corresponds to a nested key.
        """
        for key in keys[:-1]:
            dct = dct.setdefault(key, {})

        dct[keys[-1]] = value

    @staticmethod
    def safe_get(dct, *keys):
        """Safely return value from a nested dictionary.

        If any key provided does not exist, an empty dictionary is returned.

        Args:
            dct (dict): the dictionary to set the value of.
            *keys: each additional argument corresponds to a nested key.

        Returns:
            value corresponding to ``dct[keys[0]][keys[1]]`` etc.
        """
        for key in keys:
            try:
                dct = dct[key]
            except KeyError:
                return {}
        return dct
