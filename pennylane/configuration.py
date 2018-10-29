# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Configuration
=============

**Module name:** :mod:`pennylane.configuration`

.. currentmodule:: pennylane.configuration

This module contains the :class:`Configuration` class, which is used to
load, store, save, and modify configuration options for PennyLane and all
supported plugins and devices.

Behaviour
---------

On first import, PennyLane attempts to load the configuration file `config.toml`, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``PENNYLANE_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/pennylane``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\pennylane``
   * On MacOS: ``~/Library/Preferences/pennylane``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The user can access the initialized configuration via `pennylane.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys (i.e. ``pennylane.config['main.shots']``), and save/load new configuration files.

Configuration files
-------------------

The configuration file `config.toml` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [main]
    # Global PennyLane options.
    # Affects every loaded plugin if applicable.
    shots = 0

    [strawberryfields.global]
    # Options for the Strawberry Fields plugin
    hbar = 1
    shots = 100

      [strawberryfields.fock]
      # Options for the Strawberry Fields Fock plugin
      cutoff_dim = 10
      hbar = 0.5

      [strawberryfields.gaussian]
      # Indentation doesn't matter in TOML files,
      # but helps provide clarity.

    [projectq.global]
    # Options for the Project Q plugin

      [projectq.simulator]
      gate_fusion = true

      [projectq.ibmbackend]
      user = "johnsmith"
      password = "secret123"
      use_hardware = true
      device = "ibmqx4"
      num_runs = 1024

Main PennyLane options, that are passed to all loaded devices, are provided under the ``[main]``
section. Alternatively, options can be specified on a per-plugin basis, by setting the options under
``[plugin.global]``.

For example, in the above configuration file, the Strawberry Fields
devices will be loaded with a default of ``shots = 100``, rather than ``shots = 0``. Finally,
you can also specify settings on a device-by-device basis, by placing the options under the
``[plugin.device]`` settings.

Summary of methods
------------------

.. currentmodule:: pennylane.configuration.Configuration

.. autosummary::
    path
    load
    save

Helper methods
--------------

.. autosummary::
    safe_set
    safe_get

Code details
~~~~~~~~~~~~

.. currentmodule:: pennylane.configuration

"""
import os
import logging as log

import toml
from appdirs import user_config_dir

log.getLogger()


class Configuration:
    """Configuration class.

    This class is responsible for loading, saving, and storing PennyLane
    and plugin/device configurations.

    Args:
        name (str): filename of the configuration file. Default ``'config.toml'``.
        This should be a valid TOML file. You may also pass an absolute
        or a relative file path to the configuration file.
    """
    def __init__(self, name='config.toml'):
        # Look for an existing configuration file
        self._config = {}
        self._filepath = None
        self._name = name
        self._user_config_dir = user_config_dir('pennylane', 'Xanadu')
        self._env_config_dir = os.environ.get("PENNYLANE_CONF", "")

        # search the current directory the directory under environment
        # variable PENNYLANE_CONF, and default user config directory, in that order.
        directories = [os.curdir, self._env_config_dir, self._user_config_dir, '']
        for idx, directory in enumerate(directories):
            try:
                self._filepath = os.path.join(directory, self._name)
                self.load(self._filepath)
                break
            except FileNotFoundError:
                if idx == len(directories)-1:
                    log.warning('No PennyLane configuration file found.')

    def __str__(self):
        if self._config:
            return "{}".format(self._config)
        return ""

    def __repr__(self):
        return "PennyLane Configuration <{}>".format(self._filepath)

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
        with open(filepath, 'r') as f:
            self._config = toml.load(f)

    def save(self, filepath):
        """Save a configuration file.

        Args:
            filepath (str): path to the configuration file.
        """
        with open(filepath, 'w') as f:
            toml.dump(self._config, f)

    def __getitem__(self, key):
        keys = key.split('.')
        return self.safe_get(self._config, *keys)

    def __setitem__(self, key, value):
        keys = key.split('.')
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
