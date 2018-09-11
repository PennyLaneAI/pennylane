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
import os
import logging as log

import toml
from appdirs import user_config_dir

log.getLogger()


class Configuration:
    """Configuration class.

    This class is responsible for loading, saving, and storing OpenQML
    and plugin/device configurations.
    """
    def __init__(self, name='config.toml'):
        # Look for an existing configuration file
        self._config = {}
        self._filename = None
        self._name = 'config.toml'
        self._user_config_dir = user_config_dir('openqml')
        self._env_config_dir = os.environ.get("OPENQML_CONF", "")

        # search the current directory the directory under environment
        # variable OPENQML_CONF, and default user config directory, in that order.
        directories = [os.curdir, self._env_config_dir, self._user_config_dir]
        for idx, directory in enumerate(directories):
            try:
                self._filepath = os.path.join(directory, self._name)
                self.load(self._filepath)
                break
            except FileNotFoundError:
                if idx == len(directories)-1:
                    log.warning('No OpenQML configuration file found.')

    def __str__(self):
        return self._config

    def __repr__(self):
        return "OpenQML Configuration <{}>".format(self._filename)

    @property
    def filename(self):
        return self._filename

    def load(self, filepath):
        """Load a configuration file."""
        with open(filepath, 'r') as f:
            self._config = toml.load(f)

    def save(self, filepath):
        """Save a configuration file."""
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
        """Safely set value from a nested dictionary."""
        for key in keys[:-1]:
            dct = dct.setdefault(key, {})

        dct[keys[-1]] = value

    @staticmethod
    def safe_get(dct, *keys):
        """Safely return value from a nested dictionary."""
        for key in keys:
            try:
                dct = dct[key]
            except KeyError:
                return {}
        return dct
