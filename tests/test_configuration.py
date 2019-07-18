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
"""
Unit tests for the :mod:`pennylane` configuration classe :class:`Configuration`.
"""
# pylint: disable=protected-access
import unittest
import pytest
import os
import logging as log

import toml

from defaults import pennylane, BaseTest
import pennylane as qml
from pennylane import Configuration

log.getLogger('defaults')

@pytest.fixture(scope="session")
def config_path():
    return 'default_config.toml'

@pytest.fixture(scope="session")
def default_config_toml(config_path):
    return toml.load(config_path)

class TestConfigurationLoading:
    """Test loading the configuration from a file."""

    def test_loading_current_directory(self, monkeypatch, config_path, default_config_toml):
        """Test that the default configuration file can be loaded
        from the current directory."""
        
        monkeypatch.chdir(".")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        config = Configuration(name=config_path)

        assert config.path == os.path.join(os.curdir, config_path)
        assert config._config == default_config_toml

    def test_loading_environment_variable(self, monkeypatch, config_path, default_config_toml):
        """Test that the default configuration file can be loaded
        from an environment variable."""

        os.curdir = "None"
        monkeypatch.setenv("PENNYLANE_CONF", os.getcwd())

        config = Configuration(name=config_path)

        assert config._config == default_config_toml
        assert config._env_config_dir == os.environ["PENNYLANE_CONF"]
        assert config.path == os.path.join(os.environ["PENNYLANE_CONF"], config_path)

    def test_loading_absolute_path(self, monkeypatch, config_path, default_config_toml):
        """Test that the default configuration file can be loaded
        from an absolute path."""

        os.curdir = "None"
        monkeypatch.setenv("PENNYLANE_CONF", "")

        config = Configuration(name=os.path.join(os.getcwd(), config_path))

        assert config._config == default_config_toml
        assert config.path == os.path.join(os.getcwd(), config_path)

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""
        
        caplog.clear()
        caplog.set_level(log.INFO)

        Configuration("noconfig")
        
        assert len(caplog.records) == 1
        assert caplog.records[0].message == "No PennyLane configuration file found."

class TestSave:

    def test_save(self):
        """Test saving a configuration file."""
        self.logTestName()

        config = Configuration(name=filename)

        # make a change
        config['strawberryfields.global']['shots'] = 10
        config.save('test_config.toml')

        result = toml.load('test_config.toml')
        os.remove('test_config.toml')
        self.assertEqual(config._config, result)

    def test_get_item(self):
        """Test getting items."""
        self.logTestName()

        config = Configuration(name=filename)

        # get existing options
        self.assertEqual(config['main.shots'], 0)
        self.assertEqual(config['main']['shots'], 0)
        self.assertEqual(config['strawberryfields.global.hbar'], 1)
        self.assertEqual(config['strawberryfields.global']['hbar'], 1)

        # get nested dictionaries
        self.assertEqual(config['strawberryfields.fock'], {'cutoff_dim': 10})

        # get key that doesn't exist
        self.assertEqual(config['projectq.ibm.idonotexist'], {})

    def test_set_item(self):
        """Test setting items."""
        self.logTestName()

        config = Configuration(name=filename)

        # set existing options
        config['main.shots'] = 10
        self.assertEqual(config['main.shots'], 10)
        self.assertEqual(config['main']['shots'], 10)

        config['strawberryfields.global']['hbar'] = 5
        self.assertEqual(config['strawberryfields.global.hbar'], 5)
        self.assertEqual(config['strawberryfields.global']['hbar'], 5)

        # set new options
        config['projectq.ibm']['device'] = 'ibmqx4'
        self.assertEqual(config['projectq.ibm.device'], 'ibmqx4')

        # set nested dictionaries
        config['strawberryfields.tf'] = {'batched': True, 'cutoff_dim': 6}
        self.assertEqual(config['strawberryfields.tf'], {'batched': True, 'cutoff_dim': 6})

        # set nested keys that don't exist dictionaries
        config['strawberryfields.another.hello.world'] = 5
        self.assertEqual(config['strawberryfields.another'], {'hello': {'world': 5}})

    def test_bool(self):
        """Test boolean value of the Configuration object."""
        self.logTestName()

        # test false if no config is loaded
        config = Configuration('noconfig')
        self.assertFalse(config)

        # test true if config is loaded
        config = Configuration(filename)
        self.assertTrue(config)


class TestPennyLaneInit:
    """Tests to ensure that the code in PennyLane/__init__.py
    correctly knows how to load and use configuration data"""

    def test_device_load(self, default_config):
        """Test loading a device with a configuration."""
        dev = qml.device('default.gaussian', wires=2, config=default_config)

        assert dev.hbar == 2
        assert dev.shots == 0

if __name__ == '__main__':
    print('Testing PennyLane version ' + pennylane.version() + ', Configuration class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest, PennyLaneInitTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)
