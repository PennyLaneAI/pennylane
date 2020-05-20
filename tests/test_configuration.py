# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
import pytest
import os
import logging as log
import sys

import toml

import pennylane as qml
from pennylane import Configuration


log.getLogger('defaults')


config_filename = "default_config.toml"


test_config = """\
[main]
shots = 1000

[default.gaussian]
hbar = 2

[strawberryfields.global]
hbar = 1
shots = 1000
analytic = true

    [strawberryfields.fock]
    cutoff_dim = 10

    [strawberryfields.gaussian]
    shots = 1000
    hbar = 1

[qiskit.global]
backend = "qasm_simulator"

    [qiskit.aer]
    backend = "unitary_simulator"
    backend_options = {"validation_threshold" = 1e-6}

    [qiskit.ibmq]
    ibmqx_token = "XXX"
    backend = "ibmq_rome"
    hub = "MYHUB"
    group = "MYGROUP"
    project = "MYPROJECT"
"""


@pytest.fixture(scope="function")
def default_config(tmpdir):
    config_path = os.path.join(tmpdir, config_filename)

    with open(config_path, "w") as f:
        f.write(test_config)

    return Configuration(name=config_path)


@pytest.fixture(scope="function")
def default_config_toml(tmpdir):
    config_path = os.path.join(tmpdir, config_filename)

    with open(config_path, "w") as f:
        f.write(test_config)

    return toml.load(config_path), config_path


class TestConfigurationFileInteraction:
    """Test the interaction with the configuration file."""

    def test_loading_current_directory(self, monkeypatch, default_config_toml):
        """Test that the default configuration file can be loaded
        from the current directory."""
        config_toml, config_path = default_config_toml

        monkeypatch.chdir(".")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        config = Configuration(name=config_path)

        assert config.path == os.path.join(os.curdir, config_path)
        assert config._config == config_toml

    def test_loading_environment_variable(self, monkeypatch, default_config_toml):
        """Test that the default configuration file can be loaded
        from an environment variable."""
        config_toml, config_path = default_config_toml

        os.curdir = "None"
        monkeypatch.setenv("PENNYLANE_CONF", os.getcwd())

        config = Configuration(name=config_path)

        assert config._config == config_toml
        assert config._env_config_dir == os.environ["PENNYLANE_CONF"]
        assert config.path == os.path.join(os.environ["PENNYLANE_CONF"], config_path)

    def test_loading_absolute_path(self, monkeypatch, default_config_toml):
        """Test that the default configuration file can be loaded
        from an absolute path."""
        config_toml, config_path = default_config_toml

        os.curdir = "None"
        monkeypatch.setenv("PENNYLANE_CONF", "")

        config = Configuration(name=os.path.join(os.getcwd(), config_path))

        assert config._config == config_toml
        assert config.path == os.path.join(os.getcwd(), config_path)

    def test_not_found_warning(self, caplog):
        """Test that a warning is raised if no configuration file found."""

        caplog.clear()
        caplog.set_level(log.INFO)

        Configuration("noconfig")

        assert len(caplog.records) == 1
        assert caplog.records[0].message == "No PennyLane configuration file found."

    def test_save(self, tmp_path):
        """Test saving a configuration file."""
        config = Configuration(name=config_filename)

        # make a change
        config['strawberryfields.global']['shots'] = 10

        temp_config_path = tmp_path / 'test_config.toml'
        config.save(temp_config_path)

        result = toml.load(temp_config_path)
        config._config == result

class TestProperties:
    """Test that the configuration class works as expected"""

    def test_get_item(self, default_config):
        """Test getting items."""
        # get existing options
        assert default_config['main.shots'] == 1000
        assert default_config['main']['shots'] == 1000
        assert default_config['strawberryfields.global.hbar'] == 1
        assert default_config['strawberryfields.global']['hbar'] == 1

        # get nested dictionaries
        assert default_config['strawberryfields.fock'] == {'cutoff_dim': 10}

        # get key that doesn't exist
        assert default_config['qiskit.ibmq.idonotexist'] == {}

    def test_set_item(self, default_config):
        """Test setting items."""

        # set existing options
        default_config['main.shots'] = 10
        assert default_config['main.shots'] == 10
        assert default_config['main']['shots'] == 10

        default_config['strawberryfields.global']['hbar'] = 5
        assert default_config['strawberryfields.global.hbar'] == 5
        assert default_config['strawberryfields.global']['hbar'] == 5

        # set new options
        default_config['qiskit.ibmq']['backend'] = 'ibmq_rome'
        assert default_config['qiskit.ibmq.backend'] == 'ibmq_rome'

        # set nested dictionaries
        default_config['strawberryfields.tf'] = {'batched': True, 'cutoff_dim': 6}
        assert default_config['strawberryfields.tf'] == {'batched': True, 'cutoff_dim': 6}

        # set nested keys that don't exist dictionaries
        default_config['strawberryfields.another.hello.world'] = 5
        assert default_config['strawberryfields.another'] == {'hello': {'world': 5}}

    def test_bool(self, default_config):
        """Test boolean value of the Configuration object."""

        # test false if no config is loaded
        config = Configuration('noconfig')

        assert not config
        assert default_config

class TestPennyLaneInit:
    """Tests to ensure that the code in PennyLane/__init__.py
    correctly knows how to load and use configuration data"""

    def test_device_load(self, default_config):
        """Test loading a device with a configuration."""
        dev = qml.device('default.gaussian', wires=2, config=default_config)

        assert dev.hbar == 2
        assert dev.shots == 1000
