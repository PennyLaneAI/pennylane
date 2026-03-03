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
import os

import pytest
import tomlkit as toml

import pennylane as qml
from pennylane import Configuration

config_filename = "default_config.toml"


test_config = """\
[default.gaussian]
hbar = 2

[strawberryfields.global]
hbar = 1

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


def load_toml_file(file_path: str) -> dict:
    """Loads a TOML file and returns the parsed dict."""
    with open(file_path, encoding="utf-8") as file:
        return toml.load(file)


@pytest.fixture(scope="function", name="default_config")
def default_config_fixture(tmpdir):
    config_path = os.path.join(tmpdir, config_filename)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(test_config)

    return Configuration(name=config_path)


@pytest.fixture(scope="function", name="default_config_toml")
def default_config_toml_fixture(tmpdir):
    config_path = os.path.join(tmpdir, config_filename)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(test_config)

    return load_toml_file(config_path), config_path


class TestConfigurationFileInteraction:
    """Test the interaction with the configuration file."""

    # pylint: disable=protected-access
    def test_loading_current_directory(self, monkeypatch, default_config_toml):
        """Test that the default configuration file can be loaded
        from the current directory."""
        config_toml, config_path = default_config_toml

        monkeypatch.chdir(".")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        config = Configuration(name=config_path)

        assert config.path == os.path.join(os.curdir, config_path)
        assert config._config == config_toml

    # pylint: disable=protected-access
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

    # pylint: disable=protected-access
    def test_loading_absolute_path(self, monkeypatch, default_config_toml):
        """Test that the default configuration file can be loaded
        from an absolute path."""
        config_toml, config_path = default_config_toml

        os.curdir = "None"
        monkeypatch.setenv("PENNYLANE_CONF", "")

        config = Configuration(name=os.path.join(os.getcwd(), config_path))

        assert config._config == config_toml
        assert config.path == os.path.join(os.getcwd(), config_path)

    # pylint: disable=protected-access
    def test_save(self, tmp_path):
        """Test saving a configuration file."""
        config = Configuration(name=config_filename)

        # make a change
        config["strawberryfields.global"]["shots"] = 10

        temp_config_path = tmp_path / "test_config.toml"
        config.save(temp_config_path)

        result = load_toml_file(temp_config_path)
        assert config._config == result


class TestProperties:
    """Test that the configuration class works as expected"""

    def test_get_item(self, default_config):
        """Test getting items."""
        # get existing options
        assert default_config["strawberryfields.global.hbar"] == 1
        assert default_config["strawberryfields.global"]["hbar"] == 1

        # get nested dictionaries
        assert default_config["strawberryfields.fock"] == {"cutoff_dim": 10}

        # get key that doesn't exist
        assert default_config["qiskit.ibmq.idonotexist"] == {}

    def test_set_item(self, default_config):
        """Test setting items."""

        # set existing options
        default_config["main.shots"] = 10
        assert default_config["main.shots"] == 10
        assert default_config["main"]["shots"] == 10

        default_config["strawberryfields.global"]["hbar"] = 5
        assert default_config["strawberryfields.global.hbar"] == 5
        assert default_config["strawberryfields.global"]["hbar"] == 5

        # set new options
        default_config["qiskit.ibmq"]["backend"] = "ibmq_rome"
        assert default_config["qiskit.ibmq.backend"] == "ibmq_rome"

        # set nested dictionaries
        default_config["strawberryfields.tf"] = {"batched": True, "cutoff_dim": 6}
        assert default_config["strawberryfields.tf"] == {"batched": True, "cutoff_dim": 6}

        # set nested keys that don't exist dictionaries
        default_config["strawberryfields.another.hello.world"] = 5
        assert default_config["strawberryfields.another"] == {"hello": {"world": 5}}

    def test_bool(self, default_config):
        """Test boolean value of the Configuration object."""

        # test false if no config is loaded
        config = Configuration("noconfig")

        assert not config
        assert default_config

    def test_str(self):
        """Test string value of the Configuration object."""
        config = Configuration("noconfig")

        assert str(config) == ""

    def test_str_loaded_config(self, monkeypatch, default_config_toml):
        """Test string value of the Configuration object that has been
        loaded."""
        config_toml, config_path = default_config_toml

        monkeypatch.chdir(".")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        config = Configuration(name=config_path)

        assert str(config) == f"{config_toml}"

    def test_repr(self):
        """Test repr value of the Configuration object."""
        path = "noconfig"
        config = Configuration(path)

        assert repr(config) == "PennyLane Configuration <noconfig>"


# pylint: disable=too-few-public-methods
class TestPennyLaneInit:
    """Tests to ensure that the code in PennyLane/__init__.py
    correctly knows how to load and use configuration data"""

    def test_device_load(self, default_config):
        """Test loading a device with a configuration."""
        dev = qml.device("default.gaussian", wires=2, config=default_config)

        assert dev.hbar == 2
        assert not dev.shots
