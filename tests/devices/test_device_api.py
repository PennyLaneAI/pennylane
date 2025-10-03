# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the basic default behavior of the Device API.
"""
import pytest

import pennylane as qml
from pennylane.devices import Device, ExecutionConfig, MCMConfig
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
)
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch
from pennylane.wires import Wires

# pylint:disable=unused-argument,too-few-public-methods,unused-variable,protected-access,too-many-arguments


def test_execute_method_abstract():
    """Test that a device can't be instantiated without an execute method."""

    # pylint: disable=too-few-public-methods
    class BadDevice(Device):
        """A bad device"""

    with pytest.raises(TypeError, match=r"instantiate abstract class BadDevice"):
        BadDevice()  # pylint: disable=abstract-class-instantiated


EXAMPLE_TOML_FILE = """
schema = 3

[operators.gates]

[operators.observables]

[pennylane.operators.observables]

[measurement_processes]

[pennylane.measurement_processes]

[compilation]

"""

EXAMPLE_TOML_FILE_ONE_SHOT = (
    EXAMPLE_TOML_FILE
    + """
supported_mcm_methods = [ "one-shot" ]
"""
)

EXAMPLE_TOML_FILE_ALL_SUPPORT = (
    EXAMPLE_TOML_FILE
    + """
supported_mcm_methods = [ "device", "one-shot" ]
"""
)


class TestDeviceCapabilities:
    """Tests for the capabilities of a device."""

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_device_capabilities(self, request):
        """Tests that the device capabilities object is correctly initialized"""

        class DeviceWithCapabilities(Device):
            """A device with a capabilities config file defined."""

            config_filepath = request.node.toml_file

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = DeviceWithCapabilities()
        assert isinstance(dev.capabilities, DeviceCapabilities)

    def test_device_invalid_filepath(self):
        """Tests that the device raises an error when the config file does not exist."""

        with pytest.raises(FileNotFoundError):

            class DeviceWithInvalidCapabilities(Device):

                config_filepath = "nonexistent_file.toml"

                def execute(self, circuits, execution_config: ExecutionConfig | None = None):
                    return (0,)


class TestSetupExecutionConfig:
    """Tests the default implementation for setup_execution_config."""

    def test_device_implements_preprocess(self):
        """Tests that the execution config returned by device's preprocess is used."""

        default_execution_config = ExecutionConfig()

        class CustomDevice(Device):

            def preprocess(self, execution_config=None):
                return TransformProgram(), default_execution_config

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = CustomDevice()
        config = dev.setup_execution_config()
        assert config is default_execution_config

    def test_device_no_capabilities(self):
        """Tests if the device does not declare capabilities."""

        class DeviceNoCapabilities(Device):

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = DeviceNoCapabilities()
        config = dev.setup_execution_config()
        assert config == ExecutionConfig()

        DeviceNoCapabilities.supports_derivatives = lambda *_: True
        initial_config = ExecutionConfig(gradient_method="best")
        config = dev.setup_execution_config(initial_config)
        assert config.gradient_method == "device"

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file, mcm_method, shots, expected_transform, expected_error",
        [
            (EXAMPLE_TOML_FILE, "deferred", 10, qml.defer_measurements, None),
            (EXAMPLE_TOML_FILE, "deferred", None, qml.defer_measurements, None),
            (
                EXAMPLE_TOML_FILE,
                "one-shot",
                None,
                None,
                'The "one-shot" MCM method is only supported with finite shots.',
            ),
            (
                EXAMPLE_TOML_FILE,
                "one-shot",
                10,
                None,
                'Requested MCM method "one-shot" unsupported by the device.',
            ),
            (
                EXAMPLE_TOML_FILE_ONE_SHOT,
                "magic",
                10,
                None,
                'Requested MCM method "magic" unsupported by the device.',
            ),
            (
                EXAMPLE_TOML_FILE_ALL_SUPPORT,
                "tree-traversal",
                10,
                None,
                'Requested MCM method "tree-traversal" unsupported by the device.',
            ),
        ],
        indirect=("create_temporary_toml_file",),
    )
    def test_mcm_method_validation(
        self, mcm_method, shots, expected_transform, expected_error, request
    ):
        """Tests that the requested MCM method is validated."""

        class DeviceWithMCM(Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = DeviceWithMCM()
        mcm_config = MCMConfig(mcm_method=mcm_method)
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)
        initial_config = ExecutionConfig(mcm_config=mcm_config)

        if expected_error is not None:
            with pytest.raises(QuantumFunctionError, match=expected_error):
                dev.setup_execution_config(initial_config, tape)
            return

        config = dev.setup_execution_config(initial_config, tape)
        assert config.mcm_config.mcm_method == mcm_method

    @pytest.mark.parametrize(
        "mcm_method, shots, expected_error",
        [
            ("one-shot", None, 'The "one-shot" MCM method is only supported with finite shots.'),
            ("magic", None, 'Requested MCM method "magic" unsupported by the device.'),
            ("one-shot", 100, None),
        ],
    )
    def test_mcm_method_validation_without_capabilities(self, mcm_method, shots, expected_error):
        """Tests that the requested mcm method is validated without device capabilities"""

        class CustomDevice(Device):
            """A device with only a dummy execute method provided."""

            def execute(self, circuits, execution_config: ExecutionConfig | None = None):
                return (0,)

        dev = CustomDevice()
        mcm_config = MCMConfig(mcm_method=mcm_method)
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)
        initial_config = ExecutionConfig(mcm_config=mcm_config)
        if expected_error:
            with pytest.raises(QuantumFunctionError, match=expected_error):
                dev.setup_execution_config(initial_config, tape)
        else:
            final_config = dev.setup_execution_config(initial_config, tape)
            assert final_config.mcm_config.mcm_method == mcm_method

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file, shots, expected_method",
        [
            (EXAMPLE_TOML_FILE, 10, "deferred"),
            (EXAMPLE_TOML_FILE, None, "deferred"),
            (EXAMPLE_TOML_FILE_ONE_SHOT, 10, "one-shot"),
            (EXAMPLE_TOML_FILE_ONE_SHOT, None, "deferred"),
            (EXAMPLE_TOML_FILE_ALL_SUPPORT, 10, "device"),
            (EXAMPLE_TOML_FILE_ALL_SUPPORT, None, "device"),
        ],
        indirect=("create_temporary_toml_file",),
    )
    def test_mcm_method_resolution(self, request, shots, expected_method):
        """Tests that an MCM method is chosen if not specified."""

        class CustomDevice(qml.devices.Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def execute(self, circuit, **kwargs):
                """The execute method for the custom device."""
                return 0

        dev = CustomDevice()
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)
        config = dev.setup_execution_config(ExecutionConfig(), tape)
        assert config.mcm_config.mcm_method == expected_method


class TestPreprocessTransforms:
    """Tests the default implementation for preprocess_transforms."""

    def test_device_implements_preprocess(self):
        """Tests that the execution config returned by device's preprocess is used."""

        default_transform_program = TransformProgram()

        class CustomDevice(Device):

            def preprocess(self, execution_config=None):
                return default_transform_program, ExecutionConfig()

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = CustomDevice()
        program = dev.preprocess_transforms()
        assert program is default_transform_program

    def test_device_no_capabilities(self):
        """Tests if the device does not declare capabilities."""

        class DeviceNoCapabilities(Device):

            def execute(self, circuits, execution_config=None):
                return (0,)

        dev = DeviceNoCapabilities()
        program = dev.preprocess_transforms()
        assert program == TransformProgram()

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file, mcm_method, expected_transform",
        [
            (EXAMPLE_TOML_FILE_ALL_SUPPORT, "one-shot", qml.transforms.dynamic_one_shot),
            (EXAMPLE_TOML_FILE_ALL_SUPPORT, "deferred", qml.transforms.defer_measurements),
            (EXAMPLE_TOML_FILE_ALL_SUPPORT, "device", None),
        ],
        indirect=("create_temporary_toml_file",),
    )
    def test_mcm_transform_in_program(self, mcm_method, expected_transform, request):
        """Tests that the correct MCM transform is included in the program."""

        mcm_transforms = {
            qml.transforms.dynamic_one_shot,
            qml.transforms.defer_measurements,
            qml.devices.preprocess.mid_circuit_measurements,
        }

        class CustomDevice(Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def execute(
                self,
                circuits: QuantumScriptOrBatch,
                execution_config: ExecutionConfig = None,
            ) -> Result | ResultBatch:
                return (0,)

        dev = CustomDevice()
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method=mcm_method))
        transform_program = dev.preprocess_transforms(config)
        if expected_transform:
            assert expected_transform in transform_program
        for other_transform in mcm_transforms - {expected_transform}:
            assert other_transform not in transform_program

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [EXAMPLE_TOML_FILE_ALL_SUPPORT],
        indirect=True,
    )
    @pytest.mark.parametrize("supports_projector", [True, False])
    def test_deferred_allow_postselect(self, request, supports_projector):
        """Tests that the deferred measurements transform validates postselection."""

        class CustomDevice(Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def __init__(self):
                super().__init__()
                if supports_projector:
                    self.capabilities.operations["Projector"] = OperatorProperties()

            def execute(
                self,
                circuits: QuantumScriptOrBatch,
                execution_config: ExecutionConfig = None,
            ) -> Result | ResultBatch:
                return (0,)

        dev = CustomDevice()
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="deferred"))
        program = dev.preprocess_transforms(config)
        tape = QuantumScript([qml.measurements.MidMeasureMP(0, postselect=0)], [], shots=10)

        if not supports_projector:
            with pytest.raises(ValueError, match="Postselection is not allowed on the device"):
                program((tape,))
        else:
            tapes, _ = program((tape,))
            assert isinstance(tapes[0].operations[0], qml.Projector)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    @pytest.mark.parametrize("shots", [None, 10])
    def test_decomposition(self, request, shots):
        """Tests that decomposition acts correctly with or without shots."""

        class CustomDevice(Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def __init__(self):
                super().__init__()
                self.capabilities.operations.update(
                    {
                        "Rot": OperatorProperties(
                            conditions=[ExecutionCondition.ANALYTIC_MODE_ONLY]
                        ),
                        "CNOT": OperatorProperties(),
                        "RY": OperatorProperties(),
                        "RZ": OperatorProperties(),
                    }
                )

            def execute(
                self,
                circuits: QuantumScriptOrBatch,
                execution_config: ExecutionConfig = None,
            ) -> Result | ResultBatch:
                return (0,)

        dev = CustomDevice()
        program = dev.preprocess_transforms()
        tape = QuantumScript([qml.Rot(0.1, 0.2, 0.3, wires=0)], shots=shots)
        tapes, _ = program((tape,))
        if shots:
            assert tapes[0].operations == [
                qml.RZ(0.1, wires=0),
                qml.RY(0.2, wires=0),
                qml.RZ(0.3, wires=0),
            ]
        else:
            assert tapes[0].operations == [qml.Rot(0.1, 0.2, 0.3, wires=0)]

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    @pytest.mark.parametrize("shots", [None, 10])
    def test_validation(self, request, shots):
        """Tests that observable and measurement validation works correctly."""

        class CustomDevice(Device):
            """A device with capabilities config file defined."""

            config_filepath = request.node.toml_file

            def __init__(self):
                super().__init__()
                self.capabilities.observables.update(
                    {
                        "Hadamard": OperatorProperties(
                            conditions=[ExecutionCondition.ANALYTIC_MODE_ONLY]
                        ),
                        "PauliZ": OperatorProperties(),
                        "PauliY": OperatorProperties(
                            conditions=[ExecutionCondition.FINITE_SHOTS_ONLY]
                        ),
                    }
                )
                self.capabilities.measurement_processes.update(
                    {
                        "ExpectationMP": [],
                        "SampleMP": [],
                        "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
                        "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
                    }
                )

            def execute(
                self,
                circuits: QuantumScriptOrBatch,
                execution_config: ExecutionConfig = None,
            ) -> Result | ResultBatch:
                return (0,)

        dev = CustomDevice()
        program = dev.preprocess_transforms()

        valid_tape = QuantumScript([], [qml.expval(qml.Z(0))], shots=shots)
        _, __ = program((valid_tape,))

        invalid_tape = QuantumScript([], [qml.var(qml.PauliZ(0))], shots=shots)
        with pytest.raises(DeviceError, match=r"Measurement var\(Z\(0\)\) not accepted"):
            _, __ = program((invalid_tape,))

        invalid_tape = QuantumScript(
            [], [qml.expval(qml.Hermitian([[1.0, 0], [0, 1.0]], 0))], shots=shots
        )
        with pytest.raises(DeviceError, match=r"Observable Hermitian"):
            _, __ = program((invalid_tape,))

        shots_only_meas_tape = QuantumScript([], [qml.counts()], shots=shots)
        shots_only_obs_tape = QuantumScript([], [qml.expval(qml.Y(0))], shots=shots)
        analytic_only_obs_tape = QuantumScript([], [qml.expval(qml.H(0))], shots=shots)
        analytic_only_meas_tape = QuantumScript([], [qml.state()], shots=shots)

        if shots:

            _, __ = program((shots_only_meas_tape,))
            _, __ = program((shots_only_obs_tape,))

            with pytest.raises(DeviceError, match=r"Measurement .* not accepted"):
                _, __ = program((analytic_only_meas_tape,))

            with pytest.raises(DeviceError, match=r"Observable .* not supported"):
                _, __ = program((analytic_only_obs_tape,))

        else:
            _, __ = program((analytic_only_meas_tape,))
            _, __ = program((analytic_only_obs_tape,))

            with pytest.raises(DeviceError, match=r"Measurement .* not accepted"):
                _, __ = program((shots_only_meas_tape,))

            with pytest.raises(DeviceError, match=r"Observable .* not supported"):
                _, __ = program((shots_only_obs_tape,))

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    @pytest.mark.parametrize("overlapping_obs", [True, False])
    @pytest.mark.parametrize("non_commuting_obs", [True, False])
    @pytest.mark.parametrize("sum_support", [True, False])
    def test_obs_splitting_transform(
        self, request, overlapping_obs, non_commuting_obs, sum_support
    ):
        """Tests if ``split_non_commuting`` or ``split_to_single_terms`` is applied correctly."""

        if not overlapping_obs and non_commuting_obs:
            pytest.skip("Not a valid combination of capabilities")

        if overlapping_obs and sum_support:
            pytest.skip("The support for Sum doesn't matter here")

        class CustomDevice(Device):

            config_filepath = request.node.toml_file

            def __init__(self):
                super().__init__()
                self.capabilities.overlapping_observables = overlapping_obs
                self.capabilities.non_commuting_observables = non_commuting_obs
                if sum_support:
                    self.capabilities.observables.update({"Sum": OperatorProperties()})

            def execute(self, circuits, execution_config: ExecutionConfig | None = None):
                return (0,)

        dev = CustomDevice()
        program = dev.preprocess_transforms()

        if not overlapping_obs:
            assert qml.transforms.split_non_commuting in program
            assert qml.transforms.split_to_single_terms not in program
            for transform_container in program:
                if transform_container._transform_dispatcher is qml.transforms.split_non_commuting:
                    assert "grouping_strategy" in transform_container._kwargs
                    assert transform_container._kwargs["grouping_strategy"] == "wires"
        elif not non_commuting_obs:
            assert qml.transforms.split_non_commuting in program
            assert qml.transforms.split_to_single_terms not in program
            for transform_container in program:
                if transform_container._transform_dispatcher is qml.transforms.split_non_commuting:
                    assert "grouping_strategy" in transform_container._kwargs
                    assert transform_container._kwargs["grouping_strategy"] == "qwc"
        elif not sum_support:
            assert qml.transforms.split_to_single_terms in program
            assert qml.transforms.split_non_commuting not in program
        else:
            assert qml.transforms.split_to_single_terms not in program
            assert qml.transforms.split_to_single_terms not in program

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    @pytest.mark.parametrize("non_commuting_obs", [True, False])
    @pytest.mark.parametrize("all_obs_support", [True, False])
    def test_diagonalize_measurements(self, request, non_commuting_obs, all_obs_support):
        """Tests that the diagonalize_measurements transform is applied correctly."""

        class CustomDevice(Device):

            config_filepath = request.node.toml_file

            def __init__(self):
                super().__init__()
                self.capabilities.non_commuting_observables = non_commuting_obs
                if all_obs_support:
                    self.capabilities.observables.update(
                        {
                            "PauliX": OperatorProperties(),
                            "PauliY": OperatorProperties(),
                            "PauliZ": OperatorProperties(),
                            "Hadamard": OperatorProperties(),
                        }
                    )
                else:
                    self.capabilities.observables.update(
                        {
                            "PauliZ": OperatorProperties(),
                            "PauliX": OperatorProperties(),
                            "PauliY": OperatorProperties(),
                            "Hermitian": OperatorProperties(),
                        }
                    )

            def execute(self, circuits, execution_config: ExecutionConfig | None = None):
                return (0,)

        dev = CustomDevice()
        program = dev.preprocess_transforms()
        if non_commuting_obs is True:
            assert qml.transforms.diagonalize_measurements not in program
        elif all_obs_support is True:
            assert qml.transforms.diagonalize_measurements not in program
        else:
            assert qml.transforms.diagonalize_measurements in program
            for transform_container in program:
                if (
                    transform_container._transform_dispatcher
                    is qml.transforms.diagonalize_measurements
                ):
                    assert transform_container._kwargs["supported_base_obs"] == {
                        qml.Z,
                        qml.X,
                        qml.Y,
                    }


class TestMinimalDevice:
    """Tests for a device with only a minimal execute provided."""

    class MinimalDevice(Device):
        """A device with only a dummy execute method provided."""

        def execute(self, circuits, execution_config: ExecutionConfig | None = None):
            return (0,)

    dev = MinimalDevice()

    def test_device_name(self):
        """Test the default name is the name of the class"""
        assert self.dev.name == "MinimalDevice"

    def test_no_capabilities(self):
        """Test the default capabilities are empty"""
        assert self.dev.capabilities is None

    @pytest.mark.parametrize(
        "wires,shots,expected",
        [
            (None, None, "<MinimalDevice device at 0x"),
            ([1, 3], None, "<MinimalDevice device (wires=2) at 0x"),
        ],
    )
    def test_repr(self, wires, shots, expected):
        """Tests the repr of the device API"""
        assert repr(self.MinimalDevice(wires=wires, shots=shots)).startswith(expected)

    def test_shots(self):
        """Test default behavior for shots."""

        assert self.dev.shots == qml.measurements.Shots(None)

        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            shots_dev = self.MinimalDevice(shots=100)
        assert shots_dev.shots == qml.measurements.Shots(100)

        with pytest.raises(
            AttributeError, match="Shots can no longer be set on a device instance."
        ):
            self.dev.shots = 100  # pylint: disable=attribute-defined-outside-init

    def test_getattr_error(self):
        """Test that querying a property that doesn't exist informs about interface change."""

        with pytest.raises(
            AttributeError,
            match=r"You may be looking for a property or method present in the legacy device",
        ):
            _ = self.dev.expand_fn

    def test_tracker_set_on_initialization(self):
        """Test that a new tracker instance is initialized with the class."""
        assert isinstance(self.dev.tracker, qml.Tracker)
        assert self.dev.tracker is not self.MinimalDevice.tracker

    def test_preprocess_single_circuit(self):
        """Test that preprocessing wraps a circuit into a batch."""

        circuit1 = qml.tape.QuantumScript()
        program, config = self.dev.preprocess()
        batch, fn = program((circuit1,))
        assert isinstance(batch, tuple)
        assert len(batch) == 1
        assert batch[0] is circuit1
        assert callable(fn)

        a = (1,)
        assert fn(a) == (1,)
        assert config == ExecutionConfig()

    def test_preprocess_batch_circuits(self):
        """Test that preprocessing a batch doesn't do anything."""

        circuit = qml.tape.QuantumScript()
        in_config = ExecutionConfig()
        in_batch = (circuit, circuit)
        program, config = self.dev.preprocess(in_config)
        batch, fn = program(in_batch)
        assert batch is in_batch
        assert config is in_config
        a = (1, 2)
        assert fn(a) is a

    def test_supports_derivatives_default(self):
        """Test that the default behavior of supports derivatives is false."""

        assert not self.dev.supports_derivatives()
        assert not self.dev.supports_derivatives(ExecutionConfig())

    def test_compute_derivatives_notimplemented(self):
        """Test that compute derivatives raises a notimplementederror."""

        with pytest.raises(NotImplementedError):
            self.dev.compute_derivatives(qml.tape.QuantumScript())

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_derivatives(qml.tape.QuantumScript())

    def test_supports_jvp_default(self):
        """Test that the default behaviour of supports_jvp is false."""
        assert not self.dev.supports_jvp()

    def test_compute_jvp_not_implemented(self):
        """Test that compute_jvp is not implemented by default."""
        with pytest.raises(NotImplementedError):
            self.dev.compute_jvp(qml.tape.QuantumScript(), (0.1,))

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (0.1,))

    def test_supports_vjp_default(self):
        """Test that the default behavior of supports_jvp is false."""
        assert not self.dev.supports_vjp()

    def test_compute_vjp_not_implemented(self):
        """Test that compute_vjp is not implemented by default."""
        with pytest.raises(NotImplementedError):
            self.dev.compute_vjp(qml.tape.QuantumScript(), (0.1,))

        with pytest.raises(NotImplementedError):
            self.dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (0.1,))

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (None, None),
            (0, Wires([])),
            (Wires([0]), Wires([0])),
            (1, Wires([0])),
            ([1], Wires([1])),
            (2, Wires([0, 1])),
            ([1, 3], Wires([1, 3])),
        ],
    )
    def test_wires_can_be_provided(self, wires, expected):
        """Test that a device can be created with wires."""
        assert self.MinimalDevice(wires=wires).wires == expected

    def test_wires_are_read_only(self):
        """Test that device wires cannot be set after device initialization."""
        with pytest.raises(AttributeError):
            self.dev.wires = [0, 1]  # pylint:disable=attribute-defined-outside-init


def test_device_with_ambiguous_preprocess():
    """Tests that an error is raised when defining a device with ambiguous preprocess."""

    with pytest.raises(ValueError, match="A device should implement either"):

        class InvalidDevice(Device):
            """A device with ambiguous preprocess."""

            def preprocess(self, execution_config=None):
                return TransformProgram(), ExecutionConfig()

            def setup_execution_config(
                self,
                config: ExecutionConfig | None = None,
                circuit: QuantumScript | None = None,
            ) -> ExecutionConfig:
                return ExecutionConfig()

            def preprocess_transforms(
                self, execution_config: ExecutionConfig | None = None
            ) -> TransformProgram:
                return TransformProgram()

            def execute(self, circuits, execution_config: ExecutionConfig = None):
                return (0,)


class TestProvidingDerivatives:
    """Tests logic when derivatives, vjp, or jvp are overridden."""

    def test_provided_derivative(self):
        """Tests default logic for a device with a derivative provided."""

        class WithDerivative(Device):
            """A device with a derivative."""

            # pylint: disable=unused-argument
            def execute(self, circuits, execution_config: ExecutionConfig = None):
                return "a"

            def compute_derivatives(self, circuits, execution_config: ExecutionConfig = None):
                return ("b",)

        dev = WithDerivative()
        assert dev.supports_derivatives()
        assert not dev.supports_derivatives(ExecutionConfig(derivative_order=2))
        assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
        assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))

        out = dev.execute_and_compute_derivatives(qml.tape.QuantumScript())
        assert out[0] == "a"
        assert out[1] == ("b",)

    def test_provided_jvp(self):
        """Tests default logic for a device with a jvp provided."""

        # pylint: disable=unused-argnument
        class WithJvp(Device):
            """A device with a jvp."""

            def execute(self, circuits, execution_config: ExecutionConfig = None):
                return "a"

            def compute_jvp(self, circuits, tangents, execution_config: ExecutionConfig = None):
                return ("c",)

        dev = WithJvp()
        assert dev.supports_jvp()

        out = dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (1.0,))
        assert out[0] == "a"
        assert out[1] == ("c",)

    def test_provided_vjp(self):
        """Tests default logic for a device with a vjp provided."""

        # pylint: disable=unused-argnument
        class WithVjp(Device):
            """A device with a vjp."""

            def execute(self, circuits, execution_config: ExecutionConfig = None):
                return "a"

            def compute_vjp(
                self,
                circuits,
                cotangents,
                execution_config: ExecutionConfig = None,
            ):
                return ("c",)

        dev = WithVjp()
        assert dev.supports_vjp()

        out = dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (1.0,))
        assert out[0] == "a"
        assert out[1] == ("c",)


@pytest.mark.jax
def test_capture_methods_not_implemented():
    """Test that the eval_jaxpr and jaxpr_jvp methods are not implemented by default."""

    import jax

    def f(x):
        return x + 1

    # pylint: disable=too-few-public-methods
    class NormalDevice(Device):

        def execute(self, circuits, execution_config=None):
            return 0

    jaxpr = jax.make_jaxpr(f)(2)
    with pytest.raises(NotImplementedError):
        NormalDevice().eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3, execution_config=None)

    with pytest.raises(NotImplementedError):
        NormalDevice().jaxpr_jvp(jaxpr.jaxpr, (3,), (0,), execution_config=None)
