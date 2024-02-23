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
"""Tests that a device has the right attributes, arguments and methods."""
# pylint: disable=no-self-use
import pytest
import pennylane.numpy as pnp
import pennylane as qml


try:
    import tensorflow as tf

    TF_SUPPORT = True

except ImportError:
    TF_SUPPORT = False

try:
    import torch

    TORCH_SUPPORT = True

except ImportError:
    TORCH_SUPPORT = False

try:
    import jax

    JAX_SUPPORT = True

except ImportError:
    JAX_SUPPORT = False

# Shared test data =====


def qfunc_with_scalar_input(model=None):
    """Model dependent quantum function taking a single input"""

    def qfunc(x):
        if model == "qubit":
            qml.RX(x, wires=0)
        elif model == "cv":
            qml.Displacement(x, 0.0, wires=0)
        return qml.expval(qml.Identity(wires=0))

    return qfunc


# =======================


class TestDeviceProperties:
    """Test the device is created with the expected properties."""

    def test_load_device(self, device_kwargs):
        """Test that the device loads correctly."""
        device_kwargs["wires"] = 2
        device_kwargs["shots"] = 1234

        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            assert isinstance(dev.wires, qml.wires.Wires)
            assert dev.wires == qml.wires.Wires((0, 1))

            assert isinstance(dev.shots, qml.measurements.Shots)
            assert dev.shots == qml.measurements.Shots(1234)

            assert device_kwargs["name"] == dev.name
            assert isinstance(dev.tracker, qml.Tracker)
            return

        assert dev.num_wires == 2
        assert dev.shots == 1234
        assert dev.short_name == device_kwargs["name"]

    def test_no_wires_given(self, device_kwargs):
        """Test that the device requires correct arguments."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            dev = qml.device(**device_kwargs)
            if isinstance(dev, qml.devices.Device):
                pytest.skip("test is old interface specific.")

    def test_no_0_shots(self, device_kwargs):
        """Test that non-analytic devices cannot accept 0 shots."""
        # first create a valid device to extract its capabilities
        device_kwargs["wires"] = 2
        device_kwargs["shots"] = 0

        with pytest.raises(Exception):  # different types of error based on interface
            dev = qml.device(**device_kwargs)
            if isinstance(dev, qml.devices.Device):
                pytest.skip("test is old interface specific.")


class TestCapabilities:
    """Test that the device declares its capabilities correctly."""

    def test_has_capabilities_dictionary(self, device_kwargs):
        """Test that the device class has a capabilities() method returning a dictionary."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()
        assert isinstance(cap, dict)

    def test_model_is_defined_valid_and_correct(self, device_kwargs):
        """Test that the capabilities dictionary defines a valid model."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()
        assert "model" in cap
        assert cap["model"] in ["qubit", "cv"]

        if cap["model"] == "qubit":

            @qml.qnode(dev)
            def circuit():
                qml.X(0)
                return qml.expval(qml.Z(0))

        else:

            @qml.qnode(dev)
            def circuit():
                qml.Displacement(1.0, 1.2345, wires=0)
                return qml.expval(qml.QuadX(wires=0))

        # assert that device can measure observable from its model
        circuit()

    def test_passthru_interface_is_correct(self, device_kwargs):
        """Test that the capabilities dictionary defines a valid passthru interface, if not None."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()

        if "passthru_interface" not in cap:
            pytest.skip("No passthru_interface capability specified by device.")

        interface = cap["passthru_interface"]
        assert interface in ["tf", "autograd", "jax", "torch"]  # for new interface, add test case

        qfunc = qfunc_with_scalar_input(cap["model"])
        qnode = qml.QNode(qfunc, dev, interface=interface)

        # assert that we can do a simple gradient computation in the passthru interface
        # without raising an error

        if interface == "tf":
            if TF_SUPPORT:
                x = tf.Variable(0.1)
                with tf.GradientTape() as tape:
                    res = qnode(x)
                    tape.gradient(res, [x])
            else:
                pytest.skip("Cannot import tensorflow.")

        if interface == "autograd":
            x = pnp.array(0.1, requires_grad=True)
            g = qml.grad(qnode)
            g(x)

        if interface == "jax":
            if JAX_SUPPORT:
                x = pnp.array(0.1, requires_grad=True)
                g = jax.grad(lambda a: qnode(a).reshape(()))
                g(x)
            else:
                pytest.skip("Cannot import jax")

        if interface == "torch":
            if TORCH_SUPPORT:
                x = torch.tensor(0.1, requires_grad=True)
                res = qnode(x)
                res.backward()
                assert hasattr(x, "grad")
            else:
                pytest.skip("Cannot import torch")

    def test_supports_tensor_observables(self, device_kwargs):
        """Tests that the device reports correctly whether it supports tensor observables."""
        device_kwargs["wires"] = 2
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()

        if "supports_tensor_observables" not in cap:
            pytest.skip("No supports_tensor_observables capability specified by device.")

        @qml.qnode(dev)
        def circuit():
            """Model agnostic quantum function with tensor observable"""
            if cap["model"] == "qubit":
                qml.X(0)
            else:
                qml.QuadX(wires=0)
            return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))

        if cap["supports_tensor_observables"]:
            circuit()
        else:
            with pytest.raises(qml.QuantumFunctionError):
                circuit()

    def test_returns_state(self, device_kwargs):
        """Tests that the device reports correctly whether it supports returning the state."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.state()

        if not cap.get("returns_state"):
            # If the device is not defined to return state then the
            # access_state method should raise
            with pytest.raises(qml.QuantumFunctionError):
                dev.access_state()

            try:
                state = dev.state
            except (AttributeError, NotImplementedError):
                state = None

            assert state is None
        else:
            if dev.shots is not None:
                with pytest.warns(
                    UserWarning,
                    match="Requested state or density matrix with finite shots; the returned",
                ):
                    circuit()
            else:
                circuit()

            assert dev.state is not None

    def test_returns_probs(self, device_kwargs):
        """Tests that the device reports correctly whether it supports reversible differentiation."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()

        if "returns_probs" not in cap:
            pytest.skip("No returns_probs capability specified by device.")

        @qml.qnode(dev)
        def circuit():
            if cap["model"] == "qubit":
                qml.X(0)
            else:
                qml.QuadX(wires=0)
            return qml.probs(wires=0)

        if cap["returns_probs"]:
            circuit()
        else:
            with pytest.raises(NotImplementedError):
                circuit()

    def test_supports_broadcasting(self, device_kwargs, mocker):
        """Tests that the device reports correctly whether it supports parameter broadcasting
        and that it can execute broadcasted tapes in any case."""

        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        if isinstance(dev, qml.devices.Device):
            pytest.skip("test is old interface specific.")
        cap = dev.capabilities()

        assert "supports_broadcasting" in cap

        @qml.qnode(dev)
        def circuit(x):
            if cap["model"] == "qubit":
                qml.RX(x, wires=0)
            else:
                qml.Rotation(x, wires=0)
            return qml.probs(wires=0)

        spy = mocker.spy(qml.transforms, "broadcast_expand")
        circuit(0.5)
        if cap.get("returns_state"):
            orig_shape = pnp.array(dev.access_state()).shape
        spy.assert_not_called()
        x = pnp.array([0.5, 2.1, -0.6], requires_grad=True)

        if cap["supports_broadcasting"]:
            res = circuit(x)
            spy.assert_not_called()
            if cap.get("returns_state"):
                assert pnp.array(dev.access_state()).shape != orig_shape
        else:
            res = circuit(x)
            spy.assert_called()
            if cap.get("returns_state"):
                assert pnp.array(dev.access_state()).shape == orig_shape

        assert pnp.ndim(res) == 2
        assert res.shape[0] == 3  # pylint:disable=unsubscriptable-object
