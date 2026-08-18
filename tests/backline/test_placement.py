# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for backline placements and transports."""

from types import SimpleNamespace
from typing import Any

import pytest

import pennylane as qp
from pennylane.backline import (
    Controller,
    Coprocessor,
    CoprocessorFunction,
    Node,
    Placement,
    Transport,
)

UNKNOWN_HARDWARE: Any = "tpu"
ENDPOINT = qp.Endpoint("192.168.1.3", 18590)
backline_module = qp.backline


def test_nodes_default_to_cpu_hardware():
    """Controllers and coprocessors execute on CPUs by default."""
    controller = qp.Controller()
    coprocessor = qp.Coprocessor(coprocessor_fn="decoder", endpoint=qp.Endpoint("127.0.0.1", 7760))

    assert controller.hardware == "cpu"
    assert coprocessor.hardware == "cpu"


def test_controller_owns_message_sizes():
    """Controllers provide transport message sizes without backend initialization arguments."""
    controller = qp.Controller()

    assert controller.in_bytes == 8
    assert controller.out_bytes == 8


@pytest.mark.parametrize("name", ["in_bytes", "out_bytes"])
def test_controller_message_sizes_are_not_constructor_arguments(name):
    """Message sizes are fixed on the instance and cannot be passed to the constructor."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        qp.Controller(**{name: 16})


def test_controller_hides_message_sizes_from_repr():
    """Message sizes exist on the instance but are omitted from repr."""
    text = repr(qp.Controller())

    assert "in_bytes" not in text
    assert "out_bytes" not in text


def test_memcpy_coprocessor_allows_missing_endpoint():
    """Memcpy placements do not require a network endpoint on the coprocessor."""
    controller = qp.Controller()
    coprocessor = qp.Coprocessor(coprocessor_fn="decoder")

    dev = qp.Backline(controller=controller, coprocessors=[coprocessor], transport="memcpy")

    assert dev.placement.coprocessors[0].endpoint is None


def test_rdma_coprocessor_requires_endpoint():
    """RDMA placements require every coprocessor to expose a connection endpoint."""
    controller = qp.Controller()
    coprocessor = qp.Coprocessor(coprocessor_fn="decoder")

    with pytest.raises(
        ValueError, match="transport='rdma' requires every coprocessor to set endpoint"
    ):
        qp.Backline(controller=controller, coprocessors=[coprocessor], transport="rdma")


def test_coprocessor_stores_endpoint():
    """A coprocessor keeps the endpoint it was constructed with."""
    endpoint = qp.Endpoint("192.0.2.11", 7760)
    coprocessor = qp.Coprocessor(coprocessor_fn="decoder", endpoint=endpoint)

    assert coprocessor.endpoint is endpoint
    assert coprocessor.endpoint.host == "192.0.2.11"
    assert coprocessor.endpoint.port == 7760


def test_endpoint_requires_a_host():
    """An endpoint host must be a non-empty string."""
    with pytest.raises(TypeError, match="host must be a str"):
        qp.Endpoint(None, 7760)
    with pytest.raises(ValueError, match="host must be a non-empty str"):
        qp.Endpoint("", 7760)


def test_endpoint_requires_a_port():
    """An endpoint requires a port."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'port'"):
        qp.Endpoint("127.0.0.1")


@pytest.mark.parametrize("port", [7.5, "7760"])
def test_endpoint_rejects_non_int_port(port):
    """An endpoint port must be an int."""
    with pytest.raises(TypeError, match="port must be an int"):
        qp.Endpoint("127.0.0.1", port)


@pytest.mark.parametrize("port", [0, -8, 65536])
def test_endpoint_rejects_port_outside_range(port):
    """An endpoint port must be a valid TCP port."""
    with pytest.raises(ValueError, match="port must be in 1..65535"):
        qp.Endpoint("127.0.0.1", port)


@pytest.mark.parametrize("hardware", ["cpu", "gpu", "fpga"])
def test_node_accepts_supported_hardware(hardware):
    """Every public hardware kind can be represented by a node."""
    assert qp.Controller(hardware=hardware).hardware == hardware


@pytest.mark.parametrize(
    "node",
    [
        lambda: qp.Controller(hardware=UNKNOWN_HARDWARE),
        lambda: qp.Coprocessor(
            coprocessor_fn="decoder",
            endpoint=qp.Endpoint("127.0.0.1", 7760),
            hardware=UNKNOWN_HARDWARE,
        ),
    ],
)
def test_node_rejects_unknown_hardware(node):
    """Both node types validate their hardware."""
    with pytest.raises(ValueError, match="hardware must be one of"):
        node()


class TestNodes:
    """Tests for the Node hierarchy."""

    def test_controller_is_node(self):
        """A Controller is a Node carrying the device the QNode runs on."""
        ctrl = qp.Controller(name="controller-name")
        assert isinstance(ctrl, (Controller, Node))
        assert ctrl.name == "controller-name"
        assert ctrl.remote is False

    def test_controller_defaults_to_null_qubit(self):
        """A Controller with no device defaults to null.qubit."""
        ctrl = qp.Controller(name="fpga")
        assert ctrl.device.name == "null.qubit"

    def test_controller_device_override(self):
        """The controller's device can be set to any PennyLane device."""
        dev = qp.device("default.qubit", wires=2)
        ctrl = qp.Controller(device=dev, name="fpga")
        assert ctrl.device is dev

    def test_coprocessor_string_fn_normalized(self):
        """A string coprocessor_fn is normalized to a CoprocessorFunction."""
        cop = qp.Coprocessor(name="gpu-decoder", coprocessor_fn="decoder-XX", endpoint=ENDPOINT)
        assert isinstance(cop, Coprocessor)
        assert isinstance(cop.coprocessor_fn, CoprocessorFunction)
        assert cop.coprocessor_fn.name == "decoder-XX"
        assert cop.coprocessor_fn.symbol_name == "decoder-XX"

    def test_coprocessor_function_passthrough(self):
        """An existing CoprocessorFunction is stored as-is."""
        fn = CoprocessorFunction("decode", lib_path="/opt/lib/libdecode.so")
        cop = qp.Coprocessor(name="gpu", coprocessor_fn=fn, endpoint=ENDPOINT)
        assert cop.coprocessor_fn is fn

    def test_node_frozen(self):
        """Nodes are immutable."""
        ctrl = qp.Controller(name="fpga")
        with pytest.raises(AttributeError):
            ctrl.name = "other"


class TestCoprocessorEndpoint:
    """endpoint belongs to the coprocessor, which owns the connection address."""

    def test_coprocessor_carries_the_endpoint(self):
        """A coprocessor holds the address the controller dials."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", endpoint=ENDPOINT)
        assert cop.endpoint.host == "192.168.1.3"
        assert cop.endpoint.port == 18590

    def test_endpoint_is_optional_on_the_coprocessor(self):
        """endpoint may be omitted; RDMA placements reject that later."""
        assert qp.Coprocessor(coprocessor_fn="decoder").endpoint is None

    def test_controller_has_no_endpoint(self):
        """The controller dials the coprocessor's endpoint; it has none of its own."""
        assert not hasattr(qp.Controller(name="fpga"), "endpoint")
        assert not hasattr(Node(), "endpoint")

    @pytest.mark.parametrize("attr", ["comm_host", "oob_port", "addr"])
    def test_old_endpoint_names_are_gone(self, attr):
        """comm_host/oob_port/addr were replaced by Endpoint."""
        assert not hasattr(qp.Coprocessor(coprocessor_fn="decoder", endpoint=ENDPOINT), attr)
        assert not hasattr(Node(), attr)


class TestNameAndHardware:
    """name identifies the node; hardware selects where it runs."""

    def test_name_and_hardware_are_separate_fields(self):
        """Identity and hardware selection are separate fields."""
        cop = qp.Coprocessor(
            coprocessor_fn="decoder",
            endpoint=qp.Endpoint("127.0.0.1", 7760),
            name="decoder-0",
            hardware="gpu",
        )
        assert cop.name == "decoder-0"
        assert cop.hardware == "gpu"

    def test_name_defaults_to_none(self):
        """Omitted, the compiler derives one from the node's role."""
        cop = qp.Coprocessor(coprocessor_fn="d", endpoint=qp.Endpoint("127.0.0.1", 7760))
        assert cop.name is None

    def test_backend_lib_override_lives_in_init_args(self):
        """An explicit library path stays available as an escape hatch."""
        cop = qp.Coprocessor(
            coprocessor_fn="decoder",
            endpoint=qp.Endpoint("127.0.0.1", 7760),
            hardware="cpu",
            init_args={"backend_lib": "/opt/catalyst/libcustom.so"},
        )
        assert cop.hardware == "cpu"
        assert cop.init_args["backend_lib"] == "/opt/catalyst/libcustom.so"


class TestRemoteDefault:
    """Nodes are local by default; remote is opt-in."""

    def test_controller_is_local_by_default(self):
        """remote defaults to False, matching every other PennyLane device."""
        assert qp.Controller(name="fpga").remote is False

    def test_coprocessor_is_local_by_default(self):
        cop = qp.Coprocessor(coprocessor_fn="d", endpoint=qp.Endpoint("127.0.0.1", 7760))
        assert cop.remote is False

    def test_remote_is_opt_in(self):
        assert qp.Controller(name="fpga", remote=True).remote is True


class TestTripleRemoved:
    """triple lives on the executor, which detects it on the target host."""

    @pytest.mark.parametrize(
        "node",
        [Node(), Coprocessor(coprocessor_fn="decoder", endpoint=ENDPOINT)],
        ids=["Node", "Coprocessor"],
    )
    def test_no_triple_field(self, node):
        """Nodes no longer carry a cross-compilation triple."""
        assert not hasattr(node, "triple")

    def test_controller_has_no_triple(self):
        """The controller's triple comes from its executor, not the node."""
        assert not hasattr(qp.Controller(name="fpga"), "triple")

    def test_triple_rejected_as_kwarg(self):
        """Passing triple= now raises rather than being silently accepted."""
        with pytest.raises(TypeError):
            qp.Controller(name="fpga", triple="aarch64-unknown-linux-gnu")


class TestPlacement:
    """Tests for the Placement container."""

    def test_placement_construction(self):
        """Placement groups a controller, coprocessors, and a transport."""
        ctrl = qp.Controller(name="controller-name")
        cop = qp.Coprocessor(
            name="gpu-decoder",
            coprocessor_fn="decoder-XX",
            endpoint=ENDPOINT,
        )
        placement = Placement(controller=ctrl, coprocessors=(cop,), transport="rdma")
        assert placement.controller is ctrl
        assert placement.coprocessors == (cop,)
        assert placement.transport == Transport("rdma")

    def test_transport_name_resolved(self):
        """A transport name is resolved to a Transport on construction."""
        ctrl = qp.Controller(name="fpga")
        placement = Placement(controller=ctrl, transport="rdma")
        assert isinstance(placement.transport, Transport)
        assert placement.transport.name == "rdma"

    def test_coprocessors_coerced_to_tuple(self):
        """A list of coprocessors is stored as a tuple."""
        ctrl = qp.Controller(name="fpga")
        cop = qp.Coprocessor(name="gpu", coprocessor_fn="decode", endpoint=ENDPOINT)
        placement = Placement(controller=ctrl, coprocessors=[cop], transport="rdma")
        assert isinstance(placement.coprocessors, tuple)

    def test_unknown_transport_raises(self):
        """An unregistered transport name is rejected at construction."""
        ctrl = qp.Controller(name="fpga")
        with pytest.raises(ValueError, match="unknown transport"):
            Placement(controller=ctrl, transport="does-not-exist")

    def test_not_exported_at_top_level(self):
        """Placement is reached via pennylane.backline, not the top-level namespace."""
        assert not hasattr(qp, "Placement")
        assert "Placement" in backline_module.__all__


class TestSingleConstructionSurface:
    """The classes are the only way to build nodes; the lowercase factories are gone."""

    @pytest.mark.parametrize("name", ["controller", "coprocessor"])
    def test_lowercase_factories_removed(self, name):
        """The duplicate lowercase factories are no longer exported."""
        assert not hasattr(qp, name)
        assert not hasattr(backline_module, name)
        assert name not in backline_module.__all__

    def test_executor_options_is_the_declarative_input(self):
        """Backend-specific executor options are passed as a plain dict."""
        ctrl = qp.Controller(name="fpga", executor_options={"threads": 4})
        assert ctrl.executor_options == {"threads": 4}
        assert ctrl.executor is None

    def test_no_executor_requested_by_default(self):
        """executor_options defaults to None, meaning "launch nothing"."""
        assert qp.Controller(name="fpga").executor_options is None

    def test_empty_options_still_requests_an_executor(self):
        """``{}`` means "launch with all defaults" and must stay distinct from ``None``."""
        ctrl = qp.Controller(name="fpga", executor_options={})
        assert ctrl.executor_options == {}
        assert ctrl.executor_options is not None

    def test_prelaunched_executor_can_be_attached(self):
        """A already-launched executor can be set directly, bypassing executor_options."""
        ex = SimpleNamespace(address="10.0.0.5:1373", triple="aarch64-unknown-linux-gnu")
        ctrl = qp.Controller(name="fpga", executor=ex)
        assert ctrl.executor is ex

    def test_executor_spec_class_is_gone(self):
        """ExecutorSpec was replaced by the executor_options dict."""
        assert not hasattr(qp, "ExecutorSpec")
        assert not hasattr(backline_module, "ExecutorSpec")
        assert "ExecutorSpec" not in backline_module.__all__

    def test_unknown_kwarg_raises(self):
        """Typos raise instead of being silently swallowed as executor options."""
        with pytest.raises(TypeError):
            qp.Controller(nmae="typo")

    @pytest.mark.parametrize(
        "call",
        [
            # pylint: disable=too-many-function-args,missing-kwoa
            lambda: qp.Controller(qp.device("default.qubit", wires=2)),
            lambda: qp.Coprocessor("decoder", endpoint=ENDPOINT),
        ],
    )
    def test_nodes_are_keyword_only(self, call):
        """Controller and Coprocessor both reject positional arguments."""
        with pytest.raises(TypeError):
            call()
