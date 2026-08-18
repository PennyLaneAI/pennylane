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

from typing import Any

import pytest

import pennylane as qp

UNKNOWN_HARDWARE: Any = "tpu"


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
from pennylane.backline import (
    Controller,
    Coprocessor,
    CoprocessorFunction,
    Node,
    Placement,
    Transport,
)

backline_module = qp.backline


class TestNodes:
    """Tests for the Node hierarchy."""

    def test_controller_is_node(self):
        """A Controller is a Node carrying the device the QNode runs on."""
        ctrl = qp.Controller(label="controller-label")
        assert isinstance(ctrl, (Controller, Node))
        assert ctrl.label == "controller-label"
        assert ctrl.remote is False

    def test_controller_defaults_to_null_qubit(self):
        """A Controller with no device defaults to null.qubit."""
        ctrl = qp.Controller(label="fpga")
        assert ctrl.device.name == "null.qubit"

    def test_controller_device_override(self):
        """The controller's device can be set to any PennyLane device."""
        dev = qp.device("default.qubit", wires=2)
        ctrl = qp.Controller(device=dev, label="fpga")
        assert ctrl.device is dev

    def test_coprocessor_string_fn_normalized(self):
        """A string coprocessor_fn is normalized to a CoprocessorFunction."""
        cop = qp.Coprocessor(
            label="gpu-libibverbs", coprocessor_fn="decoder-XX", comm_host="192.168.1.3"
        )
        assert isinstance(cop, Coprocessor)
        assert isinstance(cop.coprocessor_fn, CoprocessorFunction)
        assert cop.coprocessor_fn.name == "decoder-XX"
        assert cop.coprocessor_fn.symbol_name == "decoder-XX"

    def test_coprocessor_function_passthrough(self):
        """An existing CoprocessorFunction is stored as-is."""
        fn = CoprocessorFunction("decode", lib_path="/opt/lib/libdecode.so")
        cop = qp.Coprocessor(label="gpu", coprocessor_fn=fn, comm_host="192.168.1.3")
        assert cop.coprocessor_fn is fn

    def test_node_frozen(self):
        """Nodes are immutable."""
        ctrl = qp.Controller(label="fpga")
        with pytest.raises(AttributeError):
            ctrl.label = "other"


class TestCoprocessorEndpoint:
    """comm_host/oob_port belong to the coprocessor, which owns the connection endpoint."""

    def test_coprocessor_carries_the_endpoint(self):
        """A coprocessor holds the address the controller dials and the port it listens on."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", comm_host="192.168.1.3", oob_port=18590)
        assert cop.comm_host == "192.168.1.3"
        assert cop.oob_port == 18590

    def test_oob_port_is_an_int(self):
        """oob_port is an int, matching the IR's IntegerAttr and the runtime's uint16."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", oob_port=18590, comm_host="192.168.1.3")
        assert isinstance(cop.oob_port, int)

    def test_comm_host_is_required(self):
        """Every coprocessor needs one — the MLIR verifier rejects a coprocessor with no peer."""
        with pytest.raises(TypeError, match="comm_host"):
            qp.Coprocessor(coprocessor_fn="decoder")  # pylint: disable=missing-kwoa

    def test_oob_port_is_optional(self):
        """Only oob_port defaults; the runtime picks one when it is not given."""
        assert qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1").oob_port is None

    def test_colocated_coprocessor_still_needs_comm_host(self):
        """A co-located coprocessor is still dialed, so it needs an address too."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1", remote=False)
        assert cop.comm_host == "127.0.0.1"

    @pytest.mark.parametrize("bad", [0, -1, 65536, 100000])
    def test_oob_port_range_is_validated(self, bad):
        """oob_port must fit the IR's i16 / the runtime's uint16."""
        with pytest.raises(ValueError, match="1..65535"):
            qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1", oob_port=bad)

    def test_oob_port_rejects_strings(self):
        """The field used to be a str; passing one now fails clearly instead of late."""
        with pytest.raises(TypeError, match="oob_port must be an int"):
            qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1", oob_port="18590")

    @pytest.mark.parametrize("attr", ["comm_host", "oob_port"])
    def test_controller_has_no_endpoint(self, attr):
        """The controller dials the coprocessor's endpoint; it has none of its own."""
        assert not hasattr(qp.Controller(label="fpga"), attr)
        assert not hasattr(Node(), attr)

    @pytest.mark.parametrize("attr", ["addr", "port"])
    def test_old_endpoint_names_are_gone(self, attr):
        """addr/port were renamed to comm_host/oob_port."""
        assert not hasattr(qp.Coprocessor(coprocessor_fn="decoder", comm_host="192.168.1.3"), attr)
        assert not hasattr(Node(), attr)


class TestBackendSelection:
    """backend names the transport implementation; the compiler resolves it to a library."""

    def test_backend_is_a_name_not_a_path(self):
        """A backend is selected by name, so no build paths appear in user code."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1", backend="gpu_verbs")
        assert cop.backend == "gpu_verbs"

    def test_controller_also_has_a_backend(self):
        """Both roles need one — each backend ships controller and coprocessor libraries."""
        assert qp.Controller(label="fpga", backend="cpu_verbs").backend == "cpu_verbs"

    def test_backend_defaults_to_none(self):
        """Omitted, the compiler picks its default."""
        assert qp.Controller(label="fpga").backend is None

    def test_backend_is_not_validated_against_a_fixed_list(self):
        """Out-of-tree backends must work, so the name is passed through unchecked."""
        cop = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1", backend="fpga_verbs")
        assert cop.backend == "fpga_verbs"

    def test_backend_lib_override_lives_in_init_args(self):
        """An explicit library path stays available as an escape hatch."""
        cop = qp.Coprocessor(
            coprocessor_fn="decoder",
            comm_host="127.0.0.1",
            backend="cpu_verbs",
            init_args={"backend_lib": "/opt/catalyst/libcustom.so"},
        )
        assert cop.backend == "cpu_verbs"
        assert cop.init_args["backend_lib"] == "/opt/catalyst/libcustom.so"


class TestLabelIsIdentityNotSelector:
    """label identifies the node; it is not a backend selector."""

    def test_label_and_backend_are_separate_fields(self):
        """Identity and backend selection are separate fields."""
        cop = qp.Coprocessor(
            coprocessor_fn="decoder",
            comm_host="127.0.0.1",
            label="decoder-0",
            backend="gpu_verbs",
        )
        assert cop.label == "decoder-0"
        assert cop.backend == "gpu_verbs"

    def test_label_defaults_to_none(self):
        """Omitted, the compiler derives one from the node's role."""
        assert qp.Coprocessor(coprocessor_fn="d", comm_host="127.0.0.1").label is None


class TestRemoteDefault:
    """Nodes are local by default; remote is opt-in."""

    def test_controller_is_local_by_default(self):
        """remote defaults to False, matching every other PennyLane device."""
        assert qp.Controller(label="fpga").remote is False

    def test_coprocessor_is_local_by_default(self):
        assert qp.Coprocessor(coprocessor_fn="d", comm_host="127.0.0.1").remote is False

    def test_remote_is_opt_in(self):
        assert qp.Controller(label="fpga", remote=True).remote is True


class TestTripleRemoved:
    """triple lives on the executor, which detects it on the target host."""

    @pytest.mark.parametrize(
        "node",
        [Node(), Coprocessor(coprocessor_fn="decoder", comm_host="192.168.1.3")],
        ids=["Node", "Coprocessor"],
    )
    def test_no_triple_field(self, node):
        """Nodes no longer carry a cross-compilation triple."""
        assert not hasattr(node, "triple")

    def test_controller_has_no_triple(self):
        """The controller's triple comes from its executor, not the node."""
        assert not hasattr(qp.Controller(label="fpga"), "triple")

    def test_triple_rejected_as_kwarg(self):
        """Passing triple= now raises rather than being silently accepted."""
        with pytest.raises(TypeError):
            qp.Controller(label="fpga", triple="aarch64-unknown-linux-gnu")


class TestPlacement:
    """Tests for the Placement container."""

    def test_placement_construction(self):
        """Placement groups a controller, coprocessors, and a transport."""
        ctrl = qp.Controller(label="controller-label")
        cop = qp.Coprocessor(
            label="gpu-libibverbs",
            coprocessor_fn="decoder-XX",
            comm_host="192.168.1.3",
            oob_port=18590,
        )
        placement = Placement(controller=ctrl, coprocessors=(cop,), transport="rdma")
        assert placement.controller is ctrl
        assert placement.coprocessors == (cop,)
        assert placement.transport == Transport("rdma")

    def test_transport_name_resolved(self):
        """A transport name is resolved to a Transport on construction."""
        ctrl = qp.Controller(label="fpga")
        placement = Placement(controller=ctrl, transport="rdma")
        assert isinstance(placement.transport, Transport)
        assert placement.transport.name == "rdma"

    def test_coprocessors_coerced_to_tuple(self):
        """A list of coprocessors is stored as a tuple."""
        ctrl = qp.Controller(label="fpga")
        cop = qp.Coprocessor(label="gpu", coprocessor_fn="decode", comm_host="192.168.1.3")
        placement = Placement(controller=ctrl, coprocessors=[cop], transport="rdma")
        assert isinstance(placement.coprocessors, tuple)

    def test_unknown_transport_raises(self):
        """An unregistered transport name is rejected at construction."""
        ctrl = qp.Controller(label="fpga")
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
        ctrl = qp.Controller(label="fpga", executor_options={"threads": 4})
        assert ctrl.executor_options == {"threads": 4}
        assert ctrl.executor is None

    def test_no_executor_requested_by_default(self):
        """executor_options defaults to None, meaning "launch nothing"."""
        assert qp.Controller(label="fpga").executor_options is None

    def test_empty_options_still_requests_an_executor(self):
        """``{}`` means "launch with all defaults" and must stay distinct from ``None``."""
        ctrl = qp.Controller(label="fpga", executor_options={})
        assert ctrl.executor_options == {}
        assert ctrl.executor_options is not None

    def test_prelaunched_executor_can_be_attached(self):
        """A already-launched executor can be set directly, bypassing executor_options."""

        class _Ex:  # duck-typed stand-in for catalyst.Executor
            address = "10.0.0.5:1373"
            triple = "aarch64-unknown-linux-gnu"

        ex = _Ex()
        ctrl = qp.Controller(label="fpga", executor=ex)
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
            lambda: qp.Coprocessor("decoder", comm_host="192.168.1.3"),
        ],
    )
    def test_nodes_are_keyword_only(self, call):
        """Controller and Coprocessor both reject positional arguments."""
        with pytest.raises(TypeError):
            call()
