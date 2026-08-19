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

"""Tests for the heterogeneous backline device."""

import pytest

import pennylane as qp
from pennylane.backline import Placement
from pennylane.backline.device import Backline
from pennylane.core.transforms.compile_pipeline import CompilePipeline
from pennylane.devices import ExecutionConfig


def _nodes(n_coprocessors=1):
    ctrl = qp.Controller(label="controller-1")
    cops = tuple(
        qp.Coprocessor(
            label=f"coproc-{i}",
            coprocessor_fn=f"decoder-{i}",
            comm_host="192.168.1.3",
            oob_port=18590,
        )
        for i in range(n_coprocessors)
    )
    return ctrl, cops


def _device(n_coprocessors=1):
    ctrl, cops = _nodes(n_coprocessors)
    return Backline(controller=ctrl, coprocessors=cops, transport="rdma")


class TestConstruction:
    """Tests for constructing the device."""

    def test_holds_placement_and_transport(self):
        """The device builds a Placement and exposes it along with the transport."""
        dev = _device()
        assert isinstance(dev.placement, Placement)
        assert dev.transport.name == "rdma"

    def test_all_arguments_are_keyword_only(self):
        """No argument may be passed positionally, so ordering can never be confused, and a
        controller is never mistaken for wires."""
        ctrl, cops = _nodes()
        with pytest.raises(TypeError):
            # pylint: disable=too-many-function-args,missing-kwoa
            Backline(ctrl, transport="rdma")
        with pytest.raises(TypeError):
            # pylint: disable=too-many-function-args,missing-kwoa
            Backline(cops[0], ctrl, transport="rdma")

    def test_wires_taken_from_controller_device(self):
        """The device's wires come from the controller's device."""
        ctrl = qp.Controller(device=qp.device("default.qubit", wires=3), label="fpga")
        dev = Backline(controller=ctrl, transport="rdma")
        assert len(dev.wires) == 3

    def test_qec_code_defaults_to_none(self):
        """qec_code is optional and reaches the placement."""
        ctrl, _ = _nodes(0)
        assert Backline(controller=ctrl, transport="rdma").qec_code is None
        dev = Backline(controller=ctrl, transport="rdma", qec_code="steane")
        assert dev.qec_code == "steane"
        assert dev.placement.qec_code == "steane"


class TestPublicUsagePattern:
    """The documented user-facing entry points keep working."""

    def test_top_level_qp_backline(self):
        """import pennylane as qp; qp.Backline(...) builds a device."""
        con = qp.Controller(device=qp.device("default.qubit", wires=2))
        cop = qp.Coprocessor(coprocessor_fn="decoder", label="gpu-verbs", comm_host="192.168.1.3")
        dev = qp.Backline(controller=con, coprocessors=[cop], transport="rdma")
        assert isinstance(dev, Backline)

    def test_classes_imported_from_submodule(self):
        """from pennylane.backline import Controller, Coprocessor -> qp.Backline(...)."""
        from pennylane.backline import Controller, Coprocessor

        con = Controller(device=qp.device("default.qubit", wires=2))
        cop = Coprocessor(coprocessor_fn="decoder", label="gpu-libibverbs", comm_host="192.168.1.3")
        dev = qp.Backline(controller=con, coprocessors=[cop], transport="rdma")
        assert dev.controller is con
        assert dev.coprocessors == (cop,)

    def test_submodule_is_reachable_as_an_attribute(self):
        """qp.backline is the submodule, not a shadowing function.

        The lowercase ``backline()`` constructor used to shadow the module as an attribute of
        ``pennylane``, so ``qp.backline.Placement`` raised ``AttributeError``.
        """
        assert qp.backline.Placement is Placement
        assert "Backline" in qp.backline.__all__

    def test_coprocessors_accepts_any_sequence(self):
        """A list or tuple of coprocessors both work, and are normalized to a tuple."""
        con = qp.Controller(label="c")
        cop = qp.Coprocessor(coprocessor_fn="decoder", label="gpu", comm_host="192.168.1.3")
        for seq in ([cop], (cop,)):
            dev = qp.Backline(controller=con, coprocessors=seq, transport="rdma")
            assert dev.coprocessors == (cop,)

    def test_coprocessors_defaults_to_empty(self):
        """coprocessors may be omitted entirely."""
        con = qp.Controller(label="c")
        dev = qp.Backline(controller=con, transport="rdma")
        assert dev.coprocessors == ()

    def test_heterogeneous_device_name_is_gone(self):
        """The device was renamed; the old name must not linger."""
        assert not hasattr(qp, "HeterogeneousDevice")
        assert not hasattr(qp.backline, "HeterogeneousDevice")


class TestAccessors:
    """Tests for the controller / coprocessors accessors."""

    def test_controller(self):
        assert _device().controller.label == "controller-1"

    def test_coprocessors(self):
        dev = _device(n_coprocessors=3)
        assert tuple(c.label for c in dev.coprocessors) == ("coproc-0", "coproc-1", "coproc-2")

    def test_no_coprocessors(self):
        assert _device(n_coprocessors=0).coprocessors == ()


class TestExecution:
    """Tests for the (absent) execution path."""

    def test_preprocess_returns_pipeline_and_config(self):
        program, config = _device().preprocess()
        assert isinstance(program, CompilePipeline)
        assert isinstance(config, ExecutionConfig)

    def test_execute_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Catalyst"):
            _device().execute([])
