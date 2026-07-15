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

"""Unit tests for backline placement types (Endpoint, backline)."""

import dataclasses

import pytest

import pennylane as qp


class TestEndpoint:
    """The Endpoint type: one network node (host + role)."""

    def test_stores_host_and_role(self):
        ep = qp.Endpoint("qpu.hostname", role="qpu")
        assert ep.host == "qpu.hostname"
        assert ep.role == "qpu"

    def test_attrs_defaults_to_none(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder")
        assert ep.attrs is None

    def test_attrs_can_be_provided(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder", attrs={"triple": "x86_64"})
        assert ep.attrs == {"triple": "x86_64"}

    def test_name_defaults_to_none(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder")
        assert ep.name is None

    def test_name_can_be_provided(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder", name="gpu-0")
        assert ep.name == "gpu-0"

    def test_local_defaults_to_true(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder")
        assert ep.local is True

    def test_local_can_be_overridden(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder", local=False)
        assert ep.local is False

    def test_is_frozen(self):
        ep = qp.Endpoint("gpu.hostname", role="decoder")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ep.host = "other.ip"

    def test_decoder_defaults_to_none(self):
        ep = qp.Endpoint("gpu.hostname", role="gpu-decoder")
        assert ep.decoder is None

    def test_decoder_accepts_selector_string(self):
        ep = qp.Endpoint("gpu.hostname", role="gpu-decoder", decoder="steane")
        assert ep.decoder == "steane"

    def test_decoder_accepts_builder_object(self):
        builder = object()  # stands in for a Gluon builder
        ep = qp.Endpoint("gpu.hostname", role="gpu-decoder", decoder=builder)
        assert ep.decoder is builder


class TestBackline:
    """The backline placement: a controller, optional coprocessors, and a transport."""

    def test_stores_controller_coprocessors_transport(self):
        ctrl = qp.Endpoint("qpu.hostname", role="qpu")
        gpu = qp.Endpoint("gpu.hostname", role="decoder")
        bl = qp.backline(controller=ctrl, coprocessors=(gpu,), transport="roce")
        assert bl.controller is ctrl
        assert bl.coprocessors == (gpu,)
        assert bl.transport == "roce"

    def test_coprocessors_default_empty(self):
        bl = qp.backline(controller=qp.Endpoint("qpu.hostname", role="qpu"), transport="roce")
        assert bl.coprocessors == ()

    def test_transport_is_required(self):
        with pytest.raises(TypeError):
            qp.backline(controller=qp.Endpoint("qpu.hostname", role="qpu"))

    def test_unknown_transport_rejected(self):
        with pytest.raises(ValueError, match="unknown transport"):
            qp.backline(controller=qp.Endpoint("qpu.hostname", role="qpu"), transport="rocev20")
