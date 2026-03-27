# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains tests for `construct_execution_config`."""

from dataclasses import replace

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig, MCMConfig
from pennylane.workflow import construct_execution_config


def dummycircuit():
    """Dummy function."""
    qml.X(0)
    return qml.expval(qml.Z(0))


@pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
def test_unresolved_construction(device_name, interface):
    """Test that an unresolved execution config is created correctly."""
    qn = qml.QNode(dummycircuit, qml.device(device_name, wires=1), interface=interface)

    config = construct_execution_config(qn, resolve=False)()

    mcm_config = MCMConfig(None, None)
    expected_config = ExecutionConfig(
        grad_on_execution=False if "jax" in interface else None,
        use_device_gradient=None,
        use_device_jacobian_product=False,
        gradient_method="best",
        gradient_keyword_arguments={},
        device_options={},
        interface=interface,
        derivative_order=1,
        mcm_config=mcm_config,
        convert_to_numpy=True,
    )

    assert config == expected_config


def test_resolved_construction_lightning_qubit(interface):
    """Test that an resolved execution config is created correctly."""
    qn = qml.QNode(dummycircuit, qml.device("lightning.qubit", wires=1), interface=interface)

    config = construct_execution_config(qn, resolve=True)()

    postselect_mode = "fill-shots" if "jax-jit" == interface else None
    mcm_config = MCMConfig("deferred", postselect_mode)
    expected_config = ExecutionConfig(
        grad_on_execution=True,
        use_device_gradient=True,
        use_device_jacobian_product=False,
        gradient_method="adjoint",
        gradient_keyword_arguments={},
        interface=interface,
        derivative_order=1,
        mcm_config=mcm_config,
        convert_to_numpy=True,
    )

    # ignore comparison of device_options, could change
    assert replace(config, device_options={}) == replace(expected_config, device_options={})


def test_resolved_construction_default_qubit(interface):
    """Test that an resolved execution config is created correctly."""
    qn = qml.QNode(dummycircuit, qml.device("default.qubit", wires=1), interface=interface)

    config = construct_execution_config(qn, resolve=True)()

    postselect_mode = "fill-shots" if "jax-jit" == interface else None
    mcm_config = MCMConfig(mcm_method="deferred", postselect_mode=postselect_mode)
    expected_config = ExecutionConfig(
        grad_on_execution=False,
        use_device_gradient=True,
        use_device_jacobian_product=False,
        gradient_method="backprop",
        gradient_keyword_arguments={},
        interface=interface,
        derivative_order=1,
        mcm_config=mcm_config,
        convert_to_numpy=True,
    )

    # ignore comparison of device_options, could change
    assert replace(config, device_options={}) == replace(expected_config, device_options={})


@pytest.mark.parametrize("mcm_method", [None, "one-shot"])
@pytest.mark.parametrize("postselect_mode", [None, "hw-like"])
@pytest.mark.parametrize("interface", ["jax", "jax-jit"])
@pytest.mark.jax
def test_jax_interface(mcm_method, postselect_mode, interface):
    """Test constructing config with JAX interface and different MCMConfig settings."""

    @qml.qnode(
        qml.device("default.qubit"),
        interface=interface,
        mcm_method=mcm_method,
        postselect_mode=postselect_mode,
    )
    def circuit():
        qml.X(0)
        return qml.expval(qml.Z(0))

    config = construct_execution_config(qml.set_shots(circuit, 100))()

    expected_mcm_config = MCMConfig(mcm_method="one-shot", postselect_mode="pad-invalid-samples")
    expected_config = ExecutionConfig(
        grad_on_execution=False,
        use_device_gradient=False,
        use_device_jacobian_product=False,
        gradient_method=qml.gradients.param_shift,
        gradient_keyword_arguments={},
        interface=interface,
        derivative_order=1,
        mcm_config=expected_mcm_config,
        convert_to_numpy=True,
    )

    # ignore comparison of device_options, could change
    assert replace(config, device_options={}) == replace(expected_config, device_options={})
