# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the `qml.workflow.resolution._resolve_mcm_config` helper function"""

import pytest

from pennylane.devices import MCMConfig
from pennylane.math import Interface
from pennylane.workflow import _resolve_mcm_config


def test_no_finite_shots():
    """Test resolution of MCM config when finite_shots is False."""
    mcm_config = MCMConfig(mcm_method="one-shot")

    with pytest.raises(ValueError, match="Cannot use the 'one-shot' method"):
        _resolve_mcm_config(mcm_config, interface=Interface.AUTO, finite_shots=False)

    mcm_config = MCMConfig(mcm_method=None, postselect_mode="hw-like")
    resolved_mcm_config = _resolve_mcm_config(
        mcm_config, interface=Interface.AUTO, finite_shots=False
    )
    assert resolved_mcm_config.postselect_mode is None


def test_single_branch_statistics():
    """Test error with using single-branch-statistics without qjit."""
    mcm_config = MCMConfig(mcm_method="single-branch-statistics")

    with pytest.raises(
        ValueError, match="Cannot use mcm_method='single-branch-statistics' without qml.qjit."
    ):
        _resolve_mcm_config(mcm_config, interface=Interface.AUTO, finite_shots=True)


def test_resolve_mcm_config_jax_jit_deferred():
    """Test resolution when interface is 'jax-jit' and mcm_method is 'deferred'."""
    mcm_config = MCMConfig(mcm_method="deferred", postselect_mode="hw-like")

    with pytest.raises(
        ValueError, match="Using postselect_mode='hw-like' is not supported with jax-jit"
    ):
        _resolve_mcm_config(mcm_config, interface=Interface.JAX_JIT, finite_shots=True)

    mcm_config = MCMConfig(mcm_method="deferred", postselect_mode=None)
    resolved = _resolve_mcm_config(mcm_config, interface=Interface.JAX_JIT, finite_shots=True)
    assert resolved.postselect_mode == "fill-shots"


@pytest.mark.parametrize("mcm_method", [None, "one-shot"])
@pytest.mark.parametrize("postselect_mode", [None, "hw-like"])
def test_resolve_mcm_config_finite_shots_pad_invalid_samples(mcm_method, postselect_mode):
    """Test resolution when finite_shots is True and interface is JAX."""
    mcm_config = MCMConfig(mcm_method=mcm_method, postselect_mode=postselect_mode)
    resolved = _resolve_mcm_config(mcm_config, interface=Interface.JAX, finite_shots=True)

    assert resolved.postselect_mode == "pad-invalid-samples"
