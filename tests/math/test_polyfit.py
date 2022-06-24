# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for custom polyfit"""
# pylint: disable=import-outside-toplevel

import pytest
import pennylane as qml


@pytest.mark.tf
def test_polyfit_tf():
    """Testing polyfit on simple polynomial using tensorflow"""
    import tensorflow as tf

    x_tf = tf.range(10, dtype="float64")
    y_tf = 0.1 + 0.7 * x_tf + 0.5 * x_tf**2
    assert qml.math.allclose(qml.math.polyfit(x_tf, y_tf, 2), [0.5, 0.7, 0.1])


@pytest.mark.torch
def test_polyfit_torch():
    """Testing polyfit on simple polynomial using torch"""
    import torch

    x_torch = torch.arange(10, dtype=torch.float64)
    y_torch = 0.1 + 0.7 * x_torch + 0.5 * x_torch**2

    assert qml.math.allclose(qml.math.polyfit(x_torch, y_torch, 2), [0.5, 0.7, 0.1])
