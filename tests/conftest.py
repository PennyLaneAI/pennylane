# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for PennyLane test suite.
"""
import pytest
import os


# defaults
TOL = 1e-3


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope='session')
def torch_support():
    """Boolean fixture for PyTorch support"""
    try:
        import torch
        from torch.autograd import Variable
        torch_support = True
    except ImportError as e:
        torch_support = False

    return torch_support


@pytest.fixture(scope='session')
def tf_support():
    """Boolean fixture for TensorFlow support"""
    try:
        import tensorflow as tf
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        tf_support = True
    except ImportError as e:
        tf_support = False

    return tf_support
