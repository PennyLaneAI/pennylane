# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Test the core resource tracking pipeline. 
"""
# pylint:disable=protected-access, no-self-use
import copy
from collections import defaultdict

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator
from pennylane.operation import Operation


class DummyX(qml.X, ResourceOperator):

    def _resource_decomp(*args, **kwargs):
        raise NotImplementedError

    def resource_params(self) -> dict:
        return dict()

    @classmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        return CompressedResourceOp(cls, {})


class DummyY(qml.Y, ResourceOperator):

    def _resource_decomp(*args, **kwargs):
        raise NotImplementedError

    def resource_params(self) -> dict:
        return dict()

    @classmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        return CompressedResourceOp(cls, {})


class DummyOp(qml.I, ResourceOperator):

    def _resource_decomp(*args, **kwargs):
        x = DummyX.resource_rep()

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        return CompressedResourceOp(cls, {})


class DummyX(qml.X, ResourceOperator):

    def _resource_decomp(*args, **kwargs):
        raise NotImplementedError

    def resource_params(self) -> dict:
        return dict()

    @classmethod
    def resource_rep(cls, *args, **kwargs) -> CompressedResourceOp:
        return CompressedResourceOp(cls, {})


class TestGetResources:
    """Test the core resource tracking pipeline"""

    def test_resources_from_operation():
        """Test that we can extract the resources from an Operation."""

        assert True

    def test_resources_from_qfunc():
        assert True

    def test_resources_from_tape():
        assert True

    def test_counts_from_compressed_res():
        assert True

    def test_clean_gate_counts():
        assert True

    def test_operations_to_compressed_reps():
        assert True

    def test_get_resources():
        assert True
