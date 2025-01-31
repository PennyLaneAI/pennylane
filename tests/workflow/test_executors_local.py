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
"""Unit tests for local `qml.workflow.executors`"""

from multiprocessing import cpu_count

import numpy as np
import pytest

import pennylane as qml
from pennylane.workflow.executors import ExecBackends, create_executor
from pennylane.workflow.executors.native import MPPoolExec, ProcPoolExec, ThreadPoolExec


def custom_func1(scalar):
    return np.cos(scalar) * np.exp(1j * scalar)


def custom_func2(scalar):
    return scalar**2


@pytest.mark.parametrize(
    "backend",
    [
        (ExecBackends.MP_Pool, MPPoolExec),
        (ExecBackends.CF_ProcPool, ProcPoolExec),
        (ExecBackends.CF_ThreadPool, ThreadPoolExec),
    ],
)
class TestLocalExecutor:
    """Test executor creation and validation."""

    def test_construction(self, backend):
        executor = create_executor(backend[0])
        assert isinstance(executor, backend[1])

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (custom_func1, range(7), [custom_func1(i) for i in range(7)]),
            (custom_func2, range(3), list(map(lambda x: x**2, range(3)))),
            (sum, range(16), None),
        ],
    )
    def test_map(self, fn, data, result, backend):
        """
        Test valid and invalid data mapping through the executor
        """

        executor = create_executor(backend[0])
        if result is None:
            with pytest.raises(Exception) as e:
                print(fn, data)
                executor.map(fn, data)
        else:
            assert np.allclose(result, list(executor.map(fn, data)))

    def test_starmap(self, backend):
        pass

    @pytest.mark.parametrize(
        "workers",
        [None, 1, 2, cpu_count()],
    )
    def test_workers(self, backend, workers):
        """
        Test executor creation with a fixed worker count
        """
        executor = create_executor(backend[0], max_workers=workers)

        if workers is None:
            assert executor.size == cpu_count()
            return
        assert executor.size == workers

    def test_call(self, backend):
        pass

    def test_persistence(self, backend):
        executor_temporal = create_executor(backend[0])
        executor_persist = create_executor(backend[0], persist=True)
        assert not executor_temporal.persist
        assert executor_persist.persist
