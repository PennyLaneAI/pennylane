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

import os
import sys
from multiprocessing import cpu_count

import numpy as np
import pytest

from pennylane.concurrency.executors import ExecBackends, create_executor
from pennylane.concurrency.executors.native import (
    MPPoolExec,
    ProcPoolExec,
    SerialExec,
    ThreadPoolExec,
)


def get_core_count():
    if sys.version_info.minor >= 13:
        return os.process_cpu_count()
    return os.cpu_count()


def custom_func1(arg0):
    return np.cos(arg0) * np.exp(1j * arg0)


def custom_func2(arg0):
    return arg0**2


def custom_func3(arg0, arg1):
    return arg0 + arg1


def custom_func4(arg0, arg1, arg2):
    return arg0 + arg1 - arg2


@pytest.mark.parametrize(
    "backend",
    [
        (ExecBackends.MP_Pool, MPPoolExec),
        (ExecBackends.CF_ProcPool, ProcPoolExec),
        (ExecBackends.CF_ThreadPool, ThreadPoolExec),
        (ExecBackends.Serial, SerialExec),
    ],
)
class TestLocalExecutor:
    """Test executor creation and validation."""

    def test_construction(self, backend):
        executor = create_executor(backend[0])
        assert isinstance(executor, backend[1])

    def test_max_workpool_size(self, backend):
        "Test executor creation with the maximum permitted worker count"
        if "Serial" in backend[1].__name__:
            pytest.skip("Serial backends have a single worker")
        executor = create_executor(backend[0])
        assert executor.size == get_core_count()

    @pytest.mark.parametrize("workers", [1, 2, 6])
    def test_fixed_workpool_size(self, backend, workers):
        "Test executor creation with a predefined worker count"
        if "Serial" in backend[1].__name__:
            pytest.skip("Serial backends have a single worker")
        executor = create_executor(backend[0], max_workers=workers)
        assert executor.size == workers

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (custom_func1, np.pi / 3, custom_func1(np.pi / 3)),
            (custom_func2, np.pi / 7, custom_func2(np.pi / 7)),
            (sum, range(16), sum(range(16))),
        ],
    )
    def test_submit_single_param(self, fn, data, result, backend):
        """
        Test valid and invalid data mapping through the executor
        """

        executor = create_executor(backend[0])
        exec_result = executor.submit(fn, data)
        if not isinstance(result, list):
            assert np.isclose(result, exec_result)
        else:
            assert np.allclose(result, exec_result)

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (custom_func3, (np.pi / 5, 1.2), custom_func3(np.pi / 5, 1.2)),
            (custom_func4, (np.pi / 3, 2.4, 5.6), custom_func4(np.pi / 3, 2.4, 5.6)),
        ],
    )
    def test_submit_multi_param(self, fn, data, result, backend):
        """
        Test valid and invalid data mapping through the executor
        """

        executor = create_executor(backend[0])
        exec_result = executor.submit(fn, *data)
        if not isinstance(result, list):
            assert np.isclose(result, exec_result)
        else:
            assert np.allclose(result, exec_result)

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (custom_func1, range(7), [custom_func1(i) for i in range(7)]),
            (custom_func2, range(3), list(map(lambda x: x**2, range(3)))),
            (custom_func3, (range(3), range(3)), list(map(custom_func3, range(3), range(3)))),
            (
                custom_func4,
                (range(3), range(3), range(3)),
                list(map(custom_func4, range(3), range(3), range(3))),
            ),
        ],
    )
    def test_map(self, fn, data, result, backend):
        """
        Test valid and invalid data mapping through the executor
        """

        executor = create_executor(backend[0])
        assert np.allclose(result, list(executor.map(fn, *data)))

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (
                custom_func3,
                list(zip(range(9), np.ones(9))),
                [custom_func3(i, j) for i, j in zip(range(9), np.ones(9))],
            ),
            (
                custom_func4,
                list(zip([np.linspace(-5, 5, 10)], [np.linspace(-5, 5, 10)], np.ones(10))),
                [
                    custom_func4(i, j, k)
                    for i, j, k in zip(
                        [np.linspace(-5, 5, 10)], [np.linspace(-5, 5, 10)], np.ones(10)
                    )
                ],
            ),
        ],
    )
    def test_starmap(self, fn, data, result, backend):
        """
        Test valid and invalid data mapping through the executor
        """

        executor = create_executor(backend[0])
        assert np.allclose(result, list(executor.starmap(fn, data)))

    def test_persistence(self, backend):
        executor_temporal = create_executor(backend[0])
        executor_persist = create_executor(backend[0], persist=True)
        assert not executor_temporal.persist
        assert executor_persist.persist

    @pytest.mark.parametrize(
        "fn,data,result",
        [
            (custom_func3, (range(3), range(3)), list(map(custom_func3, range(3), range(3)))),
            (
                custom_func4,
                (range(3), range(3), range(3)),
                list(map(custom_func4, range(3), range(3), range(3))),
            ),
        ],
    )
    @pytest.mark.parametrize("exec_kwargs", [{}, {"chunksize": 2}, {"chunksize": 4}])
    def test_functor_call(self, backend, fn, data, result, exec_kwargs):
        executor = create_executor(backend[0], **exec_kwargs)
        for i, a in enumerate(zip(*data)):
            assert np.isclose(result[i], executor("submit", fn, *a))
        assert np.allclose(result, executor("map", fn, *data))
        assert np.allclose(result, executor("map", fn, *data))
        assert np.allclose(result, executor("starmap", fn, zip(*data)))
