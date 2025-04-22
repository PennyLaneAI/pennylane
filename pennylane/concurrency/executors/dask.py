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
r"""
Contains concurrent executor abstractions for task-based workloads based on support provided by Dask's distributed backend.
"""

import os
from collections.abc import Callable, Sequence

from .base import ExtExec


class DaskExec(ExtExec):
    """
    Dask distributed abstraction class functor.
    """

    def __init__(self, max_workers=4, client_provider=None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)
        try:
            from dask.distributed import Client, LocalCluster
            from dask.distributed.deploy import Cluster
        except ImportError as ie:
            raise RuntimeError(
                "Dask Distributed cannot be found.\nPlease install via `pip install dask distributed`"
            ) from ie

        if client_provider is None:
            proc_threads = int(os.getenv("OMP_NUM_THREADS", "1"))
            self._cluster = LocalCluster(
                n_workers=max_workers, processes=True, threads_per_worker=proc_threads
            )
            self._exec_backend = Client(self._cluster)

        # Note: urllib does not validate
        # (see https://docs.python.org/3/library/urllib.parse.html#url-parsing-security),
        # so branch on str as URL
        elif isinstance(client_provider, str):
            self._exec_backend = Client(client_provider)

        elif isinstance(client_provider, Cluster):
            self._exec_backend = client_provider.get_client()

        self._size = len(self._exec_backend.scheduler_info()["workers"])

    def __call__(self, fn: Callable, data: Sequence):
        output_f = self._exec_backend.map(fn, data)
        return [o.result() for o in output_f]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def map(self, fn: Callable, *args: Sequence):
        output_f = self._exec_backend.map(fn, *args)
        return [o.result() for o in output_f]

    def starmap(self, fn: Callable, args: Sequence):
        raise NotImplementedError("Please use another backend for access to `starmap`")

    def shutdown(self):
        self._exec_backend.shutdown()
        self._cluster.close()

    def submit(self, fn, *args, **kwargs):
        return self._exec_backend.submit(fn, *args, **kwargs).result()

    @property
    def size(self):
        return self._size
