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
from typing import Optional

from ..base import ExtExec


# pylint: disable=import-outside-toplevel
class DaskExec(ExtExec):  # pragma: no cover
    """
    Dask distributed abstraction class functor.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        persist: bool = False,
        client_provider=None,
        **kwargs,
    ):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
        try:
            from dask.distributed import LocalCluster
            from dask.distributed.deploy import Cluster
        except ImportError as ie:
            raise RuntimeError(
                "Dask Distributed cannot be found.\nPlease install via ``pip install dask distributed``"
            ) from ie

        if client_provider is None:
            proc_threads = int(os.getenv("OMP_NUM_THREADS", "1"))
            self._cluster = LocalCluster(
                n_workers=max_workers if max_workers else self._get_system_core_count(),
                processes=True,
                threads_per_worker=proc_threads,
            )
            self._client = self._exec_backend()(self._cluster)

        # Note: urllib does not validate
        # (see https://docs.python.org/3/library/urllib.parse.html#url-parsing-security),
        # so branch on str as URL
        elif isinstance(client_provider, str):
            self._client = self._exec_backend()(client_provider)

        elif isinstance(client_provider, Cluster):
            self._client = client_provider.get_client()

        self._size = len(self._client.scheduler_info()["workers"])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._persist:
            self.shutdown()

    def map(self, fn: Callable, *args: Sequence, **kwargs):
        output_f = self._client.map(fn, *args, **kwargs)
        return [o.result() for o in output_f]

    def starmap(self, fn: Callable, args: Sequence, **kwargs):
        raise NotImplementedError("Please use another backend for access to ``starmap``")

    def shutdown(self):
        self._client.shutdown()
        if hasattr(self, "cluster"):
            self._cluster.close()

    def submit(self, fn, *args, **kwargs):
        return self._client.submit(fn, *args, **kwargs).result()

    def __del__(self):
        self.shutdown()

    @property
    def size(self):
        return len(self._client.scheduler_info()["workers"])

    @classmethod
    def _exec_backend(cls):
        from dask.distributed import Client

        return Client
