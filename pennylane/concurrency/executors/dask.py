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
"""
Contains concurrent executor abstractions for task-based workloads based on support provided by Dask's distributed backend.
"""

from collections.abc import Callable, Sequence

try:
    from dask.distributed import Client, LocalCluster
    from dask.distributed.deploy import Cluster

    DASK_FOUND = True
except:
    DASK_FOUND = False
from .base import ExtExecABC

if DASK_FOUND:

    class DaskExec(ExtExecABC):
        """
        Dask distributed abstraction class functor.
        """

        def __init__(self, max_workers=4, client_provider=None, **kwargs):
            super().__init__(max_workers=max_workers, **kwargs)

            if client_provider is None:
                cluster = LocalCluster(n_workers=max_workers, processes=True)
                self._exec_backend = Client(cluster)

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

        @property
        def size(self):
            return self._size

else:

    class DaskExec(ExtExecABC):
        """
        Mock Dask distributed abstraction class functor.
        """

        def __init__(self, max_workers=4, client_provider=None, **kwargs):
            raise RuntimeError(
                "Dask Distributed cannot be found.\nPlease install via `pip install dask distributed`"
            )
