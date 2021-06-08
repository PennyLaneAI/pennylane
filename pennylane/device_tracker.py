# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

class DevTracker:

    def __init__(self, dev=None):
        """
        docstring
        """

        self.data = defaultdict(int)
        self.tracking = False

        if dev is not None:
            dev.tracker = self
            
    def __enter__(self):
        """
        docstring for enter
        """
        self.data = defaultdict(int)
        self.tracking = True
        return self

    def __exit__(self, exc_type, exc_value, exc_trackeback):
        """
        docstring for exit
        """
        self.tracking = False

    def update(self, **kwargs):
        """ updating data"""
        for key in kwargs:
            if kwargs[key] is not None:
                self.data[key] += kwargs[key]

    def record(self):
        """
        record data somehow
        """
        for key, value in self.data.items():
            print(f"{key} = {value}")
