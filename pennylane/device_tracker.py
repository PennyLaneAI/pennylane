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

import time

def track(dev, version="default", reset_on_enter=True):
    if version=="timing":
        return TimingTracker(dev, reset_on_enter)
    else:
        return DevTracker(dev, reset_on_enter)


class DevTracker:
    """
    Class docstring
    """

    def __init__(self, dev=None, reset_on_enter=True):
        """
        docstring
        """
        self.reset_on_enter = reset_on_enter

        self.data = dict()
        self.tracking = False

        if dev is not None:
            dev.tracker = self
            
    def __enter__(self):
        """
        docstring for enter
        """
        if self.reset_on_enter:
            self.reset()

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
                self.data[key] = kwargs[key] + self.data.get(key, 0)

    def reset(self):
        """ reseting data"""
        self.data = dict()

    def record(self):
        """
        record data somehow
        """
        for key, value in self.data.items():
            print(f"{key} = {value}", end="\t")
        print()

class TimingTracker(DevTracker):

    def update(self, **kwargs):

        super().update(**kwargs)

        self.data["total_time"] = time.time() - self.t0
        self.times.append(self.data["total_time"])

    def record(self):
        """
        record data somehow
        """
        for key, value in self.data.items():
            print(f"{key} = {value}", end="\t")
        print()

    def reset(self):
        super().reset()
        self.t0 = time.time()
        self.times = []
