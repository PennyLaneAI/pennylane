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
"""
Contains the dataset class.
"""
import os
import sys
import json


class Dataset:
    "The dataset data type. Allows users to create datasets with their own data"

    def __init__(self, **kwargs):
        self.variables = kwargs

    def from_dataset(self, dataset):
        for (key, val) in dataset.__dict__.items():
            setattr(self, key, val)
            #if getattr(self, key, None):
            #    setattr(self, key, val)

