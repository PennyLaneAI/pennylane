# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file provides support for logging framework filters. For more information please see the
official Python documentation on filters at https://docs.python.org/3/library/logging.html#filter"""
import logging
import os
from logging import Filter


# pylint: disable=too-few-public-methods
class LocalProcessFilter(Filter):
    """
    Filters logs not originating from the current executing Python process ID.
    """

    def __init__(self):
        super().__init__()
        self._pid = os.getpid()

    def filter(self, record):
        if record.process == self._pid:
            return True
        return False


# pylint: disable=too-few-public-methods
class DebugOnlyFilter(Filter):
    """
    Filters logs that are less verbose than the DEBUG level (CRITICAL, ERROR, WARN & INFO).
    """

    def filter(self, record):
        super().__init__()
        if record.levelno > logging.DEBUG:
            return False
        return True
