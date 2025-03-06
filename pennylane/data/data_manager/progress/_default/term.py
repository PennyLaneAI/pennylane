# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions wrapping ANSI terminal control sequences. See:
https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_(Control_Sequence_Introducer)_sequences
"""


def cursor_up(n: int) -> str:
    """Return 'A' control sequence, which to moves the cursor up ``n`` columns.

    >>> cursor_up(2)
    '\x1b[2;A'

    """
    return f"\x1b[{n};A"


def erase_line() -> str:
    """Return 'K' control sequence, which erases the current line starting from
    the cursor.

    >>> erase_line()
    '\x1b[0;K'
    """
    return "\x1b[0;K"
