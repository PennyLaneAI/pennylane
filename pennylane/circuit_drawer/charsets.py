# Copyright 2020 Xanadu Quantum Technologies Inc.

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
This module contains the different charsets supported by PennyLane's CircuitDrawer.
"""
import abc


class CharSet(abc.ABC):
    """Charset base class."""

    # pylint: disable=too-few-public-methods

    WIRE = None
    MEASUREMENT = None
    TOP_MULTI_LINE_GATE_CONNECTOR = None
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = None
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = None
    EMPTY_MULTI_LINE_GATE_CONNECTOR = None
    CONTROL = None
    LANGLE = None
    RANGLE = None
    VERTICAL_LINE = None
    CROSSED_LINES = None
    PIPE = None
    OTIMES = None

    @staticmethod
    @abc.abstractmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""

    @staticmethod
    @abc.abstractmethod
    def to_subscript(num):
        """Convert the given number to a subscripted string."""


class UnicodeCharSet(CharSet):
    """Charset for CircuitDrawing made of Unicode characters."""

    # pylint: disable=too-few-public-methods

    WIRE = "─"
    MEASUREMENT = "┤"
    TOP_MULTI_LINE_GATE_CONNECTOR = "╭"
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = "├"
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = "╰"
    EMPTY_MULTI_LINE_GATE_CONNECTOR = "│"
    CONTROL = "C"
    LANGLE = "⟨"
    RANGLE = "⟩"
    VERTICAL_LINE = "│"
    CROSSED_LINES = "╳"
    PIPE = "|"
    OTIMES = "⊗"

    _superscript_dict = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }

    _subscript_dict = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₅",
        "5": "⁵",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
    }

    @staticmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""
        ret = str(num)
        for old, new in UnicodeCharSet._superscript_dict.items():
            ret = ret.replace(old, new)

        return ret

    @staticmethod
    def to_subscript(num):
        """Convert the given number to a subscripted string."""
        ret = str(num)
        for old, new in UnicodeCharSet._subscript_dict.items():
            ret = ret.replace(old, new)

        return ret


class AsciiCharSet(CharSet):
    """Charset for CircuitDrawing made of ASCII characters."""

    # pylint: disable=too-few-public-methods

    WIRE = "-"
    MEASUREMENT = "|"
    TOP_MULTI_LINE_GATE_CONNECTOR = "+"
    MIDDLE_MULTI_LINE_GATE_CONNECTOR = "+"
    BOTTOM_MULTI_LINE_GATE_CONNECTOR = "+"
    EMPTY_MULTI_LINE_GATE_CONNECTOR = "|"
    CONTROL = "C"
    LANGLE = "<"
    RANGLE = ">"
    VERTICAL_LINE = "|"
    CROSSED_LINES = "X"
    PIPE = "|"
    OTIMES = "@"

    @staticmethod
    def to_superscript(num):
        """Convert the given number to a superscripted string."""
        return "^" + str(num)

    @staticmethod
    def to_subscript(num):
        """Convert the given number to a subscripted string."""
        return "_" + str(num)


CHARSETS = {
    "unicode": UnicodeCharSet,
    "ascii": AsciiCharSet,
}
"""Dict[str, CharSet]: Dictionary mapping character sets to all available :class:`~.CharSet` classes."""
