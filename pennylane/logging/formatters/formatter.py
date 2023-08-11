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
"""The PennyLane log-level formatters are defined here with default options, and ANSI-terminal color-codes."""
import logging
from logging import Formatter
from typing import NamedTuple, Tuple, Union

# For color-code definitions see https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
# 24-bit mode support RGB color codes in 8bit-wide r;g;b format
_ANSI_CODES = {
    "begin": "\x1b[",
    "end": "\x1b[0m",
    "foreground_rgb": "38;2;",
    "background_rgb": "48;2;",
    "foreground": "38;5;",
    "background": "48;5;",
}


class ColorScheme(NamedTuple):
    """Utility class to contain level-controlled color-codes for log messages."""

    use_rgb: bool
    debug: str
    info: str
    warning: str
    error: str
    critical: str
    debug_bg: Union[None, str]
    info_bg: Union[None, str]
    warning_bg: Union[None, str]
    error_bg: Union[None, str]
    critical_bg: Union[None, str]


def build_code_rgb(rgb: Tuple[int, int, int], rgb_bg: Union[None, Tuple[int, int, int]] = None):
    """
    Utility function to generate the appropriate ANSI RGB codes for a given set of foreground (font) and background colors.
    """
    output = _ANSI_CODES["begin"]
    output += _ANSI_CODES["foreground_rgb"]
    output += ";".join([str(i) for i in rgb])
    output += "m"
    if rgb_bg:
        output += _ANSI_CODES["begin"]
        output += _ANSI_CODES["background_rgb"]
        output += ";".join([str(i) for i in rgb_bg])
        output += "m"
    return output


def bash_ansi_codes():
    """Utility function to generate a bash command for all ANSI color-codes in 24-bit format using both foreground and background colors."""
    str = r"""
        for r in {0..255}; do
            for g in {0..255}; do
                for b in {0..255}; do
                    echo -e "\e[38;2;${r};${g};${b}m"'\\e[38;2;'"${r}"m" FOREGROUND\e[0m"
                    echo -e "\e[48;2;${r};${g};${b}m"'\\e[48;2;'"${r}"m" FOREGROUND\e[0m"
                done
            done
        done"""
    return str


class DefaultFormatter(Formatter):
    """This formatter has the default rules used for formatting PennyLane log messages."""

    fmt_str = '[%(asctime)s][%(levelname)s][<PID %(process)d:%(processName)s>] - %(name)s.%(funcName)s()::"%(message)s"'

    # 0x000000 Background
    _text_bg = (0, 0, 0)

    cmap = ColorScheme(
        debug=(220, 238, 200),  # Grey 1
        debug_bg=_text_bg,
        info=(80, 125, 125),  # Blue
        info_bg=_text_bg,
        warning=(208, 167, 133),  # Orange
        warning_bg=_text_bg,
        error=(208, 133, 133),  # Red 1
        error_bg=_text_bg,
        critical=(135, 53, 53),  # Red 2
        critical_bg=(210, 210, 210),  # Grey 2
        use_rgb=True,
    )

    FORMATS = {
        logging.DEBUG: build_code_rgb(cmap.debug, cmap.debug_bg) + fmt_str + _ANSI_CODES["end"],
        logging.INFO: build_code_rgb(cmap.info, cmap.info_bg) + fmt_str + _ANSI_CODES["end"],
        logging.WARNING: build_code_rgb(cmap.warning, cmap.warning_bg)
        + fmt_str
        + _ANSI_CODES["end"],
        logging.ERROR: build_code_rgb(cmap.error, cmap.error_bg) + fmt_str + _ANSI_CODES["end"],
        logging.CRITICAL: build_code_rgb(cmap.critical, cmap.critical_bg)
        + fmt_str
        + _ANSI_CODES["end"],
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SimpleFormatter(Formatter):
    """This formatter has a simplified layout and rules used for formatting PennyLane log messages."""

    indigo = "\x1b[38;4;45m"
    blue = "\x1b[38;4;44m"
    yellow = "\x1b[38;4;43m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;107m"
    reset = "\x1b[0m"
    str_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: "\x1b[38;5;10m" + str_format + reset,
        logging.INFO: "\x1b[38;5;12m" + str_format + reset,
        logging.WARNING: "\x1b[38;5;11m" + str_format + reset,
        logging.ERROR: "\x1b[38;5;9m" + str_format + reset,
        logging.CRITICAL: "\x1b[38;5;13m" + str_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
