import logging
from logging import Formatter
from typing import NamedTuple, Tuple, Union

# For color-code definitions see https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
# 24-bit mode support RGB color codes in 8bit-wide r;g;b format

ansi_codes = {
    "begin": "\x1b[",
    "end": "\x1b[0m",
    "foreground_rgb": "38;2;",
    "background_rgb": "48;2;",
    "foreground": "38;5;",
    "background": "48;5;",
}


class ColorScheme(NamedTuple):
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


red_shades = {
    0: (135, 53, 53),
    1: (208, 133, 133),
    2: (169, 85, 85),
    3: (108, 28, 28),
    4: (71, 7, 7),
}

orange_shades = {
    0: (135, 90, 53),
    1: (208, 167, 133),
    2: (169, 123, 85),
    3: (108, 64, 28),
    4: (71, 36, 7),
}

blue_shades = {
    0: (32, 81, 81),
    1: (80, 125, 125),
    2: (51, 101, 101),
    3: (17, 65, 65),
    4: (4, 43, 43),
}

green_shades = {
    0: (43, 108, 43),
    1: (106, 166, 106),
    2: (68, 135, 68),
    3: (23, 86, 23),
    4: (6, 57, 6),
}


def build_code_rgb(rgb: Tuple[int, int, int], rgb_bg: Union[None, Tuple[int, int, int]] = None):
    """
    Utility function to generate the appropriate ANSI RGB codes for a given set of foreground (font) and background colors.
    """
    output = ansi_codes["begin"]
    output += ansi_codes["foreground_rgb"]
    output += ";".join([str(i) for i in rgb])
    output += "m"
    if rgb_bg:
        output += ansi_codes["begin"]
        output += ansi_codes["background_rgb"]
        output += ";".join([str(i) for i in rgb_bg])
        output += "m"
    return output


text_bg = (0, 0, 0)

color_maps = {
    "default": ColorScheme(
        debug=(220, 238, 200),
        debug_bg=text_bg,
        info=blue_shades[1],
        info_bg=text_bg,
        warning=orange_shades[1],
        warning_bg=text_bg,
        error=red_shades[1],
        error_bg=text_bg,
        critical=red_shades[0],
        critical_bg=(210, 210, 210),
        use_rgb=True,
    )
}


def print_bash_codes():
    """
    Utility function to generate a bash command for all ANSI color-codes in 24-bit format using both foreground and background colors.
    """
    str = """
        for r in {0..255}; do
            for g in {0..255}; do
                for b in {0..255}; do
                    echo -e \"\\e[38;2;${r};${g};${b}m\"\'\\\\e[38;2;\'\"${r}\"m\" FOREGROUND\e[0m\"
                    echo -e \"\\e[48;2;${r};${g};${b}m\"\'\\\\e[48;2;\'\"${r}\"m\" FOREGROUND\e[0m\"
                done
            done
        done"""
    return str


class DefaultFormatter(Formatter):
    fmt_str = '[%(asctime)s][%(levelname)s][<PID %(process)d:%(processName)s>] - %(name)s.%(funcName)s()::"%(message)s"'

    cmap = color_maps["default"]

    FORMATS = {
        logging.DEBUG: build_code_rgb(cmap.debug, cmap.debug_bg) + fmt_str + ansi_codes["end"],
        logging.INFO: build_code_rgb(cmap.info, cmap.info_bg) + fmt_str + ansi_codes["end"],
        logging.WARNING: build_code_rgb(cmap.warning, cmap.warning_bg)
        + fmt_str
        + ansi_codes["end"],
        logging.ERROR: build_code_rgb(cmap.error, cmap.error_bg) + fmt_str + ansi_codes["end"],
        logging.CRITICAL: build_code_rgb(cmap.critical, cmap.critical_bg)
        + fmt_str
        + ansi_codes["end"],
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class AnotherLogFormatter(Formatter):
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
