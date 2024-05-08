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
"""This file expands the PennyLane logging functionality to allow additions for function entry and exit logging via decorators."""

import inspect
import logging
from functools import partial, wraps

# Stack level allows moving up the stack with the log records, and prevents
# the decorator function names appearing in the resulting messages.
_debug_log_kwargs = {"stacklevel": 2}


def log_string_debug_func(func, log_level, use_entry, override=None):
    """
    This decorator utility generates a string containing the called function, the passed arguments, and the source of the function call.
    """
    lgr = logging.getLogger(func.__module__)

    def _get_bound_signature(*args, **kwargs) -> str:
        s = inspect.signature(func)
        # pylint:disable = broad-except
        try:
            ba = s.bind(*args, **kwargs)
        except Exception as e:
            # If kwargs are concat onto args, attempt to unpack. Fail otherwise
            if len(args) == 2 and len(kwargs) == 0 and isinstance(args[1], dict):
                ba = s.bind(*args[0], **args[1])
            else:
                raise e
        ba.apply_defaults()
        if override and len(override):
            for k, v in override.keys():
                ba[k] = v

        f_string = str(ba).replace("BoundArguments ", func.__name__)
        return f_string

    @wraps(func)
    def wrapper_entry(*args, **kwargs):
        if lgr.isEnabledFor(log_level):  # pragma: no cover
            f_string = _get_bound_signature(*args, **kwargs)
            s_caller = "::L".join(
                [str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]]
            )
            lgr.debug(
                f"Calling {f_string} from {s_caller}",
                **_debug_log_kwargs,
            )
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper_exit(*args, **kwargs):
        output = func(*args, **kwargs)
        if lgr.isEnabledFor(log_level):  # pragma: no cover
            f_string = _get_bound_signature(*args, **kwargs)
            s_caller = "::L".join(
                [str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]]
            )
            lgr.debug(
                f"Calling {f_string}={output} from {s_caller}",
                **{"stacklevel": 2},
            )
        return output

    return wrapper_entry if use_entry else wrapper_exit


# For ease-of-use ``debug_logger`` is provided for decoration of public methods and free functions, with ``debug_logger_init`` provided for use with class constructors.
debug_logger = partial(log_string_debug_func, log_level=logging.DEBUG, use_entry=True)
debug_logger_init = partial(log_string_debug_func, log_level=logging.DEBUG, use_entry=False)
