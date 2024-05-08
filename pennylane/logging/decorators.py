import inspect
from functools import partial, wraps
import logging

# Stack level allows moving up the stack with the log records, and prevents
# the decorator function names appearing in the resulting messages.
_debug_log_kwargs = {"stacklevel": 2}


def log_string_debug_func(func, log_level, use_entry, override={}):
    """
    This decorator utility generates a string containing the called function, the passed arguments, and the source of the function call.
    """
    lgr = logging.getLogger(func.__module__)

    def _get_bound_signature(*args, **kwargs) -> str:
        s = inspect.signature(func)

        try:
            ba = s.bind(*args, **kwargs)
        except Exception as e:
            # If kwargs are concat onto args, attempt to unpack. Fail otherwise
            if len(args) == 2 and len(kwargs) == 0 and isinstance(args[1], dict):
                ba = s.bind(*args[0], **args[1])
            else:
                raise e
        ba.apply_defaults()
        if len(override):
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


def log_string_debug_class(cl, log_level):
    import types

    lgr = logging.getLogger(cl.__module__)
    cl_func = inspect.getmembers(cl, predicate=inspect.isfunction)
    f_par_entry = partial(log_string_debug_func, log_level=log_level, use_entry=True)
    f_par_exit = partial(log_string_debug_func, log_level=log_level, use_entry=False)

    for f_name, f in cl_func:
        if f_name == "__init__":
            setattr(cl, f_name, types.MethodType(f_par_exit(f), cl))
        else:
            setattr(cl, f_name, types.MethodType(f_par_entry(f), cl))
    return cl

# For ease-of-use ``debug_logger`` is provided for decoration of public methods and free functions, with ``debug_logger_init`` provided for use with class constructors.
debug_logger = partial(log_string_debug_func, log_level=logging.DEBUG, use_entry=True)
debug_logger_init = partial(log_string_debug_func, log_level=logging.DEBUG, use_entry=False)
