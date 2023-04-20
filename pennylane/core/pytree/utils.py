from functools import partial, wraps

   
def property_cached(fun):
    """
    Better implementation of cached_property from the standard library.
    """
    @wraps(fun)
    def property_fun(self, *args, **kwargs):
        property_name = f"__cached_{fun.__name__}"
        if not hasattr(self, property_name):
            value = fun(*args, **kwargs)
            object.__setattr__(self, property_name, value)
        return object.__getattribute__(self, property_name)

    return property_fun