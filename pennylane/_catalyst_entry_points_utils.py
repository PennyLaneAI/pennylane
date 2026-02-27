import sys
from importlib import metadata
from typing import List, Callable, Tuple

def _setup_entry_points_from_catalyst(module_name: str, group_name):
    """Returns dunder methods required to import group elements from an entry-point. Entry-point 
    elements are lazy-loaded.

    Args: 
        module_name (str): The name of the module that this function gets called in.
        group_name (str | list(str)): The entry-point group names.

    Returns:
        Tuple(Callable): 
            The module's modified __all__, __getattr__, and __dir__ methods.
    """
    if len(group_name) > 0:
        eps = metadata.entry_points(group=group_name[0])
        for i in range(1, len(group_name)):
            eps += metadata.entry_points(group=group_name[i])
    else:
        # dist = metadata.distribution(module_name.split('.')[0])
        # eps = [ep for ep in dist.entry_points if ep.group == group_name]
        eps = metadata.entry_points(group=group_name)
    
    ep_dict = {ep.name: ep for ep in eps}
    ep_names = list(ep_dict.keys())

    current_module = sys.modules[module_name]

    current_module_funcs = [
        name for name, obj in current_module.__dict__.items()
        if callable(obj) and not name.startswith("_")
    ]

    module_all = list(current_module_funcs + ep_names)

    def module_getattr(name):
        """The new __getattr__ method for the current_module"""
        if name in ep_dict:
            func = ep_dict[name].load()
            func.__module__ = module_name
            setattr(current_module, name, func)
            return func
        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    def module_dir():
        """The new __dir__ method for the current_module"""
        return sorted(set(module_all) | set(current_module.__dict__.keys()))

    return module_all, module_getattr, module_dir
