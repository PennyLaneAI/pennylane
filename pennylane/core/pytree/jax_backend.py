from functools import partial

import jax


def register_pytree_with_keys(cls, flatten_fun, unflatten_fun, static_fields, cache_fields):
    if hasattr(jax.tree_util, "register_pytree_with_keys"):
        jax.tree_util.register_pytree_with_keys(
            cls,
            partial(
                flatten_fun,
                static_fields,
                cache_fields,
                with_key_paths=True,
            ),
            unflatten_fun,
        )
    else:
        jax.tree_util.register_pytree_node(
            cls,
            partial(
                flatten_fun,
                static_fields,
                cache_fields,
                with_key_paths=False,
            ),
            unflatten_fun,
        )
