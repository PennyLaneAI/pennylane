


# pylint: disable=import-outside-toplevel
def _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn):

    try:
        import jax
    except ImportError:
        return

    jax.tree_util.register_pytree_node(pytree_type, flatten_fn, unflatten_fn)
    return


def register_pytree(pytree_type, flatten_fn, unflatten_fn):

    _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)

    return