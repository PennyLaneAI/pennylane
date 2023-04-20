
backends = []

try:
	from . import jax_backend
	backends.append(jax_backend)
except ImportError:
	pass

def register_pytree_with_keys(cls, flatten_fun, unflatten_fun, static_fields):
	for backend in backends:
		backend.register_pytree_with_keys(cls, flatten_fun, unflatten_fun, static_fields)