from functools import lru_cache


has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


ABSTRACT_TYPE_CACHE = {}


def construct_abstract_type(type_name: str) -> type:

    if not has_jax:
        return None

    if type_name in ABSTRACT_TYPE_CACHE:
        return ABSTRACT_TYPE_CACHE[type_name]

    namespace = {
        "__eq__": lambda self, other: isinstance(other, type(self)),
        "__hash__": lambda self: hash(type_name),
    }
    AbstractType = type(type_name, (jax.core.AbstractValue,), namespace)

    jax.core.raise_to_shaped_mappings[AbstractType] = lambda aval, _: aval
    ABSTRACT_TYPE_CACHE[type_name] = AbstractType

    return AbstractType


class Meta(type):
    def __init__(cls, *_, **__):
        if not has_jax:
            cls._primitive = None
            return

        cls._primitive = jax.core.Primitive(cls.__name__)

        @cls._primitive.def_impl
        def default_call(*args, **kwargs):
            inst = cls.__new__(cls, *args, **kwargs)
            cls.__init__(inst, *args, **kwargs)
            return inst

        # -1 is object
        # -2 is top parent
        top_parent = cls.__mro__[-2]
        abstract_name = f"Abstract{top_parent.__name__}"
        abstract_type = construct_abstract_type(abstract_name)

        @cls._primitive.def_abstract_eval
        def abstract_init(*args, **kwargs):
            return abstract_type()

    def __call__(cls, *args, **kwargs):
        if has_jax:
            return cls._primitive.bind(*args, **kwargs)
        return super().__call__(cls, *args, **kwargs)
