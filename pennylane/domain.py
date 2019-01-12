class Domain():
    def __init__(self):
        pass

    #using this for comparison allows to write something like: if cls.par_domain in qml.domain.Reals(): ...
    def __contains__(self, other):
        return issubclass(type(self), other)

    def __eq__():
        raise NotImplementedError


class Scalars(Domain):
    def __init__(self):
        raise NotImplementedError

class Complex(Scalars):
    def __init__(self, non_negative=false):
        raise NotImplementedError

class Reals(Scalars):
    def __init__(self, non_negative=false):
        raise NotImplementedError

class Ints(Reals):
    def __init__(self, non_negative=false):
        raise NotImplementedError

class Interval(Reals):
    def __init__(self, min, max):
        raise NotImplementedError

class Matrices(Domain):
    def __init__(self, shape):
        raise NotImplementedError

class Unitaries(Matrices):
    def __init__(self, shape, non_negative=false):
        raise NotImplementedError

class Hermitians(Matrices):
    def __init__(self, shape, non_negative=false):
        pass

__all__ = [cls.__name__ for cls in all_ops]
