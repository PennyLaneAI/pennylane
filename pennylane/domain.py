class Domain():
    def __init__(self):
        pass

    #using this for comparison allows to write something like: if cls.par_domain in qml.domain.Reals(): ...
    def __contains__(self, other):
        return issubclass(type(other), self.__class__)

    def __eq__():
        raise NotImplementedError


class Scalars(Domain):
    def __init__(self):
        pass

class Complex(Scalars):
    def __init__(self, non_negative=False):
        pass

class Reals(Scalars):
    def __init__(self, non_negative=False):
        pass

class Ints(Reals):
    def __init__(self, non_negative=False):
        raise NotImplementedError

class Interval(Reals):
    def __init__(self, min, max):
        raise NotImplementedError

class Matrices(Domain):
    def __init__(self, shape):
        raise NotImplementedError

class Unitaries(Matrices):
    def __init__(self, shape, non_negative=False):
        raise NotImplementedError

class Hermitians(Matrices):
    def __init__(self, shape, non_negative=False):
        pass

all_domains = [
    Domain,
    Scalars,
    Complex,
    Reals,
    Ints,
    Interval,
    Matrices,
    Unitaries,
    Hermitians
]

__all__ = [cls.__name__ for cls in all_domains]
