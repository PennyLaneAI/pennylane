import abc

class OperatorError(abc.ABC):
    """A general class to represent the Operator Norm Error"""

    def __init__(self, error=0):
        self.error = error
    
    @abc.abstractclassmethod
    def __add__(self, other):
        """How errors are combined"""
        return NotImplemented
    
    @staticmethod
    def get_error(*args, **kwargs):
        """Compute the operator error for a given operator"""
        raise NotImplemented
    
    def __repr__(self) -> str:
        return f"{self.error}"