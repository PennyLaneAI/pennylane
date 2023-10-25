import numpy as np


class OperatorError():
    """A general class to represent the Operator Norm Error"""

    def __init__(self, error=0):
        self.error = error
    
    def __add__(self, other):
        """Simple case, its additive"""
        return OperatorError(self.error + other.error)
    
    @staticmethod
    def get_error(op):
        """Compute the operator error for a given operator"""
        raise NotImplemented
    
    def __repr__(self) -> str:
        return f"{self.error}"