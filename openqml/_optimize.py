import autograd


class GradientDescentOptimizer(object):
    """Base class for gradient-descent-based optimizers."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, objective_fn, x, grad_fn=None):
        g = self.compute_grad(objective_fn, x, grad_fn=grad_fn)
        x_out = self.apply_grad(g, x)
        return x_out

    def compute_grad(self, objective_fn, x, grad_fn=None):
        """Compute gradient of objective_fn at the point x"""
        if grad_fn is not None:
            g = grad_fn(x)  # just call the supplied grad function
        else:
            g = autograd.grad(objective_fn)(x)  # default is autograd
        return g

    def apply_grad(self, grad, x):
        """Update x to take a single optimization step"""
        return x - self.learning_rate * grad


class MomentumOptimizer(GradientDescentOptimizer):
    """Gradient-descent optimizer with momentum."""

    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.accumulation = 0.

    def apply_grad(self, grad, x):
        self.accumulation = self.momentum * self.accumulation + grad # todo: check momentum formulation
        return x - self.learning_rate * self.accumulation

