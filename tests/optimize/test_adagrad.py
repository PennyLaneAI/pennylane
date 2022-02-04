import pytest

from pennylane import numpy as np
from pennylane.optimize import AdagradOptimizer


class TestAdagradOptimizer:
    """Test the AdaGrad (adaptive gradient) optimizer"""

    @pytest.mark.parametrize(
        "grad,args",
        [
            ([40, -4, 12, -17, 400], [0, 30, 6, -7, 800]),
            ([0.00033, 0.45e-5, 0.0], [1.3, -0.5, 8e3]),
            ([43], [0.8]),
        ],
    )
    def test_apply_grad(self, grad, args, tol):
        """
        Test that the gradient can be applied correctly to a set of parameters
        and that accumulation works correctly.
        """
        stepsize, eps = 0.1, 1e-8
        sgd_opt = AdagradOptimizer(stepsize, eps=eps)
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        a1 = grad**2
        expected = args - stepsize / np.sqrt(a1 + eps) * grad
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

        # Simulate a new step
        grad = grad + args
        args = expected

        a2 = a1 + grad**2
        expected = args - stepsize / np.sqrt(a2 + eps) * grad
        res = sgd_opt.apply_grad(grad, args)
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    def test_adagrad_optimizer_univar(self, x_start, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for univariate functions."""
        stepsize = 0.1
        adag_opt = AdagradOptimizer(stepsize)

        univariate_funcs = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x**2]
        grad_uni_fns = [
            lambda x: (np.cos(x),),
            lambda x: (np.exp(x / 10.0) / 10.0,),
            lambda x: (2 * x,),
        ]

        for gradf, f in zip(grad_uni_fns, univariate_funcs):
            adag_opt.reset()

            x_onestep = adag_opt.step(f, x_start)
            past_grads = gradf(x_start)[0] * gradf(x_start)[0]
            adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
            x_onestep_target = x_start - gradf(x_start)[0] * adapt_stepsize
            assert np.allclose(x_onestep, x_onestep_target, atol=tol)

            x_twosteps = adag_opt.step(f, x_onestep)
            past_grads = (
                gradf(x_start)[0] * gradf(x_start)[0] + gradf(x_onestep)[0] * gradf(x_onestep)[0]
            )
            adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
            x_twosteps_target = x_onestep - gradf(x_onestep)[0] * adapt_stepsize
            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)

    def test_adagrad_optimizer_multivar(self, tol):
        """Tests that adagrad optimizer takes one and two steps correctly
        for multivariate functions."""
        stepsize = 0.1
        adag_opt = AdagradOptimizer(stepsize)

        multivariate_funcs = [
            lambda x: np.sin(x[0]) + np.cos(x[1]),
            lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
            lambda x: np.sum([x_**2 for x_ in x]),
        ]
        grad_multi_funcs = [
            lambda x: (np.array([np.cos(x[0]), -np.sin(x[1])]),),
            lambda x: (
                np.array(
                    [
                        np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                        np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2),
                    ]
                ),
            ),
            lambda x: (np.array([2 * x_ for x_ in x]),),
        ]

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for gradf, f in zip(grad_multi_funcs, multivariate_funcs):
            for jdx in range(len(x_vals[:-1])):
                adag_opt.reset()

                x_vec = x_vals[jdx : jdx + 2]
                x_onestep = adag_opt.step(f, x_vec)
                past_grads = gradf(x_vec)[0] * gradf(x_vec)[0]
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_onestep_target = x_vec - gradf(x_vec)[0] * adapt_stepsize
                assert np.allclose(x_onestep, x_onestep_target, atol=tol)

                x_twosteps = adag_opt.step(f, x_onestep)
                past_grads = (
                    gradf(x_vec)[0] * gradf(x_vec)[0] + gradf(x_onestep)[0] * gradf(x_onestep)[0]
                )
                adapt_stepsize = stepsize / np.sqrt(past_grads + 1e-8)
                x_twosteps_target = x_onestep - gradf(x_onestep)[0] * adapt_stepsize
                assert np.allclose(x_twosteps, x_twosteps_target, atol=tol)
