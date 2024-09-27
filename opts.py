from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Int, PRNGKeyArray, PyTree


class Momentum(eqx.Module):
    """The momentum for any optimizer.
    Just computes a running average of the gradients, and returns it."""

    discounting: float = 0.9
    state: PyTree | None = None

    def update(self, grad):
        if self.state is None:
            return eqx.tree_at(lambda s: s.state, self, grad)

        new_state = jax.tree.map(
            lambda x, y: self.discounting * x + (1 - self.discounting) * y,
            self.state,
            grad,
        )
        return eqx.tree_at(lambda s: s.state, self, new_state)

    def get(self):
        if self.state is None:
            raise AssertionError("Momentum state is not initialized yet!")
        return self.state

    def bias_corrected(self, t):
        return eqx.tree_at(
            lambda s: s.state,
            self,
            jax.tree.map(lambda m: m / (1.0 - self.discounting**t), self.state),
        )


class Preconditioner(eqx.Module):
    """Base class for all the preconditioners out there"""

    def update(self, grad):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class AdaGradConditioner(Preconditioner):
    """AdaGrad preconditioner.
    Keeps track of sum of squares of gradients, and approximates the Hessian diagonal with it."""

    sq_grads: PyTree | None = None
    discounting: float = 0.999
    eps: float = 1e-8

    def update(self, grad):
        if self.sq_grads is None:
            return eqx.tree_at(lambda s: s.sq_grads, self, grad**2)

        new_sq_grads = jax.tree.map(lambda x, y: x + y**2, self.sq_grads, grad)
        return eqx.tree_at(lambda s: s.sq_grads, self, new_sq_grads)

    def apply(self, grads):
        if self.sq_grads is None:
            raise AssertionError("AdaGrad state is not initialized yet!")
        return jax.tree.map(lambda x, y: x / (jnp.sqrt(y) + self.eps), grads, self.sq_grads)

    def bias_corrected(self, t):
        return eqx.tree_at(
            lambda s: s.sq_grads,
            self,
            jax.tree.map(lambda m: m / (1.0 - self.discounting**t), self.sq_grads),
        )


class Optimizer(eqx.Module):
    """Base class for all the optimizers."""

    def step(self, grad):
        raise NotImplementedError


class AdaGrad(Optimizer):
    """AdaGrad optimizer."""

    conditioner: AdaGradConditioner = AdaGradConditioner()

    def step(self, grad):
        new_conditioner = self.conditioner.update(grad)
        updates = new_conditioner.apply(grad)
        return AdaGrad(conditioner=new_conditioner), updates


class Adam(Optimizer):
    """Adam optimizer."""

    momentum: Momentum = Momentum(discounting=0.9)
    conditioner: AdaGradConditioner = AdaGradConditioner(discounting=0.999)
    t: Int[Array, ""] = jnp.array(0)

    def step(self, grad):
        t = self.t + 1
        m_hat = self.momentum.update(grad)
        v_hat = self.conditioner.update(grad)

        averaged_gradient = m_hat.bias_correctd(time=t).get()
        updates = v_hat.bias_corrected(time=t).apply(averaged_gradient)

        updates = jax.tree.map(lambda x: x * 1e-3, updates)

        return Adam(momentum=m_hat, conditioner=v_hat, t=t), updates


class Solver(eqx.Module):
    """Differential equation solver base class."""

    def step_with(self, opt, grad_fn_with_params, key):
        raise NotImplementedError


class Euler(eqx.Module):
    """Euler solver for differential equations."""

    optimizer: Optimizer

    def step_with(self, grad_fn_with_params, params, key):
        grad = grad_fn_with_params(params, key)
        new_optimizer, updates = self.optimizer.step(grad)
        return eqx.tree_at(lambda s: s.optimizer, self, new_optimizer), updates


class Heun(eqx.Module):
    """Heun solver for differential equations."""

    optimizer: Optimizer

    def step_with(self, grad_fn_with_params, params, key):
        first_key, second_key = jr.split(key)
        # first step: same as euler
        start_grad = grad_fn_with_params(params, first_key)
        new_optimizer, start_updates = self.optimizer.step(start_grad)
        # apply updates
        new_params = eqx.apply_updates(params, start_updates)
        # second step: compute the gradient at the next point
        end_grad = grad_fn_with_params(new_params, second_key)
        final_optimizer, end_updates = new_optimizer.step(end_grad)

        # apply the average between the two updates
        averaged_updates = jax.tree.map(lambda x, y: (x + y) / 2, start_updates, end_updates)
        return eqx.tree_at(lambda s: s.optimizer, self, final_optimizer), eqx.apply_updates(
            params, averaged_updates
        )


class HeunGrad(eqx.Module):
    """Heun solver, but averages the raw gradients instead of the updates with a single optimizer."""

    optimizer: Optimizer

    def step_with(self, grad_fn_with_params, params, key):
        first_key, second_key = jr.split(key)
        # first step: same as euler
        start_grad = grad_fn_with_params(params, first_key)
        new_optimizer, start_updates = self.optimizer.step(start_grad)
        # apply updates
        new_params = eqx.apply_updates(params, start_updates)
        # second step: compute the gradient at the next point
        end_grad = grad_fn_with_params(new_params, second_key)
        end_optimizer, end_updates = new_optimizer.step(end_grad)

        # apply the average between the two updates
        averaged_updates = jax.tree.map(lambda x, y: (x + y) / 2, start_updates, end_updates)
        final_optimizer, final_updates = self.optimizer.step(averaged_updates)
        return eqx.tree_at(lambda s: s.optimizer, self, final_optimizer), final_updates


class HeunGradDouble(eqx.Module):
    """Heun solver, but averages the raw gradients instead of the updates."""

    fast_optimizer: Optimizer  # optimizer for the intermediate steps
    slow_optimizer: Optimizer  # optimizer for the averaged steps

    def step_with(self, grad_fn_with_params, params, key):
        first_key, second_key = jr.split(key)
        # first step: same as euler
        start_grad = grad_fn_with_params(params, first_key)
        new_fast_optimizer, start_updates = self.fast_optimizer.step(start_grad)
        end_params = eqx.apply_updates(params, start_updates)

        # second step: compute the gradient at the next point
        end_grad = grad_fn_with_params(end_params, second_key)
        new_fast_optimizer, _ = new_fast_optimizer.step(end_grad)  # just update the opt state

        # average the gradients
        averaged_grads = jax.tree.map(lambda x, y: (x + y) / 2, start_grad, end_grad)
        new_slow_optimizer, final_updates = self.slow_optimizer.step(averaged_grads)

        new_self = self.__class__(
            fast_optimizer=new_fast_optimizer, slow_optimizer=new_slow_optimizer
        )

        return new_self, final_updates
