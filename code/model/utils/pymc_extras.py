import aesara.tensor as at
import pymc as pm
import jax

# Fix reshape
# Source: https://github.com/pymc-devs/pymc/issues/5927
from aesara.graph import Constant
from aesara.link.jax.dispatch import jax_funcify
from aesara.tensor.shape import Reshape

@jax_funcify.register(Reshape)
def jax_funcify_Reshape(op, node, **kwargs):
    shape = node.inputs[1]
    if isinstance(shape, Constant):
        constant_shape = shape.data
        def reshape(x, _):
            return jax.numpy.reshape(x, constant_shape)
    else:
        def reshape(x, shape):
            return jax.numpy.reshape(x, shape)
    return reshape

# reference: https://discourse.pymc.io/t/avoiding-looping-when-using-gp-prior-on-latent-variables/9113/9
class FixedMatrixCovariance(pm.gp.cov.Covariance):
    def __init__(self, cov):
        # super().__init__(1, None)
        self.cov = at.as_tensor_variable(cov)
        self.input_dim = 1

    def full(self, X, Xs):
        # covariance matrix known, not explicitly function of X
        return self.cov

    def diag(self, X):
        return at.diag(self.cov)

def condition_zero_cov(cov_free):
    return cov_free[1:, 1:] - (1 / cov_free[0, 0]) * cov_free[1:, 0].dimshuffle(
        0, "x"
    ) * cov_free[0, 1:].dimshuffle("x", 0)


def zero_start_gp(name, L_cond, size):
    gp_free = pm.MvNormal(name + "_free", mu=0, chol=L_cond, size=size)
    gp = at.zeros((size, L_cond.shape[0] + 1))
    gp = at.set_subtensor(gp[:, 0], 0)
    gp = at.set_subtensor(gp[:, 1:], gp_free)
    if size == 1:
        return pm.Deterministic(name, gp[0])
    else:
        return pm.Deterministic(name, gp)