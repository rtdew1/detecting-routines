# Base imports
import numpy as np

# PyMC-related imports
import pymc as pm
import aesara
import aesara.tensor as at

# Other imports
from numpy import pi as pi

# Custom imports
from utils.pymc_extras import FixedMatrixCovariance, jax_funcify_Reshape


def create_model(y, nz_mask, include_obs, n_week_fore, args):

    n_cust, n_week_train, n_dayhour = y.shape
    n_week_total = n_week_train + n_week_fore

    # inputs are always just the range of possible values, used for creating GPs later
    cust_inputs = np.arange(1, n_cust + 1)[:, None]
    week_inputs = np.arange(1, n_week_total + 1)[:, None]
    day_inputs = np.arange(1, 8)[:, None]
    hour_inputs = np.arange(0, 24)[:, None]

    routines_model = pm.Model()
    with routines_model:
        # Create aesara objects from data:
        y_at = aesara.shared(y)
        nz_mask_at = aesara.shared(nz_mask)
        include_obs_at = aesara.shared(include_obs)

        # This will be used several times for hierarchical GPs:
        identity_matrix = at.eye(n_cust)
        identity_cov = FixedMatrixCovariance(identity_matrix)

        # GP model for random scaling term, alpha: ----------------------------------

        if args.HIER_ICEPTS:
            alpha0_scale = pm.HalfNormal("alpha0_scale", sigma=10)
            if args.EST_ICEPT_LOC:
                alpha0_loc = pm.Normal("alpha0_loc", mu=0, sigma=10)
            else:
                alpha0_loc = 0
            alpha0 = pm.Normal(
                "alpha0", mu=alpha0_loc, sigma=alpha0_scale, shape=(n_cust)
            )
        else:
            alpha0 = pm.Normal("alpha0", mu=0, sigma=10)

        ## Hyperparameters
        alpha_amp = pm.HalfNormal("alpha_amp")
        alpha_ls = pm.InverseGamma("alpha_ls", alpha=5, beta=5)

        ## Alpha cov
        if args.COV_TYPE == "expon":
            cov_alpha = alpha_amp**2 * pm.gp.cov.Exponential(input_dim=1, ls=alpha_ls)
        elif args.COV_TYPE == "mat32":
            cov_alpha = alpha_amp**2 * pm.gp.cov.Matern32(input_dim=1, ls=alpha_ls)
        elif args.COV_TYPE == "mat52":
            cov_alpha = alpha_amp**2 * pm.gp.cov.Matern52(input_dim=1, ls=alpha_ls)

        ## Alpha prior
        alpha_offset_gp = pm.gp.LatentKron(cov_funcs=[identity_cov, cov_alpha])
        alpha_offset = alpha_offset_gp.prior(
            "alpha_offset", Xs=[cust_inputs, week_inputs]
        )
        if args.HIER_ICEPTS:
            alpha = pm.Deterministic(
                "alpha",
                alpha0.dimshuffle(0, "x")
                + alpha_offset.reshape((n_cust, n_week_total)),
            )
        else:
            alpha = pm.Deterministic(
                "alpha",
                alpha0 + alpha_offset.reshape((n_cust, n_week_total)),
            )

        # Common dayhour rate term, mu ----------------------------------------------
        ## Day correlation term:
        mu_omega_chol, mu_omega_corr, mu_omega_scale = pm.LKJCholeskyCov(
            "mu_omega",
            n=7,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(shape=7),
            compute_corr=True,
        )
        cov_mu_day = FixedMatrixCovariance(mu_omega_corr)

        ## Periodic hour term:
        mu_amp = pm.HalfNormal("mu_amp")
        mu_ls = pm.TruncatedNormal("mu_ls", mu=0.5 * pi, sigma=0.25 * pi, lower=0)
        cov_mu_periodic = mu_amp**2 * pm.gp.cov.Periodic(
            input_dim=1,
            period=24,
            ls=mu_ls / 2,  # note: /2 needed to recover original per kernel defn
        )

        ## Combine using Kronecker structure:
        mu_gp = pm.gp.LatentKron(cov_funcs=[cov_mu_day, cov_mu_periodic])
        mu = mu_gp.prior("mu", Xs=[day_inputs, hour_inputs])

        # GP model for routine scaling term, gamma ----------------------------------
        ## Define gamma = gamma0 + gamma_offset

        if args.HIER_ICEPTS:
            gamma0_scale = pm.HalfNormal("gamma0_scale", sigma=10)
            if args.EST_ICEPT_LOC:
                gamma0_loc = pm.Normal("gamma0_loc", mu=0, sigma=10)
            else:
                gamma0_loc = 0
            gamma0 = pm.Normal(
                "gamma0", mu=gamma0_loc, sigma=gamma0_scale, shape=(n_cust)
            )
        else:
            gamma0 = pm.Normal("gamma0", mu=0, sigma=10)

        ## Hyperparameters
        gamma_amp = pm.HalfNormal("gamma_amp")
        gamma_ls = pm.InverseGamma("gamma_ls", alpha=5, beta=11)

        ## Gamma Covariance
        if args.COV_TYPE == "expon":
            cov_gamma = gamma_amp**2 * pm.gp.cov.Exponential(input_dim=1, ls=gamma_ls)
        elif args.COV_TYPE == "mat32":
            cov_gamma = gamma_amp**2 * pm.gp.cov.Matern32(input_dim=1, ls=gamma_ls)
        elif args.COV_TYPE == "mat52":
            cov_gamma = gamma_amp**2 * pm.gp.cov.Matern52(input_dim=1, ls=gamma_ls)

        ## Gamma prior
        gamma_offset_gp = pm.gp.LatentKron(cov_funcs=[identity_cov, cov_gamma])
        gamma_offset = gamma_offset_gp.prior(
            "gamma_offset", Xs=[cust_inputs, week_inputs]
        )

        if args.HIER_ICEPTS:
            gamma = pm.Deterministic(
                "gamma",
                gamma0.dimshuffle(0, "x")
                + gamma_offset.reshape((n_cust, n_week_total)),
            )
        else:
            gamma = pm.Deterministic(
                "gamma",
                gamma0 + gamma_offset.reshape((n_cust, n_week_total)),
            )

        # GP model for routine rate, eta ---------------------------------------------

        ## Day correlation term:
        eta_omega_chol, eta_omega_corr, eta_omega_scale = pm.LKJCholeskyCov(
            "eta_omega",
            n=7,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(shape=7),
            compute_corr=True,
        )
        cov_eta_day = FixedMatrixCovariance(eta_omega_corr)

        ## Periodic hour term:
        eta_amp = pm.HalfNormal("eta_amp")
        eta_ls = pm.TruncatedNormal("eta_ls", mu=0.5 * pi, sigma=0.25 * pi, lower=0)
        cov_eta_periodic = eta_amp**2 * pm.gp.cov.Periodic(
            input_dim=1,
            period=24,
            ls=eta_ls / 2,  # note: /2 needed to recover original per kernel defn
        )

        ## Combine using Kronecker structure:
        eta_gp = pm.gp.LatentKron(
            cov_funcs=[identity_cov, cov_eta_day, cov_eta_periodic]
        )
        eta_unshaped = eta_gp.prior(
            "eta_unshaped", Xs=[cust_inputs, day_inputs, hour_inputs]
        )
        eta = pm.Deterministic("eta", eta_unshaped.reshape((n_cust, n_dayhour)))

        # Compute likelihood -----------------------------------------------------
        intensity = at.exp(
            alpha.dimshuffle(0, 1, "x")[:, :n_week_train, :]
            + mu.dimshuffle("x", "x", 0)
        ) + at.exp(
            gamma.dimshuffle(0, 1, "x")[:, :n_week_train, :] + eta.dimshuffle(0, "x", 1)
        )
        lp1 = at.sum(y_at[nz_mask_at] * at.log(intensity[nz_mask_at]))
        lp2 = at.sum(intensity[include_obs_at])
        pm.Potential("lp", lp1 - lp2)

    return routines_model
