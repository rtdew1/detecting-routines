# Usage: see ../runfile.sh, which contains the basic command for using this script

import os

# PyMC-related imports
import jax
import pymc.sampling_jax
from numpyro.infer.initialization import init_to_median

print(f"JAX default backend: {jax.default_backend()}")

# Custom imports
from utils.config import args
from utils.data_io.process_data import *
from main.model import create_model

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs(f"outputs/{args.OUTPUT_NAME}", exist_ok=True)

# Create model
n_weeks_train = train.week.max()
n_weeks_test = test.week.nunique()
routines_model = create_model(y, nz_mask, include_obs, n_week_fore=n_weeks_test, args=args)

# Sampling

with routines_model:
    trace = pymc.sampling_jax.sample_numpyro_nuts(
        tune=args.WARMUP,
        draws=args.SAMPLES,
        chains=args.CHAINS,
        postprocessing_backend="cpu",
        idata_kwargs=dict(log_likelihood=False),
        nuts_kwargs={"init_strategy": init_to_median()},
        # progress_bar=False,
    )

# Save args to text file, to track settings:
with open(f"outputs/{args.OUTPUT_NAME}/summary.txt", "w") as f:
    print(args, file=f)

print("Saving posterior.nc")
trace.to_netcdf(filename=f"outputs/{args.OUTPUT_NAME}/posterior.nc")

# Stack for easier indexing:
print("Stacking results... (may take some time)")
trace.stack(sample=["chain", "draw"], inplace = True)

# Get dimensions
n_people, n_weeks_total, n_samples = trace.posterior['alpha'].shape
n_dayhours = 168

# Posterior median decomp
print("Computing posterior median use decomposition...")
decomp_routine = np.array([[np.median(np.exp(trace.posterior['eta'].values[i] + trace.posterior['gamma'].values[i,w]).sum(axis = 0)) for w in range(n_weeks_total)] for i in range(n_people)])
decomp_random = np.array([[np.median(np.exp(trace.posterior['alpha'].values[i,w] + trace.posterior['mu'].values).sum(axis = 0)) for w in range(n_weeks_total)] for i in range(n_people)])

pd.DataFrame(decomp_routine).to_csv(os.path.join("outputs", args.OUTPUT_NAME,"decomp_routine.csv"))
pd.DataFrame(decomp_random).to_csv(os.path.join("outputs", args.OUTPUT_NAME,"decomp_random.csv"))


# Same thing but with 95% intervals
print("Computing 95% posterior intervals of use decomposition...")
decomp_routine_bounds = np.array([[np.quantile(np.exp(trace.posterior['eta'].values[i] + trace.posterior['gamma'].values[i,w]).sum(axis = 0), q=[0.025,0.5,0.975]) for w in range(n_weeks_total)] for i in range(n_people)])
decomp_random_bounds = np.array([[np.quantile(np.exp(trace.posterior['alpha'].values[i,w] + trace.posterior['mu'].values).sum(axis = 0), q=[0.025,0.5,0.975]) for w in range(n_weeks_total)] for i in range(n_people)])

pd.DataFrame(decomp_routine_bounds[:,:,2]).to_csv(os.path.join("outputs", args.OUTPUT_NAME, "decomp_routine_upper.csv"))
pd.DataFrame(decomp_random_bounds[:,:,2]).to_csv(os.path.join("outputs", args.OUTPUT_NAME, "decomp_random_upper.csv"))
pd.DataFrame(decomp_routine_bounds[:,:,0]).to_csv(os.path.join("outputs", args.OUTPUT_NAME, "decomp_routine_lower.csv"))
pd.DataFrame(decomp_random_bounds[:,:,0]).to_csv(os.path.join("outputs", args.OUTPUT_NAME, "decomp_random_lower.csv"))