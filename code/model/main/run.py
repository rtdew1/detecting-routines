# PyMC-related imports
import jax
import pymc.sampling_jax
from numpyro.infer.initialization import init_to_median

print(f"JAX default backend: {jax.default_backend()}")

# Custom imports
from utils.config import args
from utils.data_io.og_data import *
from main.model import create_model

# Create model

routines_model = create_model(y, nz_mask, include_obs, n_week_fore=10, args=args)

# Sampling

with routines_model:
    samples = pymc.sampling_jax.sample_numpyro_nuts(
        tune=args.WARMUP,
        draws=args.SAMPLES,
        chains=args.CHAINS,
        postprocessing_backend="cpu",
        idata_kwargs=dict(log_likelihood=False),
        nuts_kwargs={"init_strategy": init_to_median()},
        # progress_bar=False,
    )

# Save args to text file, to track settings:
os.chdir(os.path.expanduser(args.MAIN_DIR))

with open(f"{args.OUTPUT_FILE}.txt", "w") as f:
    print(args, file=f)

samples.to_netcdf(filename=f"{args.OUTPUT_FILE}.nc")
