import numpy as np
import pandas as pd
import os
import arviz as az
from numpy import pi as pi

trace = az.from_netcdf(os.path.expanduser("posterior.nc"))
trace.stack(sample=["chain", "draw"], inplace = True)

n_weeks_train = 38
n_people, n_weeks_total, n_samples = trace.posterior['alpha'].shape
n_dayhours = 168

# Posterior median decomp
decomp_routine = np.array([[np.median(np.exp(trace.posterior['eta'].values[i] + trace.posterior['gamma'].values[i,w]).sum(axis = 0)) for w in range(n_weeks_total)] for i in range(n_people)])
decomp_random = np.array([[np.median(np.exp(trace.posterior['alpha'].values[i,w] + trace.posterior['mu'].values).sum(axis = 0)) for w in range(n_weeks_total)] for i in range(n_people)])

pd.DataFrame(decomp_routine).to_csv("decomp_routine.csv")
pd.DataFrame(decomp_random).to_csv("decomp_random.csv")


# Same thing but with 95% intervals
decomp_routine_bounds = np.array([[np.quantile(np.exp(trace.posterior['eta'].values[i] + trace.posterior['gamma'].values[i,w]).sum(axis = 0), q=[0.025,0.5,0.975]) for w in range(n_weeks_total)] for i in range(n_people)])
decomp_random_bounds = np.array([[np.quantile(np.exp(trace.posterior['alpha'].values[i,w] + trace.posterior['mu'].values).sum(axis = 0), q=[0.025,0.5,0.975]) for w in range(n_weeks_total)] for i in range(n_people)])

pd.DataFrame(decomp_routine_bounds[:,:,2]).to_csv(os.path.expanduser("decomp_routine_upper.csv"))
pd.DataFrame(decomp_random_bounds[:,:,2]).to_csv(os.path.expanduser("decomp_random_upper.csv"))
pd.DataFrame(decomp_routine_bounds[:,:,0]).to_csv(os.path.expanduser("decomp_routine_lower.csv"))
pd.DataFrame(decomp_random_bounds[:,:,0]).to_csv(os.path.expanduser("decomp_random_lower.csv"))