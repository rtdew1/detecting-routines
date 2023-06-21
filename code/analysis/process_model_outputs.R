rm(list = ls())

library(data.table)
library(tidyverse)
library(ncdf4)

post = nc_open("posterior.nc")

fl = list(
  alpha = aperm(ncvar_get(post, "posterior/alpha"), c(3, 1, 2)),
  mu = aperm(ncvar_get(post, "posterior/mu"), c(2, 1)),
  gamma = aperm(ncvar_get(post, "posterior/gamma"), c(3, 1, 2)),
  eta = aperm(ncvar_get(post, "posterior/eta"), c(3, 1, 2))
)

rm(post)

# NEED TO CHANGE THIS; THIS IS FOR THE SIMULATION
dl = list(W = 38, N = 523)

inputs = cbind(rep(1:7, each=24), rep(0:23, 7))

save.image(file = "analysis_inputs/dl_fl_workspace.RData")

rm(list = ls())

routine = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/sim_cases_routine_898699.csv",
    drop = 1,
    skip = 1
  )
))
random = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/sim_cases_random_898699.csv",
    drop = 1,
    skip = 1
  )
))

sim_decomp = list(routine = routine, random = random)

save(sim_decomp, file = "~/Dropbox/1_proj/via/me/r1/analysis_inputs/sim_decomp.RData")



routine_upper = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/decomp_841601_routine_upper.csv",
    drop = 1,
    skip = 1
  )
))
random_upper = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/decomp_841601_random_upper.csv",
    drop = 1,
    skip = 1
  )
))

routine_lower = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/decomp_841601_routine_lower.csv",
    drop = 1,
    skip = 1
  )
))
random_lower = t(as.matrix(
  fread(
    "~/Dropbox/1_proj/via/me/r1/analysis_inputs/decomp_841601_random_lower.csv",
    drop = 1,
    skip = 1
  )
))

decomp_bounds = list(routine = routine, 
                        routine_lower = routine_lower, 
                        routine_upper = routine_upper, 
                        random = random,
                        random_lower = random_lower,
                        random_upper = random_upper)

save(decomp_bounds, file = "~/Dropbox/1_proj/via/me/r1/analysis_inputs/decomp_bounds.RData")