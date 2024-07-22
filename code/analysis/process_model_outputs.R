# Usage: make sure to set working directory to source file location,
# then simply run the script, after sestting these options:

# REVISE THIS FILEPATH TO POINT TO THE DIRECTORY WHERE posterior.nc IS LOCATED
results_dir = "../model/outputs/routines_model_2024-07-19_16-20-30-747185"

library(data.table)
library(tidyverse)
library(ncdf4)

# Read in training and test data
train = fread("../../data/train.csv")
test = fread("../../data/test.csv")
W = max(train$week)
W_star = max(test$week) - W


## PROCESS THE POSTERIOR ------------------------------------

# Load posterior:
post = nc_open(paste(results_dir, "/posterior.nc", sep=""))

# Process the posterior into a "Stan-like" fit list object:
fl = list(
  alpha = aperm(ncvar_get(post, "posterior/alpha"), c(3, 1, 2)),
  mu = aperm(ncvar_get(post, "posterior/mu"), c(2, 1)),
  gamma = aperm(ncvar_get(post, "posterior/gamma"), c(3, 1, 2)),
  eta = aperm(ncvar_get(post, "posterior/eta"), c(3, 1, 2))
)

# Remove 'post' to save memory::
rm(post)

# Create a few useful objects (dl = data list, inputs = useful indices)
n_cust = dim(fl$alpha)[3] # (Note: this is important, because before running the 
                          #  model, we filter by a certain number of customers; 
                          #  it can't be read from train or test)


dl = list(W = W,
          W_star = W_star,
          N = n_cust,
          id = train$id[train$id <= n_cust],
          id_star = test$id[test$id <= n_cust],
          dayhour = train$dayhour[train$id <= n_cust],
          dayhour_star = test$dayhour[test$id <= n_cust],
          week = train$week[train$id <= n_cust],
          week_star = test$week[test$id <= n_cust],
          y = train$y[train$id <= n_cust],
          y_star = test$y[test$id <= n_cust])

inputs = cbind(rep(1:7, each=24), rep(0:23, 7))

# Save fl, dl, and inputs as a workspace, for easier use in subsequent analyses:
save.image(file = "analysis_inputs/dl_fl_workspace.RData")

# Remove everything created thus far, again for memory management:
rm(fl, dl, inputs)


## COMPUTE PERSON/WEEK SUMMARIES OF DATA ------------------------------------

# Create summaries of the data by week (num requests per week):
y_training = data.frame(train[,c("id","week","y")]) %>%
  filter(id <= n_cust) %>%
  group_by(week, id) %>%
  summarise(y = sum(y), .groups = 'drop')

y_test = data.frame(test[,c("id","week","y")]) %>%
  filter(id <= n_cust) %>%
  group_by(week, id) %>%
  summarise(y = sum(y), .groups = 'drop')

# Combine y_training and y_test
combined_data <- bind_rows(y_training, y_test) 

# Pivot the data to create the desired matrix
y_actual <- combined_data %>%
  pivot_wider(names_from = id, values_from = y, values_fill = 0)

# Convert to matrix
y_actual <- as.matrix(y_actual %>% select(-week))
rownames(y_actual) <- unique(combined_data$week)

sorted_ids <- order(as.numeric(colnames(y_actual)))
y_actual <- y_actual[, sorted_ids]

save(y_training, y_test, y_actual, file="analysis_inputs/y_nsessions.RData")

# Clear memory:
rm(y_training, y_test, y_actual)


## PROCESS THE ROUTINE DECOMPOSITION ------------------------------------
routine = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_routine.csv", sep=""),
    drop = 1,
    skip = 1
  )
))
random = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_random.csv", sep=""),
    drop = 1,
    skip = 1
  )
))
decomp = list(routine = routine, random = random)
save(decomp, file = "analysis_inputs/decomp.RData")



routine_upper = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_routine_upper.csv", sep=""),
    drop = 1,
    skip = 1
  )
))
random_upper = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_random_upper.csv", sep=""),
    drop = 1,
    skip = 1
  )
))
routine_lower = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_routine_lower.csv", sep=""),
    drop = 1,
    skip = 1
  )
))
random_lower = t(as.matrix(
  fread(
    paste(results_dir, "/decomp_random_lower.csv", sep=""),
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

save(decomp_bounds, file = "analysis_inputs/decomp_bounds.RData")