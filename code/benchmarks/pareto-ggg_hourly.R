#
# Running Pareto-GGG Model on Routines data
#
# Author: Soham Mahadik
#
library(data.table)
library(BTYDplus)
library(doParallel)
library(abind)
library(data.table)
registerDoParallel(detectCores() - 2)

set.seed(2048)

# Change for run on local directory
setwd("~/routines/data")


################ Read and clean training data ################
routines_train <- read.csv("train.csv", header = TRUE)

# Model need parameters cust (int id), date (in as.POSIXct format), 
# sales (optional)
colnames(routines_train)[which(names(routines_train) == "id")] <-
  "cust"
routines_train$date <- as.POSIXct(
  paste(
    2014,
    routines_train$week,
    routines_train$wday,
    routines_train$hour,
    sep = "-"
  ),
  format = "%Y-%U-%u-%H",
  origin = "1970-01-01",
  tz = "GMT"
)
colnames(routines_train)[which(names(routines_train) == "y")] <-
  "sales"

# Create dfs without sales included
routines_train <- routines_train[, c("cust", "date")]

################ Read and clean test data ################

# Follow same process of cleaning for training data
test0 <- read.csv("test.csv", header = TRUE)
routines_test <- test0

colnames(routines_test)[which(names(routines_test) == "id")] <-
  "cust"
routines_test$date <- as.POSIXct(
  paste(
    2014,
    routines_test$week,
    routines_test$wday,
    routines_test$hour,
    sep = "-"
  ),
  format = "%Y-%U-%u-%H",
  origin = "1970-01-01",
  tz = "GMT"
)
colnames(routines_test)[which(names(routines_test) == "y")] <-
  "sales"

routines_test <- routines_test[, c("cust", "date")]
routines_test_all_vars <- read.csv("test.csv", header = TRUE)

############### Merge Testing and Training Data for Forecasting ###############

# Without Sales
routines <- rbind(routines_train, routines_test)

################ Build Model ################

# Create customer-by-sufficient-statistic summary needed for model
# and run on training data
# T.cal set to last date of training data, everything after date is test data
# to be used for future predictions.
cbs <- elog2cbs(
  routines,
  units = "hour",
  T.cal = max(routines_train$date),
  T.tot = max(routines$date)
)

## COMMENTED OUT IF LOADING OLD WORKSPACE BELOW ----------------------------------
#
# pggg_train.draws <-
#   pggg.mcmc.DrawParameters(
#     cal.cbs = cbs,
#     mcmc = 2500,
#     burnin = 500,
#     chains = 2,
#     thin = 20,
#     mc.cores = 1
#   )
# 
# # Summary plots and outputs (summary, estimated parameter distributions,
# # MCMC convergence)
# summary(pggg_train.draws$level_2)
# 
# 
# 
# # Run future predictions on test data and calculate mean over
# # future transaction draws for each customer
# 
# abind3 = function(...) {
#   abind(..., along=3)
# }
# 
# post_pred_cum_trans = foreach(i=1:1680, .combine=abind3, .multicombine=TRUE) %dopar% {
#     mcmc.DrawFutureTransactions(
#       cbs,
#       pggg_train.draws,
#       T.star = i,
#       sample_size = 1000
#     )
# }
# 
# post_pred_holdout_trans = post_pred_cum_trans[,,1680]




load("~/Dropbox/1_proj/via/me/r1/outputs/benchmarks/pggg_post_pred.RData")

# Doing a little reformatting of the test data...
test0 = as.data.table(test0)
test0$id = factor(test0$id, levels = 1:2000)
test_reqs_by_week = as.matrix(table(test0$id, test0$week))
test_reqs = rowSums(test_reqs_by_week)


# Computing stats:
E_fore_reqs = apply(post_pred_holdout_trans, 2, median)

# Find mean absolute error
abs_err = abs(E_fore_reqs - test_reqs)
t.test(abs_err)

## CONDITIONAL ACCEPT -- Break down the fit

train = t(y_actual[1:38,])
req_std = apply(train, 1, sd)
hist(req_std, breaks=40)

low_req_var = (req_std < median(req_std))
high_req_var = !low_req_var

mean(abs_err[low_req_var])t2
mean(abs_err[high_req_var])

last5_zero = (rowSums(train[,34:38]) == 0)
prev5_zero = (rowSums(train[,29:33]) == 0)
maybe_churned = (last5_zero & !prev5_zero)

mean(abs_err[maybe_churned])
mean(abs_err[!maybe_churned])

test_zero = (rowSums(test_reqs_by_week) == 0)
mean(abs_err[test_zero])
mean(abs_err[!test_zero])


last10_zero = (rowSums(train[,29:38]) == 0)
prev10_zero = (rowSums(train[,19:28]) == 0)
maybe_churned10 = (last10_zero & !prev10_zero)

mean(abs_err[maybe_churned10])
mean(abs_err[!maybe_churned10])
