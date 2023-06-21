rm(list = ls())

set.seed(7926)

library(data.table)
library(xtable)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(doParallel)
library(foreach)
library(ncdf4)

registerDoParallel(cores = detectCores() - 2)

load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")
load("analysis_inputs/decomp_bounds.RData"))

post = nc_open("posterior.nc")

fl = list(
  alpha = aperm(ncvar_get(post, "posterior/alpha"), c(3, 1, 2)),
  mu = aperm(ncvar_get(post, "posterior/mu"), c(2, 1)),
  gamma = aperm(ncvar_get(post, "posterior/gamma"), c(3, 1, 2)),
  eta = aperm(ncvar_get(post, "posterior/eta"), c(3, 1, 2))
)

routine = t(as.matrix(
  fread(
    "decomp_routine.csv",
    drop = 1,
    skip = 1
  )
))
random = t(as.matrix(
  fread(
    "decomp_random.csv",
    drop = 1,
    skip = 1
  )
))


# ## The original "method 2": Condition on number of rides, for each iteration, predict the top times will be requests
# 
# cond.top.cp = function(fl, dl) {
#   cp = matrix(NA, dl$N, dl$W + dl$W_star)
#   for (i in 1:dl$N) {
#     for (w in 1:(dl$W + dl$W_star)) {
#       true_times = c(unique(dl$dayhour[dl$week == w & dl$id == i]),
#                      unique(dl$dayhour_star[dl$week_star == w &
#                                               dl$id_star == i]))
#       M = length(true_times)
#       
#       if (M == 0) {
#         next
#       }
#       
#       alpha = fl$alpha[, w, i]
#       mu = fl$mu
#       
#       rate = exp(alpha + mu)
#       if (exists("eta", fl)) {
#         eta = fl$eta[, , i]
#         gamma = fl$gamma[, w, i]
#         rate = rate + exp(gamma + eta)
#       }
#       pred_times = order(colMeans(rate), decreasing = TRUE)[1:M]
#       cp[i, w] = sum(pred_times %in% true_times) / M
#     }
#   }
#   return(cp)
# }
# 
# full_cp = cond.top.cp(fl, dl)
# 
# full_cp_insamp = mean(full_cp[, 1:38], na.rm = TRUE)
# full_cp_fore = mean(full_cp[, 39:48], na.rm = TRUE)
# 
# c(full_cp_insamp, full_cp_fore)
# 
# 
# 
# # Mean average precision
# 
# full_map = matrix(NA, dl$N, dl$W + dl$W_star)
# for (i in 1:dl$N) {
#   wks_i = c(dl$week[dl$id == i], dl$week_star[dl$id_star == i])
#   j_i = c(dl$dayhour[dl$id == i], dl$dayhour_star[dl$id_star == i])
#   map_i = c()
#   for (w in unique(wks_i)) {
#     wk_rank = order(colMeans(exp(fl$alpha[, w, i] + fl$mu) + exp(fl$gamma[, w, i] + fl$eta[, , i])), decreasing = TRUE)
#     relevant_ranks = sort(which(wk_rank %in% j_i[wks_i == w]))
#     full_map[i, w] = mean(1:length(relevant_ranks) / relevant_ranks)
#   }
# }
# 
# full_map_insamp = mean(full_map[, 1:38], na.rm = TRUE)
# full_map_fore = mean(full_map[, 39:48], na.rm = TRUE)
# 
# rbind(c(full_map_insamp, full_map_fore))
# 
# 
# 
# 
# 
# ## NEW conditional precision - condition on only the number of routine rides
# 
# cond.routine.cp = function(fl, dl, routine, extra = 0) {
#   cp = matrix(NA, dl$N, dl$W + dl$W_star)
#   for (i in 1:dl$N) {
#     for (w in 1:(dl$W + dl$W_star)) {
#       true_times = c(unique(dl$dayhour[dl$week == w & dl$id == i]),
#                      unique(dl$dayhour_star[dl$week_star == w &
#                                               dl$id_star == i]))
#       M = min(round(routine[w, i]), length(true_times))
#       
#       if (M == 0) {
#         next
#       }
#       
#       alpha = fl$alpha[, w, i]
#       mu = fl$mu
#       
#       rate = exp(alpha + mu)
#       if (exists("eta", fl)) {
#         eta = fl$eta[, , i]
#         gamma = fl$gamma[, w, i]
#         rate = rate + exp(gamma + eta)
#       }
#       pred_times = order(colMeans(rate), decreasing = TRUE)[1:(M + extra)]
#       cp[i, w] = sum(pred_times %in% true_times) / M
#     }
#   }
#   return(cp)
# }
# 
# full_rcp = cond.routine.cp(fl, dl, routine)
# full_rcp_insamp = mean(full_rcp[, 1:38], na.rm = TRUE)
# full_rcp_fore = mean(full_rcp[, 39:48], na.rm = TRUE)
# 
# 
# rbind(c(full_rcp_insamp, full_rcp_fore))
# 
# 
# 
# # Check extra = 1, 2, ...
# 
# x_range = 1:6
# 
# rcpx = c()
# for (x in x_range) {
#   full_rcp_x = cond.routine.cp(fl, dl, routine, extra = x)
#   rcpx = cbind(rcpx,
#                c(
#                  mean(full_rcp_x[, 1:38], na.rm = TRUE),
#                  mean(full_rcp_x[, 39:48], na.rm = TRUE)
#                ))
# }
# 
# rcpx = rbind(c(full_rcp_insamp,
#                full_rcp_fore),
#              t(rcpx))
# 
# matplot(c(0, x_range), cbind(rcpx), type = "l")
# 
# 
# 
# save(full_cp, full_map, full_rcp, rcpx, file="hourly_fit.RData")


load("hourly_fit.RData")

t.test(rowMeans(full_map[,39:48], na.rm=TRUE))
t.test(rowMeans(full_cp[,39:48], na.rm=TRUE))


map_routine_df = data.frame(
  id = 1:2000,
  Routineness = colSums(routine[39:48,]),
  MAP = rowMeans(full_map[,39:48]),
  CP = rowMeans(full_cp[,39:48])
)



map_plot = ggplot(na.omit(map_routine_df)) +
  geom_point(aes(x = Routineness, y = MAP)) + 
  geom_smooth(aes(x = Routineness, y = MAP)) +
  theme_bw() +
  theme(legend.position = "none", 
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) 

cp_plot = ggplot(na.omit(map_routine_df)) +
  geom_point(aes(x = Routineness, y = CP)) + 
  geom_smooth(aes(x = Routineness, y = CP)) +
  theme_bw() 

setwd("../tabs_and_figs/")
pdf("map_vs_routine.pdf", height=2.5, width=4)
print(map_plot)
dev.off()