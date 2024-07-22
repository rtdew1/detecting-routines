
# WARNING: This script may take a long time to run. 

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
load("analysis_inputs/decomp_bounds.RData")

train_end = dl$W
test_end = dl$W + dl$W_star


## Conditional precision (CP)

cond.top.cp = function(fl, dl) {
  cp = matrix(NA, dl$N, dl$W + dl$W_star)
  for (i in 1:dl$N) {
    for (w in 1:(dl$W + dl$W_star)) {
      true_times = c(unique(dl$dayhour[dl$week == w & dl$id == i]),
                     unique(dl$dayhour_star[dl$week_star == w &
                                              dl$id_star == i]))
      M = length(true_times)

      if (M == 0) {
        next
      }

      alpha = fl$alpha[, w, i]
      mu = fl$mu

      rate = exp(alpha + mu)
      if (exists("eta", fl)) {
        eta = fl$eta[, , i]
        gamma = fl$gamma[, w, i]
        rate = rate + exp(gamma + eta)
      }
      pred_times = order(colMeans(rate), decreasing = TRUE)[1:M]
      cp[i, w] = sum(pred_times %in% true_times) / M
    }
  }
  return(cp)
}

full_cp = cond.top.cp(fl, dl)

full_cp_insamp = mean(full_cp[, 1:train_end], na.rm = TRUE)
full_cp_fore = mean(full_cp[, (train_end+1):test_end], na.rm = TRUE)

c(full_cp_insamp, full_cp_fore)



# Mean average precision (MAP)

full_map = matrix(NA, dl$N, dl$W + dl$W_star)
for (i in 1:dl$N) {
  wks_i = c(dl$week[dl$id == i], dl$week_star[dl$id_star == i])
  j_i = c(dl$dayhour[dl$id == i], dl$dayhour_star[dl$id_star == i])
  map_i = c()
  for (w in unique(wks_i)) {
    wk_rank = order(colMeans(exp(fl$alpha[, w, i] + fl$mu) + exp(fl$gamma[, w, i] + fl$eta[, , i])), decreasing = TRUE)
    relevant_ranks = sort(which(wk_rank %in% j_i[wks_i == w]))
    full_map[i, w] = mean(1:length(relevant_ranks) / relevant_ranks)
  }
}

full_map_insamp = mean(full_map[, 1:train_end], na.rm = TRUE)
full_map_fore = mean(full_map[, (train_end+1):test_end], na.rm = TRUE)

rbind(c(full_map_insamp, full_map_fore))

# Save intermediate results
save(full_cp, full_map, file="results/hourly_fit.RData")


# If previous results were saved, you can load them here instead of re-running
# load("hourly_fit.RData")

map_routine_df = data.frame(
  id = 1:dl$N,
  Routineness = colSums(routine[(train_end+1):test_end,]),
  MAP = rowMeans(full_map[,(train_end+1):test_end]),
  CP = rowMeans(full_cp[,(train_end+1):test_end])
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

pdf("results/map_vs_routine.pdf", height=2.5, width=4)
print(map_plot)
dev.off()