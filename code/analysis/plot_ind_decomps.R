rm(list = ls())

library(tidyverse)
library(data.table)
library(stargazer)
library(ggplot2)
library(xtable)
library(gridExtra)
library(showtext)

showtext_auto()

source("plotting_functions.R")

# Loading -----------------------------------------------------------------

weeks_lower = 52 # (greater than)
weeks_upper = 90 # (less than or equals to)
n_weeks = weeks_upper - weeks_lower


load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")
load("analysis_inputs/decomp_bounds.RData")

setwd("results/")


# Plots and tables --------------------------------------------------------


# Comparing routine and random

i = 110

pdf(file = "real_case-110-decomp_intervals.pdf",
    height = 4,
    width = 7)
plot_decomp_intervals(decomp_bounds, i, dl$W, "Case Study: Customer 110", ylab = "# Requests")
dev.off()

pdf(
  file = paste("real_case-110-combined_intervals.pdf", sep = ""),
  width = 9,
  height = 8
)
make_combined_plot_intervals(fl, decomp_bounds, i, dl$W)
dev.off()



i = 1520

pdf(file = "real_case-1520-decomp_intervals.pdf",
    height = 4,
    width = 7)
plot_decomp_intervals(decomp_bounds, i, dl$W, "Case Study: Customer 1520", ylab = "# Requests")
dev.off()

pdf(
  file = paste("real_case-1520-combined_intervals.pdf", sep = ""),
  width = 9,
  height = 8
)
make_combined_plot_intervals(fl, decomp_bounds, i, dl$W)
dev.off()


# FIND NEW CASE STUDY THAT IS ACTIVE DURING WHOLE PERIOD

num_wks = do.call(c, by(dl$week, dl$id, function(x) length(unique(x)), simplify=FALSE))
which(colMeans(decomp$random[1:38,]) > 1 & num_wks > 20)
start_w = do.call(c, by(dl$week, dl$id, min, simplify=FALSE))

for(i in which(colMeans(decomp$random[1:38,]) > 1 & num_wks > 20)) {
    pdf(
    file = paste("new_case_candidates/real_case-", i, "-combined_intervals.pdf", sep = ""),
    width = 9,
    height = 8
  )
  make_combined_plot_intervals(fl, decomp_bounds, i, dl$W)
  dev.off()
}
