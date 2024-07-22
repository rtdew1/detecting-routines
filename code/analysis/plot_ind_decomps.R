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

load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")
load("analysis_inputs/decomp_bounds.RData")


# Plots and tables --------------------------------------------------------


i = 1

pdf(file = paste("results/case_", i, "-decomp.pdf", sep=""),
    height = 4,
    width = 7)
plot_decomp_intervals(decomp_bounds, i, dl$W, paste("Case Study: Customer", i), ylab = "# Requests")
dev.off()

pdf(
  file = paste("results/case_", i, "-decomp_and_pars.pdf", sep=""),
  width = 9,
  height = 8
)
make_combined_plot_intervals(fl, decomp_bounds, i, dl$W)
dev.off()
