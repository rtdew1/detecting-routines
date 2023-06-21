rm(list = ls())

library(tidyverse)
library(data.table)
library(stargazer)
library(ggplot2)


# Loading -----------------------------------------------------------------

weeks_lower = 52 # (greater than)
weeks_upper = 90 # (less than or equals to)
n_weeks = weeks_upper - weeks_lower

load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")

setwd("results/")


# Plots and tables --------------------------------------------------------


fit_df = data.frame()
for(i in 1:dl$N){
  for(w in W_min[i]:dl$W){
    iw_df = data.frame(Person = i, Week = w, Expected = decomp$random[w,i] + decomp$routine[w,i], Actual = y_actual[w,i])
    fit_df = rbind(fit_df, iw_df)
  }
}

insamp_fit_plot = ggplot(data = fit_df) +
  geom_abline(slope = 1, intercept = 0, col = "gray80") + 
  geom_point(aes(x = Expected, y = Actual, alpha = Week), color = "steelblue", fill = "steelblue") +
  scale_alpha_continuous(range = c(0.2,0.6)) +
  theme_bw() +
  theme(legend.position = "none") + theme(text = element_text(size = 14)) 

png("in_sample_fit.png", height = 4, width = 5, units = "in", res = 300)
insamp_fit_plot
dev.off()

cor.test(fit_df$Expected, fit_df$Actual)
