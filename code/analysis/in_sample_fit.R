rm(list = ls())

library(tidyverse)
library(data.table)
library(ggplot2)


# Loading -----------------------------------------------------------------

load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")

n_weeks = nrow(y_actual)



# Plots and tables --------------------------------------------------------


fit_df = data.frame()
for(i in 1:dl$N){
  for(w in 1:dl$W){
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

png("results/in_sample_fit.png", height = 4, width = 5, units = "in", res = 300)
insamp_fit_plot
dev.off()
