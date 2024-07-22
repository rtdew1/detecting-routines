rm(list = ls())

library(tidyverse)
library(data.table)
library(stargazer)
library(ggplot2)
library(corrplot)




# Loading -----------------------------------------------------------------

load("analysis_inputs/dl_fl_workspace.RData")
load("analysis_inputs/y_nsessions.RData")
load("analysis_inputs/decomp.RData")

n_weeks = nrow(y_actual)


# Plots and tables --------------------------------------------------------

mu_plot = ggplot(data = data.frame(wday = inputs[,1], fhour = inputs[,2], lambda = colMeans(fl$mu)),
                 aes(x = -wday, y = fhour, fill = lambda)) + 
  geom_tile(color = "white", alpha=1) + 
  # scale_fill_gradient2(low = "#ffeda0", mid = "#feb24c", high = "#f03b20", name=expression(mu), midpoint = -9) + 
  scale_fill_gradient(low = "white", high = "steelblue", name=expression(mu)) + 
  scale_x_discrete(
    limits = -(1:7),
    labels = c("Sun","Mon","Tue","Wed","Thu","Fri","Sat")) +
  theme_bw() + 
  xlab("Day of the Week") + 
  ylab("Hour of the Day") + 
  ggtitle(paste("Population Rate (\u00b5)", sep = "")) +
  coord_flip()+ 
  theme(text = element_text(size = 14)) 

pdf("results/nonroutine_ride_heatmap.pdf", height=3, width=5.5)
mu_plot
dev.off()

decomp_plot = ggplot(data = data.frame(Random = decomp$random[dl$W,], Routine = decomp$routine[dl$W,])) +
  geom_point(aes(x = Random, y = Routine), color = "steelblue",fill = "steelblue", alpha=0.6, size=2) +
  xlab("Random") + 
  ylab("Routine") + 
  xlim(0,12) + ylim(0,12) + 
  ggtitle(paste("Decomposition: Week", dl$W))  + 
  theme_bw() + 
  theme(text = element_text(size = 14), plot.margin = unit(c(0.5,0.5,0.5,0.5),"cm"))

png(file = "results/routine_vs_random_rides.png", height=5, width=5, units = "in", res = 300)
decomp_plot
dev.off()