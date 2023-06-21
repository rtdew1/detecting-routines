library(ggplot2)
library(showtext)
library(gridExtra)
showtext_auto()

lightred = rgb(1,0.8,0.8)

plot_decomp = function(decomp, i, max_w, title, ylab = "Intensity") {
  decomp_plot_df = pivot_longer(
    data.frame(
      Week = 1:max_w,
      Random = decomp$random[1:max_w, i],
      Routine = decomp$routine[1:max_w, i]
    ),
    c("Random", "Routine")
  )
  
  decomp_plot = ggplot(decomp_plot_df,
                       aes(
                         x = Week,
                         y = value,
                         color = name,
                         linetype = name
                       )) +
    scale_color_manual(values = c("black", "red3")) +
    geom_line() +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      title = element_text(size = 14),
      legend.text = element_text(size = 12)
    ) +
    ylab(ylab) +
    ggtitle(title)
  
  return(decomp_plot)
}

plot_decomp_intervals = function(decomp_w_bounds, i, max_w, title, ylab = "Intensity", alpha=0.1, start_w=1) {
  decomp = decomp_w_bounds
  
  random_plot_df = data.frame(
    Week = start_w:max_w,
    Median = decomp$random[1:max_w, i],
    Lower = decomp$random_lower[1:max_w, i],
    Upper = decomp$random_upper[1:max_w, i],
    name = "Random"
  )
  
  routine_plot_df = data.frame(
    Week = start_w:max_w,
    Median = decomp$routine[1:max_w, i],
    Lower = decomp$routine_lower[1:max_w, i],
    Upper = decomp$routine_upper[1:max_w, i],
    name = "Routine"
  )
  
  decomp_plot_df = rbind(random_plot_df, routine_plot_df)
  
  
  decomp_plot = ggplot(decomp_plot_df,
         aes(
           x = Week,
           y = Median,
           color = name,
           linetype = name
         )) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper, fill = name, alpha = alpha), linetype = 0) +
    scale_color_manual(values = c("black", "red3")) +
    scale_fill_manual(values = c("grey80", lightred)) +
    geom_line() +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      title = element_text(size = 14),
      legend.text = element_text(size = 12)
    ) +
    guides(alpha = guide_none()) +
    ylab(ylab) +
    ggtitle(title)
  
  return(decomp_plot)
}




plot_dayhour_heatmap = function(intensity, title, fill_name) {
  inputs = cbind(rep(1:7, each=24), rep(0:23, 7))
  
  dh_plot = ggplot(
    data = data.frame(
      wday = inputs[, 1],
      fhour = inputs[, 2],
      lambda = intensity
    ),
    aes(x = -wday, y = fhour, fill = lambda)
  ) +
    geom_tile(color = "white", alpha = 0.7) +
    scale_fill_gradient(low = "white",
                        high = "steelblue",
                        name = fill_name) +
    scale_x_discrete(
      limits = -(1:7),
      labels = c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
    ) +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      title = element_text(size = 14),
      legend.text = element_text(size = 12)
    ) +
    xlab("Day of the Week") +
    ylab("Hour of the Day") +
    ggtitle(title) +
    coord_flip()
  
  return(dh_plot)
}

plot_scale = function(scale, title, ylab) {
  W = length(scale)
  qplot(1:W, scale, geom = "line") +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      title = element_text(size = 14),
      legend.text = element_text(size = 12)
    ) +
    xlab("Week") +
    ylab(ylab) +
    ggtitle(title)
}


plot_scale_intervals = function(post, title, ylab, alpha=0.1) {
  
  max_w = dim(post)[2]

  plot_df = data.frame(
    scale = apply(post, 2, median),
    lower = apply(post, 2, quantile, probs=0.025),
    upper = apply(post, 2, quantile, probs=0.975)
  )
  
  ggplot(plot_df, aes(x = 1:max_w, y = scale)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "black", alpha = alpha) +
    geom_line() +
    theme_bw() +
    labs(
      x = "Week",
      y = ylab,
      title = title
    ) +
    theme(
      legend.title = element_blank(),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      title = element_text(size = 14),
      legend.text = element_text(size = 12)
    )
}

compute_expected_reqs = function(fl, i, w) {
  apply(
    exp(fl$gamma[, w, i] + fl$eta[, , i]) +
      exp(fl$alpha[, w, i] + fl$mu),
    2,
    median
  )
}

combine_plots = function(alpha_plot, gamma_plot, eta_plot, reqs_plot) {
  arrmat = rbind(c(1, 2), c(3, 3), c(4, 4))
  return(gridExtra::grid.arrange(alpha_plot, gamma_plot, eta_plot, reqs_plot, layout_matrix = arrmat))
}

combine_plots_w_decomp = function(decomp_plot,
                                  alpha_plot,
                                  gamma_plot,
                                  eta_plot,
                                  reqs_plot,
                                  small_plot_reps = 2,
                                  large_plot_reps = 4) {
  
  arrmat = c(rep(1, small_plot_reps), rep(NA, 4))
  for (j in 1:20) {
    arrmat = rbind(arrmat,
                   c(
                     rep(1, small_plot_reps),
                     rep(2, small_plot_reps),
                     rep(3, small_plot_reps)
                   ))
  }
  for (j in 1:20) {
    arrmat = rbind(arrmat, c(NA, rep(4, large_plot_reps), NA))
  }
  for (j in 1:20) {
    arrmat = rbind(arrmat, c(NA, rep(5, large_plot_reps), NA))
  }
 
  gridExtra::grid.arrange(decomp_plot, alpha_plot, gamma_plot, eta_plot, reqs_plot, layout_matrix = arrmat)
}


make_combined_plot = function(fl, decomp, i, max_w, start_w = 1) {
  decomp_plot = plot_decomp(decomp = decomp, i = i, start_w = start_w, max_w = max_w, title = "Decomposition") +
    theme(
      legend.title = element_blank(),
      legend.position = 'top',
      legend.margin=margin(0,0,0,0),
      legend.box.margin=margin(-8,-10,-10,-10)
    )
  
  est_eta = apply(fl$eta[, , i], 2, median)
  eta_plot = plot_dayhour_heatmap(est_eta, title = "Routine Rate (\u03B7)", fill_name = expression(eta))
  
  e_reqs = compute_expected_reqs(fl, i, w_max)
  reqs_plot = plot_dayhour_heatmap(e_reqs, title = "Expected Requests", fill_name = "E(# Reqs)")
  
  gamma_plot = plot_scale(colMeans(fl$gamma[, 1:w_max, i]), title = "Routine Scale (\u03B3)", ylab = "\u03B3")
  alpha_plot = plot_scale(colMeans(fl$alpha[, 1:w_max, i]), title = "Random Scale (\u03B1)", ylab = "\u03B1")
  
  return(combine_plots_w_decomp(decomp_plot, alpha_plot, gamma_plot, eta_plot, reqs_plot))
}


make_combined_plot_intervals = function(fl, decomp_bounds, i, w_max) {
  decomp_plot = plot_decomp_intervals(decomp_bounds, i = i, max_w = w_max, title = "Decomposition") +
    theme(
      legend.title = element_blank(),
      legend.position = 'top',
      legend.margin=margin(0,0,0,0),
      legend.box.margin=margin(-8,-10,-10,-10)
    )
  
  est_eta = apply(fl$eta[, , i], 2, median)
  eta_plot = plot_dayhour_heatmap(est_eta, title = "Routine Rate (\u03B7)", fill_name = expression(eta))
  
  e_reqs = compute_expected_reqs(fl, i, w_max)
  reqs_plot = plot_dayhour_heatmap(e_reqs, title = "Expected Requests", fill_name = "E(# Reqs)")
  
  gamma_plot = plot_scale_intervals(fl$gamma[, 1:w_max, i], title = "Routine Scale (\u03B3)", ylab = "\u03B3")
  alpha_plot = plot_scale_intervals(fl$alpha[, 1:w_max, i], title = "Random Scale (\u03B1)", ylab = "\u03B1")
  
  return(combine_plots_w_decomp(decomp_plot, alpha_plot, gamma_plot, eta_plot, reqs_plot))
}