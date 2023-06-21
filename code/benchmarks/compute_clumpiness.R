library(data.table)
train = fread("~/routines/data/train.csv")
train$trans_times = 168 * (train$week - 1) + train$dayhour
setorder(train, cols = "id", "trans_times")

compute_clumpiness = function(data) {
  trans_times = data$trans_times
  first_trans_week = min(data$week - data$iweek)
  first_kept_week = max(first_trans_week + 3, 1)
  trans_time_0 = 168 * (first_kept_week - 1)
  iets = c(trans_times[1] - trans_time_0, 
           diff(trans_times), 
           38*168 + 1 - tail(trans_times, 1))
  x = iets / (38*168)
  1 + sum(log(x) * x) / log(length(x))
}

clumpinesses = do.call(c, by(train, train$id, compute_clumpiness, simplify = FALSE))
export = data.frame(id = 1:2000, C = clumpinesses)
write.csv(export, file = "~/Dropbox/1_proj/via/me/r1/analysis_inputs/clumpiness_scores.csv")
