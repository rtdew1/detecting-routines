lstm_results = as.matrix(read.csv("~/Dropbox/1_proj/via/me/r1/outputs/lstm/individual_out_predictions.csv", header = FALSE))
weekly_forecasts = matrix(0, nrow = 2000, ncol = 10)
for (week in 1:10) {
  hours = ((week-1)*168+1):(week*168)
  weekly_forecasts[,week] = rowSums(lstm_results[,hours])
}

nreqs_forecast = rowSums(lstm_results)

test0 <- read.csv("~/routines/data/test.csv", header = TRUE)
test0$id = factor(test0$id, levels = 1:2000)
test_reqs_by_week = as.matrix(table(test0$id, test0$week))
test_reqs = rowSums(test_reqs_by_week)

length(test_reqs)
length(nreqs_forecast)

mean(abs(test_reqs - nreqs_forecast))
t.test(abs(test_reqs - nreqs_forecast))


