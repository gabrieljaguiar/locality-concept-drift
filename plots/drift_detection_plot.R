library (ggplot2)
library (reshape2)

classifiers = c("LB", "HT")
dds = c("ADWIN", "DDM", "RDDM")
stream = c("rt_10", "rt_5","rf_10", "rf_5")


dd <- "ADWIN"
dd_data <- data.frame()
for (c in classifiers) {
    for (s in stream) {
      file_name = paste0(c, "_", dd, "_stable_", s, ".csv")
      path_file = paste0("../exp_output/", file_name)
      data <- read.csv(path_file)
      data <- data[, c("idx", "drifts_alerts")]
      data$stream <- s
      data$classifier <- c
      data$dd <- dd
      dd_data <- rbind(dd_data, data)
    }
}

dd_data$drifts_alerts[dd_data$drifts_alerts >=1] <- 1
dd_data$drifts_alerts <- as.factor(dd_data$drifts_alerts)
dd_data <- dd_data[dd_data$drifts_alerts == 1,]

ggplot(data = dd_data, aes(x=idx, y = stream)) + 
  geom_point(shape=15, color="red") + 
  scale_x_continuous(limits=c(0,100000)) +
  geom_vline(xintercept = c(30000, 60000), color="black", alpha=0.8, linetype=3) +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) + facet_grid(. ~ classifier)