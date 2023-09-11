library (ggplot2)
library (reshape2)

classifiers = c("LB", "HT")
dds = c("ADWIN", "DDM", "RDDM", "DDM_OCI", "MCADWIN")
stream_rf =  c("no_imbalance_switching_rf_10_sudden_fix_majority",
               "no_imbalance_switching_rf_10_gradual_fix_majority",
               "imbalance_switching_rf_10_sudden_fix_majority", 
               "imbalance_switching_rf_10_gradual_fix_majority")

stream_rt = c("no_imbalance_switching_rt_10_sudden_fix_majority",
              "no_imbalance_switching_rt_10_gradual_fix_majority",
              "imbalance_switching_rt_10_sudden_fix_majority", 
              "imbalance_switching_rt_10_gradual_fix_majority")

stream = stream_rt

dd_data <- data.frame()
for (dd in dds){
  for (c in classifiers) {
      for (s in stream) {
        file_name = paste0(c, "_", dd, "_", s, ".csv")
        path_file = paste0("../exp_output/", file_name)
        data <- read.csv(path_file)
        if (dd == "MCADWIN"){
          data <- data[, c("idx", "drifts_alerts", "local_alerts")]
          data$drifts_alerts <- data$local_alerts
          data <- data[, c("idx", "drifts_alerts")]
        }else{
          data <- data[, c("idx", "drifts_alerts")]
        }
        data$stream <- gsub("*.switching", "", gsub("_", ".", gsub("_fix_majority", "", s)))
        data$classifier <- c
        data$dd <- dd
        dd_data <- rbind(dd_data, data)
      }
  }
}

#dd_data$drifts_alerts <- dd_data$drifts_alerts + dd_data$local_drifts
dd_data$drifts_alerts[dd_data$drifts_alerts >=1] <- 1
dd_data$drifts_alerts <- as.factor(dd_data$drifts_alerts)
dd_data <- dd_data[dd_data$drifts_alerts == 1,]

g <- ggplot(data = dd_data, aes(x=idx, y = stream)) + 
  geom_point(shape=15, color="red") + 
  scale_x_continuous(limits=c(0,100000)) +
  geom_vline(xintercept = c(20000, 40000, 60000, 80000), color="black", alpha=0.8, linetype=3) +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) + facet_grid(dd ~ classifier )
g
#ggsave("drift_detection_rt_5.pdf", g, width = 11, height = 10)