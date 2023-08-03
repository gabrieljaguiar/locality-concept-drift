library (ggplot2)
library (reshape2)

classifiers = c("LB", "HT", "NB")
dds = c("ADWIN", "DDM", "RDDM")
stream = c("rt_10", "rt_5","rf_10", "rf_5")

for (c in classifiers) {
  for (dd in dds) {
    for (s in stream) {
      file_name = paste0(c, "_", dd, "_stable_", s, ".csv")
      path_file = paste0("../exp_output/", file_name)
    }
  }
}

data <- read.csv("../exp_output/LB_ADWIN_stable_rt_5.csv")

data <- data[, 2:(ncol(data) - 1)]

data_melt <- melt(data, id.vars = c("idx"))
data_melt$local <- "local"
data_melt[data_melt$variable=="G.Mean",]$local <- "global"

ggplot(data_melt, aes(x=idx, y=value)) + 
  geom_line(aes(color=variable)) + 
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  geom_vline(xintercept = c(30000, 60000), color="black", alpha=0.8, linetype=3) +
  facet_grid(.~local)
