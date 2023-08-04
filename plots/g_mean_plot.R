library (ggplot2)
library (reshape2)

classifiers = c("LB", "HT", "NB")
dds = c("ADWIN")
stream =  c("no_imbalance_switching_rf_5_sudden",
    "no_imbalance_switching_rf_5_gradual",
    "imbalance_switching_rf_5_sudden", 
    "imbalance_switching_rf_5_gradual")

for (c in classifiers) {
  data_all_melt <- data.frame()
  for (dd in dds) {
    for (s in stream) {
      file_name = paste0(c, "_", dd, "_", s, ".csv")
      path_file = paste0("../exp_output/", file_name)
      data <- read.csv(path_file)
      
      data <- data[, 2:(ncol(data) - 1)]
      
      data_melt <- melt(data, id.vars = c("idx"))
      data_melt$local <- "local"
      data_melt[data_melt$variable=="accuracy",]$local <- "global"
      data_melt[data_melt$variable=="gmean",]$local <- "global"
      data_melt$stream <- s
      data_all_melt <- rbind(data_all_melt, data_melt)
      

    }
  }
  g <- ggplot(data_all_melt, aes(x=idx, y=value)) + 
    geom_line(aes(color=variable)) + 
    theme_bw() +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      strip.text.y = element_text(size=7)
    ) +
    geom_vline(xintercept = c(20000, 40000, 60000, 80000), color="black", alpha=0.8, linetype=3) +
    facet_grid(stream~local) +
    ggtitle(c)
  
  ggsave(paste0(c,".pdf"), g, width = 11)
}
g

