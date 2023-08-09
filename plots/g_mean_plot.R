library (ggplot2)
library (reshape2)

classifiers = c("LB", "HT")
dds = c("ADWIN")
stream_rf =  c("no_imbalance_switching_rf_10_sudden_fix_majority",
    "no_imbalance_switching_rf_10_gradual_fix_majority",
    "imbalance_switching_rf_10_sudden_fix_majority", 
    "imbalance_switching_rf_10_gradual_fix_majority")

stream_rt = c("no_imbalance_switching_rt_10_sudden_fix_majority",
    "no_imbalance_switching_rt_10_gradual_fix_majority",
    "imbalance_switching_rt_10_sudden_fix_majority", 
    "imbalance_switching_rt_10_gradual_fix_majority")

stream = stream_rf

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
      
      data_melt[grepl("_prop_", data_melt$variable),]$local <- "ratio"
      data_melt$variable <- gsub("_prop_", "_", data_melt$variable)
      data_melt[data_melt$variable=="accuracy",]$local <- "global"
      data_melt[data_melt$variable=="gmean",]$local <- "global"
      data_melt$stream <- gsub("*.switching", "", gsub("_", ".", gsub("_fix_majority", "", s)))
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
    facet_grid(stream~local, scales="free_y") +
    ggtitle(c)
  
  ggsave(paste0(c,"_fix_majority_rf_10.pdf"), g, width = 18, height = 10)
}
g

