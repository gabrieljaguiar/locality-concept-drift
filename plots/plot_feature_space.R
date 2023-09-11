library(ggplot2)
library(plotly)

folder = "../datasets/"

files <- list.files(folder, pattern = "*.csv$")

files <- c("test_stream_5_features.csv")
for (file in files){
  print (file)
  data <- read.csv(paste0(folder,file))
  colnames(data) <- c("x_1", "x_2","x_3", "x_4", "x_5", "class")
  
  data$class <- as.factor(data$class)
  
  data$id <- seq(1,100000, by=1)
  data$timing <- "during_drift"
  data[data$id < 40000,]$timing <- "before_drift"
  data[data$id > 60000,]$timing <- "after_drift"
  
  data$timing <- factor(data$timing, levels=c("before_drift", "during_drift", "after_drift"))
  
  g <- ggplot(data, aes(x=x_4, y=x_5)) + 
    geom_point(aes(color=class, shape=class)) +
    scale_shape_manual(values=c(seq(1,length(unique(data$class))))) +
    theme_bw() +
    facet_grid(. ~ timing)
  #ggsave(filename = paste0(folder,file, ".pdf"), plot=g, width = 12, height = 4)
  
}
