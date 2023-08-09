library(ggplot2)
library(plotly)

data <- read.csv("../datasets/no_imbalance_switching_rf_5_sudden.csv")
colnames(data) <- c("x_1", "x_2", "class")

data$class <- as.factor(data$class)

data$id <- rep(seq(2500,100000, by=2500), each=2500)

g <- ggplot(data, aes(x=x_1, y=x_2, frame=id)) + geom_point(aes(color=class, shape=class)) 

ggplotly(g)

