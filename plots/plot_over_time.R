library (ggplot2)
library(gapminder)
library(gganimate)
library (plotly)

data <- read.csv("../inter_swapping_cluster.csv")
colnames(data) <- c("x_1", "x_2", "class")

data$class <- as.factor(data$class)

data$id <- rep(seq(1, 100000/1000, by = 1), each=1000)
data$id <- as.integer(data$id)
#data <- data[data$id > 25, ]
g<- ggplot(data, aes(x=x_1, y=x_2, frame = id)) +
  geom_point(aes(color=class, shape=class)) +
  transition_time(id) +
  labs(title = 'Instances*1000: {frame_time}') + 
  ease_aes('linear')
g
#ggplotly(g)