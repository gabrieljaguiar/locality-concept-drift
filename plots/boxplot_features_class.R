library(ggplot2)
library(reshape2)

df <- read.csv("../metrics/specs_metrics.csv")
df$n_classes <- as.factor(df$n_classes)
df$n_features <- as.factor(df$n_features)
df$case <- factor(df$case, levels=c("single_class_local", "single_class_global", "multi_class_local", "multi_class_global"))
df$f1 <- as.numeric(df$f1)/100

g <- ggplot(df, aes(x=case, y=f1)) + 
  geom_boxplot(aes(color=n_features)) + 
  theme_bw() +
  labs(color="# of features") + 
  theme(strip.text = element_text(size=12), legend.position="top") +
  xlab("Scenario") +
  ylab("F1") +
  facet_grid(drift_detector~.)


ggsave("n_features_boxplot.pdf", g, width = 6.95, height=5.15)
