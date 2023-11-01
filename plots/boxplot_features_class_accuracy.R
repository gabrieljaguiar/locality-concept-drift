library(ggplot2)
library(reshape2)

df_1 <- read.csv("../metrics/single_class_local_predictive_metrics.csv")
df_2 <- read.csv("../metrics/single_class_global_predictive_metrics.csv")
df_3 <- read.csv("../metrics/multi_class_local_predictive_metrics.csv")
df_4 <- read.csv("../metrics/multi_class_global_predictive_metrics.csv")

df <- rbind(df_1, df_2, df_3, df_4)
df$n_classes <- as.factor(df$n_classes)
df$n_features <- as.factor(df$n_features)
df$scenario <- factor(df$scenario, levels=c("single_class_local", "single_class_global", "multi_class_local", "multi_class_global"))
#df$f1 <- as.numeric(df$f1)/100

g <- ggplot(df, aes(x=scenario, y=accuracy)) + 
  geom_boxplot(aes(color=n_classes)) + 
  theme_bw() +
  labs(color="# of features") + 
  theme(legend.position="top") +
  xlab("Scenario") +
  ylab("Accuracy")


colors <-c( 
"#1f77b4",
"#ff7f0e",
"#2ca02c",
"#d62728",
"#9467bd",
"#8c564b",
"#e377c2",
"#7f7f7f",
"#bcbd22",
"#17becf",
"#aec7e8"
)

scenario_to_be_plotted = "multi_class_local"
g <- ggplot(df[df$scenario == scenario_to_be_plotted,], aes(x=difficulty, y=accuracy)) + 
  geom_boxplot(aes(color=difficulty)) + 
  scale_y_continuous(limits=c(0.25,0.99)) +
  scale_color_manual(values=colors) + 
  theme_bw() +
  labs(color="") + 
  #theme(legend.position="top", axis.text.x = element_text(angle=30, vjust = 1, size=10, hjust = 1)) +
  theme(legend.position="top", axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  xlab("") +
  ylab("Accuracy") 
#facet_grid(scenario~.)

ggsave("accuracy_multi_class_local_boxplot.pdf", g, width = 7.80, height=2.55)

