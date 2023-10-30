library(dplyr)

df_1 <- read.csv("../metrics/single_class_local_predictive_metrics.csv")
df_2 <- read.csv("../metrics/single_class_global_predictive_metrics.csv")
df_3 <- read.csv("../metrics/multi_class_local_predictive_metrics.csv")
df_4 <- read.csv("../metrics/multi_class_global_predictive_metrics.csv")

df <- rbind(df_1, df_2, df_3, df_4)

df$scenario <- factor(df$scenario, levels=c("single_class_local", "single_class_global", "multi_class_local", "multi_class_global"))


filtered <- df %>%
  group_by(classifier, scenario) %>%
  summarize(mean(accuracy), sd(accuracy))

write.csv(filtered, file="result.csv")
