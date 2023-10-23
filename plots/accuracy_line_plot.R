library (ggplot2)
library(zoo)

ht <- read.csv("../output/HT_ADWIN_single_class_local_emerging_branch_ds_1_c_5_ca_1_f_5_1_1.csv")
ht <- read.csv("../output/HT_ADWIN_single_class_local_emerging_branch_ds_1_c_5_ca_1_f_5_1_1.csv")
aht <- read.csv("../output/AHT_ADWIN_single_class_local_emerging_branch_ds_1_c_5_ca_1_f_5_1_1.csv")
ht$accuracy <- rollmean(ht$accuracy, 2, na.pad = TRUE, align = "left")
aht$accuracy <- rollmean(aht$accuracy, 2, na.pad = TRUE, align = "left")

ht$classifier <- "HT"
aht$classifier <- "AHT"

df <- rbind(ht, aht)
g <- ggplot(df, aes(x=idx, y=accuracy)) +
  geom_line(aes(color=classifier))
g
