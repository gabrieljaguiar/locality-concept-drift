library (ggplot2)
library(zoo)




#ht <- read.csv("../output/HT_ADWIN_single_class_local_prune_growth_new_branch_ds_5000_c_5_ca_1_f_10_1_1.csv")
#aht <- read.csv("../output/AHT_ADWIN_single_class_local_prune_growth_new_branch_ds_5000_c_5_ca_1_f_10_1_1.csv")
#htdd <- read.csv("../output/HTDD_ADWIN_single_class_local_prune_growth_new_branch_ds_5000_c_5_ca_1_f_10_1_1.csv")

#ht <- read.csv("../output/HT_ADWIN_single_class_local_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")
#aht <- read.csv("../output/AHT_ADWIN_single_class_local_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")
#htdd <- read.csv("../output/HTDD_ADWIN_single_class_local_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")

ht <- read.csv("../output/HT_ADWIN_single_class_global_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")
aht <- read.csv("../output/AHT_ADWIN_single_class_global_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")
htdd <- read.csv("../output/HTDD_ADWIN_single_class_global_prune_growth_new_branch_ds_1_c_5_ca_1_f_10_1_1.csv")

#ht <- read.csv("../output/HT_ADWIN_multi_class_local_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")
#aht <- read.csv("../output/AHT_ADWIN_multi_class_local_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")
#htdd <- read.csv("../output/HTDD_ADWIN_multi_class_local_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")

#ht <- read.csv("../output/HT_ADWIN_multi_class_global_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")
#aht <- read.csv("../output/AHT_ADWIN_multi_class_global_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")
#htdd <- read.csv("../output/HTDD_ADWIN_multi_class_global_prune_growth_new_branch_ds_1_c_5_ca_3_f_10_1_1.csv")

ht$accuracy <- rollmean(ht$accuracy, 5, na.pad = TRUE, align = "left")
aht$accuracy <- rollmean(aht$accuracy, 5, na.pad = TRUE, align = "left")
htdd$accuracy <- rollmean(htdd$accuracy, 5, na.pad = TRUE, align = "left")

ht$classifier <- "HT"
aht$classifier <- "AHT"
htdd$classifier <- "HT-DW"

df <- rbind(ht, aht, htdd)
g <- ggplot(df, aes(x=idx, y=accuracy)) +
  geom_line(aes(color=classifier), linewidth=1) +
  scale_y_continuous(limits=c(0.35,0.575), expand = c(0.01,0.01)) +
  geom_vline(xintercept = c(50000), color="black", alpha=0.8, linetype=3, linewidth=1) +
  theme_bw() + ylab("Accuracy") + xlab("Instances") +
  labs(title="Single-Class Global", color="Classifier") +
  theme(plot.title = element_text(family = "Helvetica", face = "plain", 
                                  size = 22, hjust=0.5),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size=14),
        axis.ticks.x = element_blank(),
        axis.title.x = element_text(size=16),
        axis.title.y = element_text(size=16),
        legend.position = "none",
        legend.text = element_text(size=16),
        legend.title = element_text(size=18),
        panel.border = element_rect(colour = "black", fill=NA, size=2))
g

#library(grid)
#library(gridExtra)
#library(cowplot)

#legend <- get_legend(g)                     

# Create new plot window 
#grid.newpage()                               

# Draw Only legend  
#grid.draw(legend) 

ggsave("accuracy_single_class_global_line.pdf", g, width = 4.5, height=4.5)
