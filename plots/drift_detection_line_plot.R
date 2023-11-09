library (ggplot2)
library (reshape2)



#dd <- read.csv("../output/HT_ADWIN_single_class_local_emerging_branch_ds_1_c_3_ca_1_f_2_1_1.csv")
#dd_alerts <- read.csv("../output/drift_alerts_HT_ADWIN_single_class_local_emerging_branch_ds_1_c_3_ca_1_f_2_1_1.csv")

#dd <- read.csv("../output/HT_ADWIN_single_class_global_class_emerging_rbf_ds_5000_c_3_ca_1_f_2_1_1.csv")
#dd_alerts <- read.csv("../output/drift_alerts_HT_ADWIN_single_class_global_class_emerging_rbf_ds_5000_c_3_ca_1_f_2_1_1.csv")

dd <- read.csv("../output/HT_ADWIN_multi_class_local_swap_cluster_ds_1_c_5_ca_5_f_2_1_1.csv")
dd_alerts <- read.csv("../output/drift_alerts_HT_ADWIN_multi_class_local_swap_cluster_ds_1_c_5_ca_5_f_2_1_1.csv")

#dd <- read.csv("../output/HT_ADWIN_multi_class_global_swap_leaves_ds_5000_c_5_ca_5_f_2_1_1.csv")
#dd_alerts <- read.csv("../output/drift_alerts_HT_ADWIN_multi_class_global_swap_leaves_ds_5000_c_5_ca_5_f_2_1_1.csv")

g <- ggplot(data = dd, aes(x=idx, y = accuracy)) + 
  scale_x_continuous(limits=c(0,100000)) +
  scale_y_continuous(limits=c(0.25,0.65),expand = c(0, 0)) +
  #geom_rect(aes(xmin = 47500,xmax = 52500,  ymin=0.25, ymax=Inf),fill="gray", alpha=0.015) +
  geom_line(color="#1f77b4") + 

  geom_vline(xintercept = c(50000), color="black", alpha=0.8, linetype=3, linewidth=1) +
  geom_vline(data=dd_alerts, aes(xintercept=idx), color="red", linetype=3, alpha=0.8) +
  theme_bw()+ ylab("Accuracy") + xlab("Instances")
g
ggsave(filename="multi_class_local_line_drift_plot.pdf", width=8.51, height=2.5)