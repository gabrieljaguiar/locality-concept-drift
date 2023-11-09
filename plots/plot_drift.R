library (ggplot2)
library (reshape2)

#single_class_local <- read.csv(file = "../datasets/datasets/single_class_local_emerging_branch_ds_1_c_3_ca_1_f_2_1_1.csv")
#single_class_local <- read.csv(file = "../datasets/datasets/single_class_global_class_emerging_rbf_ds_5000_c_3_ca_1_f_2_1_1.csv")
#single_class_local <- read.csv(file = "../datasets/datasets/multi_class_local_swap_cluster_ds_1_c_5_ca_5_f_2_1_1.csv")
single_class_local <- read.csv(file = "../datasets/datasets/multi_class_global_swap_leaves_ds_5000_c_5_ca_5_f_2_1_1.csv")


require(MASS)
require(class)
require(dplyr)

test <- expand.grid(x=seq(min(single_class_local[,1] - 0.15), max(single_class_local[,1]+0.15),
                          by=0.1),
                    y=seq(min(single_class_local[,2] - 0.15), max(single_class_local[,2]+0.15), 
                          by=0.1))

before <- c(20000:22500)
during <- c(49500:51000)
after <- c(92501:95000)

single_class_local$id <- c(1:100000)


single_class_local[before,"drift_point"] <- "Before"
single_class_local[during,"drift_point"] <- "During"
single_class_local[after, "drift_point"] <- "After"


classif_before <- knn(single_class_local[before,c(1,2)], test, single_class_local[before,3], k = 3, prob=TRUE)
prob_before <- attr(classif_before, "prob")

dataf_before <- bind_rows(mutate(test,
                          prob=prob_before,
                          class=0,
                          prob_class=ifelse(classif_before==class,
                                          1, 0)),
                   mutate(test,
                          prob=prob_before,
                          class=1,
                          prob_class=ifelse(classif_before==class,
                                          1, 0)),
                   mutate(test,
                          prob=prob_before,
                          class=2,
                          prob_class=ifelse(classif_before==class,
                                          1, 0)))

dataf_before$drift_point = "Before"

classif_during <- knn(single_class_local[during,c(1,2)], test, single_class_local[during,3], k = 3, prob=TRUE)
prob_during <- attr(classif_during, "prob")

dataf_during <- bind_rows(mutate(test,
                          prob=prob_during,
                          class=0,
                          prob_class=ifelse(classif_during==class,
                                          1, 0)),
                   mutate(test,
                          prob=prob_during,
                          class=1,
                          prob_class=ifelse(classif_during==class,
                                          1, 0)),
                   mutate(test,
                          prob=prob_during,
                          class=2,
                          prob_class=ifelse(classif_during==class,
                                          1, 0)))

dataf_during$drift_point = "During"

classif_after <- knn(single_class_local[after,c(1,2)], test, single_class_local[after,3], k = 3, prob=TRUE)
prob_after <- attr(classif_after, "prob")

dataf_after <- bind_rows(mutate(test,
                                 prob=prob_after,
                                 class=0,
                                 prob_class=ifelse(classif_after==class,
                                                 1, 0)),
                          mutate(test,
                                 prob=prob_after,
                                 class=1,
                                 prob_class=ifelse(classif_after==class,
                                                 1, 0)),
                          mutate(test,
                                 prob=prob_after,
                                 class=2,
                                 prob_class=ifelse(classif_after==class,
                                                 1, 0)))

dataf_after$drift_point = "After"

dataf <- rbind(dataf_before, dataf_during, dataf_after)
df <- single_class_local[!is.na(single_class_local$drift_point),]
colnames(df) <- c("x","y", "class", "id", "drift_point")
df$class <- as.factor(df$class)
df$drift_point <- factor(df$drift_point, levels=c("Before", "During", "After"))
dataf$drift_point <- factor(dataf$drift_point,  levels=c("Before", "During", "After"))

test_before <- test
test_before$class <- classif_before
test_before$drift_point <- "Before"

test_during <- test
test_during$class <- classif_during
test_during$drift_point <- "During"

test_after <- test
test_after$class <- classif_after
test_after$drift_point <- "After"

test <- rbind(test_before, test_after, test_during)
test$drift_point <- factor(test$drift_point, levels=c("Before", "During", "After"))
test$class <- factor(test$class)

colors <-c( 
  "#1f77b4",
  "#d62728",
  "#2ca02c",
  "#ff7f0e",
  "#9467bd"
)

#df <- df[df$drift_point!="During", ]
#df$drift_point <- factor(df$drift_point, levels=c("Before",  "After"))
#test <- test[test$drift_point!="During", ]
#test$drift_point <- factor(test$drift_point, levels=c("Before", "After"))


g <- ggplot(data=df, aes(x=x, y=y)) + 
  geom_point(aes(color=class, shape=class)) + 
  geom_point(aes(color=class, x=x, y=y), data=test, shape=16, size=10, alpha=0.07) +
  theme_bw() +
  theme(legend.position = "none", legend.text = element_text(size=14), legend.title = element_text(size=14)) +
  #theme(legend.) +
  theme(axis.ticks = element_blank(), axis.text = element_blank(), strip.text = element_text(size=16)) +
  scale_color_manual(values=colors) + 
  xlab("") + ylab("") + 
  facet_grid(.~drift_point)
g
ggsave("multi_class_global_distribution_drift_plot.pdf", width = 10.65, height = 3.55)
#ggsave("multi_class_local_distribution_drift_plot.pdf", width = 7.1, height = 3.55)
