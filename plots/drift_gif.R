library (ggplot2)
library (reshape2)

single_class_local <- read.csv(file = "../multi_class_local.csv")
#single_class_local <- read.csv(file = "../datasets/datasets/multi_class_global_swap_cluster_ds_10000_c_3_ca_3_f_2_1_1.csv")
#single_class_local <- read.csv(file = "../datasets/datasets/single_class_global_class_emerging_rt_ds_10000_c_3_ca_1_f_2_1_1.csv")
#single_class_local <- read.csv(file = "../datasets/datasets/single_class_local_emerging_branch_ds_10000_c_3_ca_1_f_2_1_1.csv")
#single_class_local <- read.csv(file="../single_class_local.csv")

require(MASS)
require(class)
require(dplyr)

test <- expand.grid(x=seq(min(single_class_local[,1] - 0.15), max(single_class_local[,1]+0.15),
                          by=0.1),
                    y=seq(min(single_class_local[,2] - 0.15), max(single_class_local[,2]+0.15), 
                          by=0.1))



single_class_local$id <- c(1:100000)


classif <- knn(single_class_local[,c(1,2)], test, single_class_local[,3], k = 3, prob=TRUE)
prob <- attr(classif, "prob")

dataf <- bind_rows(mutate(test,
                                 prob=prob,
                                 class=0,
                                 prob_class=ifelse(classif==class,
                                                   1, 0)),
                          mutate(test,
                                 prob=prob,
                                 class=1,
                                 prob_class=ifelse(classif==class,
                                                   1, 0)),
                          mutate(test,
                                 prob=prob,
                                 class=2,
                                 prob_class=ifelse(classif==class,
                                                   1, 0)))



df <- single_class_local
colnames(df) <- c("x","y", "class", "id")
df$class <- as.factor(df$class)


test$class <- classif

test$class <- factor(test$class)

df$frame <- rep(1:20, each=5000)


g <- ggplot(data=df, aes(x=x, y=y)) + 
  geom_point(aes(color=class, shape=class)) + 
  #geom_point(aes(color=class, x=x, y=y), data=test, shape=16, size=10, alpha=0.07) +
  theme_bw() +
  theme(legend.position = "none", legend.text = element_text(size=14), legend.title = element_text(size=14)) +
  #theme(legend.) +
  theme(axis.ticks = element_blank(), axis.text = element_blank(), strip.text = element_text(size=16)) +
  xlab("") + ylab("") 
g
#ggsave("multi_class_local_swap.pdf", width = 10.65, height = 3.55)

a <- animate(g + transition_time(frame) + labs(title = "Instances: {frame_time*5000}"), renderer = gifski_renderer(), nframes = 20, fps=1)
anim_save("../figures/multi_local_drift.gif", animation = a)
