# Generate the figures in Cognition 2017 paper
# Author: Yang Xu
# Time: 2/22/2017

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)

# The palette with grey:
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


##
# Plot Figure 1
# combine the original Fig 1 & 2
dt.swbd = fread('data/SWBD_entropy_db.csv')
dt.swbd.norm = fread('data/SWBD_entropy_db_norm.csv')
dt.bnc = fread('data/BNC_entropy_db.csv')
dt.bnc.norm = fread('data/BNC_entropy_db_norm.csv')

# sent info
dt.swbd[, Corpus := 'SWBD']
dt.bnc[, Corpus := 'BNC']
dt.si = rbindlist(list(dt.swbd[, c(4, 5, 9)], dt.bnc[, c(4, 6, 10)]))
p = ggplot(dt.si[globalId<=100], aes(x = globalId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Corpus, lty=Corpus), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Corpus)) +
    scale_x_continuous(breaks = c(1,25,50,75,100)) +
    labs(x = 'Sentence position within dialogue', y = 'Sentence information (bit)') +
    theme_light() + theme(legend.position=c(.1,.8))
pdf('figs/si_vs_global.pdf', 5, 5)
plot(p)
dev.off()

# norm sent info
dt.swbd.norm[, Corpus := 'SWBD']
dt.bnc.norm[, Corpus := 'BNC']
dt.nsi = rbindlist(list(dt.swbd.norm[, c(4, 10, 11)], dt.bnc.norm[, c(4, 11, 12)]))
dt.nsi[Corpus=='BNC', ent_norm := ent_norm+.05] # place offset
dt.nsi[Corpus=='SWBD', ent_norm := ent_norm-.05]
p = ggplot(dt.nsi[globalId<=100], aes(x = globalId, y = ent_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Corpus, lty=Corpus), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Corpus)) +
    scale_x_continuous(breaks = c(1,25,50,75,100)) +
    labs(x = 'Sentence position within dialogue', y = 'Normalized sentence information') +
    theme_light() + theme(legend.position=c(.1,.8))
pdf('figs/nsi_vs_global.pdf', 5, 5)
plot(p)
dev.off()



##
# What are mean topic length
dt.swbd = fread('data/SWBD_entropy_db.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 9.22
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 5.66
summary(dt.swbd[, .N, keyby=.(convId, topicId)]$N)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.000   5.000   8.000   9.222  11.000  49.000

dt.bnc = fread('data/BNC_entropy_db.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 10.9
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 7.3
summary(dt.bnc[, .N, by=.(convId, topicId)]$N)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.00    5.00   10.00   10.85   15.00   52.00

# other topic segmentation results
# dp (bayes-seg)
dt.swbd = fread('data/SWBD_text_db_dp.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 19.7
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 27.0

dt.bnc = fread('data/BNC_text_dbfull_mlrcut_dp.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 13.1
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 35.1

# mcsopt (mincut)
dt.swbd = fread('data/SWBD_text_db_mcsopt.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 19.7
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 26.8

dt.bnc = fread('data/BNC_text_dbfull_mlrcut_mcsopt.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 13.1
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 29.9
