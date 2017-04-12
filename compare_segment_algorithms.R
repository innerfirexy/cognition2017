# Compare Bayes-seg and MinCut with TextTiling in:
# 1. whether they can capture the entropy increase patterns within topic episodes
# 2. how does entropy look like at boundaries?
# 3. can they demonstrate the entropy convergence patterns?
# Yang Xu
# 4/10/2017

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)


#####
# Bayes-seg

##
# SWBD
dt1 = fread('data/SWBD_entropy_crossvalidate_samepos_dp.csv')
# add uniqueTopicId
dt1[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt1.stats = dt1[, .N, keyby = uniqueTopicId]
mean(dt1.stats$N) # 19.7
median(dt1.stats$N) # 9
sd(dt1.stats$N) # 27.0

# check ent ~ globalId
m = lmer(ent ~ globalId + (1|convId), dt1[globalId<100])
summary(m)
# increase

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100])
summary(m)
# inTopicId   -8.580e-03  1.974e-03  2.250e+04  -4.345  1.4e-05 ***
# decrease

m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId <= 20])
summary(m)
# inTopicId   -7.270e-03  8.064e-03  5.737e+04  -0.902    0.367
# n.s. for the first 20 sentences

m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId<=9]) # Median length
summary(m)
# inTopicId   5.494e-02  2.280e-02 3.568e+04   2.409    0.016 *
# increas at the biginning for the first 8 sents
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId>9 & inTopicId<=20])
summary(m)
# n.s. no change between 9th and 20th sents


# plot ent ~ inTopicId
p = ggplot(dt1[globalId<=100 & inTopicId<=20], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')

#
# plot bayes-seg, mincut, and TextTiling together
dt.swbd1 = fread('data/SWBD_entropy_crossvalidate_samepos_dp.csv')
dt.swbd2 = fread('data/SWBD_entropy_crossvalidate_samepos_mcsopt.csv')
dt.swbd3 = fread('data/SWBD_entropy_crossvalidate_samepos_tt.csv')
dt.swbd1[, Algorithm := 'BayesianSeg'][, ent := ent+1]
dt.swbd2[, Algorithm := 'MinCutSeg']
dt.swbd3[, Algorithm := 'TextTiling'][, ent := ent-1]
dt.swbd = rbindlist(list(dt.swbd1, dt.swbd2, dt.swbd3))

p = ggplot(dt.swbd[(Algorithm=='BayesianSeg' & inTopicId<=20) | (Algorithm=='MinCutSeg' & inTopicId<=20) | (Algorithm=='TextTiling' & inTopicId<=9)],
        aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Algorithm), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Algorithm)) +
    geom_vline(xintercept=9, color='red', size=.75, lty='dashed') +
    theme_light() + theme(legend.position=c(.8,.15)) +
    labs(x='Relative sentence position within topic episode', y='Information (bit)')
pdf('figs/algo_compare_SWBD.pdf', 5, 5)
plot(p)
dev.off()


##
# BNC
dt2 = fread('data/BNC_entropy_crossvalidate_samepos_dp.csv')
# add uniqueTopicId
dt2[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt2.stats = dt2[, .N, keyby = uniqueTopicId]
mean(dt2.stats$N) # 13.1
median(dt2.stats$N) # 4
sd(dt2.stats$N) # 35.1

# check ent ~ globalId
m = lmer(ent ~ globalId + (1|convId), dt2[globalId<100])
summary(m)
# increase

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt2[globalId<=100 & inTopicId<=13])
summary(m)
# inTopicId   1.237e-01  2.218e-02 3.283e+04   5.578 2.45e-08 ***
# increase

# plot ent ~ inTopicId
p = ggplot(dt2[globalId<=100 & inTopicId<=13], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')

#
# plot bayes-seg, mincut, and TextTiling together
dt.bnc1 = fread('data/BNC_entropy_crossvalidate_samepos_dp.csv')
dt.bnc2 = fread('data/BNC_entropy_crossvalidate_samepos_mcsopt.csv')
dt.bnc3 = fread('data/BNC_entropy_db.csv')
dt.bnc1[, Algorithm := 'BayesianSeg'][, ent := ent+1]
dt.bnc2[, Algorithm := 'MinCutSeg']
dt.bnc3[, Algorithm := 'TextTiling'][, ent := ent-1]
dt.bnc = rbindlist(list(dt.bnc1, dt.bnc2,
    dt.bnc3[,.(convId, globalId, ent, speaker, wordNum, topicId, inTopicId, Algorithm)]))

p = ggplot(dt.bnc[(Algorithm=='BayesianSeg' & inTopicId<=13) | (Algorithm=='MinCutSeg' & inTopicId<=13) | (Algorithm=='TextTiling' & inTopicId<=13)],
        aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Algorithm), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Algorithm)) +
    # geom_vline(xintercept=9, color='red', size=.75, lty='dashed') +
    theme_light() + theme(legend.position=c(.15,.85)) +
    labs(x='Relative sentence position within topic episode', y='Information (bit)')
pdf('figs/algo_compare_BNC.pdf', 5, 5)
plot(p)
dev.off()

# dt.bnc3[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# m=lmer(ent~inTopicId+(1|uniqueTopicId), dt.bnc3)
# summary(m)
# p = ggplot(dt.bnc3[globalId<=100 & inTopicId<=13], aes(x=inTopicId, y=ent)) +
#     stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
#     stat_summary(fun.y = mean, geom = 'line')

#####
# MinCut

##
# SWBD
dt1 = fread('data/SWBD_entropy_crossvalidate_samepos_mcsopt.csv')
# add uniqueTopicId
dt1[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt1.stats = dt1[, .N, keyby = uniqueTopicId]
mean(dt1.stats$N) # 19.7
median(dt1.stats$N) # 7
sd(dt1.stats$N) # 26.7

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId<=20]) # Mean length
summary(m)
# inTopicId   5.089e-02  8.434e-03 5.339e+04   6.034 1.61e-09 ***

m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId<=7]) # Median length
summary(m)
# inTopicId   1.724e-01  3.544e-02 2.176e+04   4.865 1.15e-06 ***

# plot ent ~ inTopicId
p = ggplot(dt1[globalId<=100 & inTopicId<=20], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')


##
# BNC
dt2 = fread('data/BNC_entropy_crossvalidate_samepos_mcsopt.csv')
# add uniqueTopicId
dt2[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt2.stats = dt2[, .N, keyby = uniqueTopicId]
mean(dt2.stats$N) # 13.1
median(dt2.stats$N) # 1
sd(dt2.stats$N) # 29.9

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt2[globalId<=100 & inTopicId<=13]) # Mean length
summary(m)
# inTopicId   1.834e-01  2.170e-02 2.825e+04   8.453   <2e-16 ***

# plot ent ~ inTopicId
p = ggplot(dt2[globalId<=100 & inTopicId<=13], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')



#####
# TextTiling

##
# SWBD
dt1 = fread('data/SWBD_entropy_crossvalidate_samepos_tt.csv')
dt1[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt1.stats = dt1[, .N, keyby = uniqueTopicId]
mean(dt1.stats$N) # 9.2
median(dt1.stats$N) # 8
sd(dt1.stats$N) # 5.7

##
# BNC
# dt2 = fread('data/BNC_entropy_crossvalidate_samepos_tt.csv')
dt2 = fread('data/BNC_entropy_db.csv')
dt2[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt2.stats = dt2[, .N, keyby = uniqueTopicId]
mean(dt2.stats$N) # 10.9
median(dt2.stats$N) # 10
sd(dt2.stats$N) # 7.3

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt2) # Mean length
summary(m)
