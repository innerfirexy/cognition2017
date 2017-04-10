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

m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId<=8])
summary(m)
# inTopicId   9.845e-02  2.674e-02 3.218e+04   3.681 0.000233 ***
# increas at the biginning for the first 8 sents
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt1[globalId<=100 & inTopicId>9 & inTopicId<=20])
summary(m)
# n.s. no change between 9th and 20th sents


# plot ent ~ inTopicId
p = ggplot(dt1[globalId<=100 & inTopicId<=20], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')

##
# BNC
dt2 = fread('data/BNC_entropy_crossvalidate_samepos_dp.csv')
# add uniqueTopicId
dt2[, uniqueTopicId := .GRP, keyby = .(convId, topicId)]
# mean topic episode length
dt2.stats = dt2[, .N, keyby = uniqueTopicId]
mean(dt2.stats$N) # 13.1
sd(dt2.stats$N) # 35.1

# check ent ~ globalId
m = lmer(ent ~ globalId + (1|convId), dt2[globalId<100])
summary(m)
# increase

# check ent ~ inTopicId
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt2[globalId<=100])
summary(m)
# inTopicId   5.789e-02  4.493e-03 4.528e+04   12.88   <2e-16 ***
# increase

# plot ent ~ inTopicId
p = ggplot(dt2[globalId<=100 & inTopicId<=13], aes(x=inTopicId, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon') +
    stat_summary(fun.y = mean, geom = 'line')



#####
# MinCut

##
# SWBD
