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
    stat_summary(fun.y = mean, geom='point', size=2.5, aes(shape=Algorithm)) +
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
    stat_summary(fun.y = mean, geom='point', size=2.5, aes(shape=Algorithm)) +
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
# inTopicId   1.309e-02  3.740e-03 3.556e+04     3.5 0.000466 ***



######
# Plot the entropy change near topic boundaries
# Use the method in model_comparison.R

# the func that returns the boundary positions, and melt the data
getBound = function(dt) {
    dt.b = dt[, {
        # find the positions where topic shift happens
        beforeInd1 = which(diff(topicId)==1)
        atInd = which(c(0, diff(topicId))==1)
        afterInd1 = atInd + 1
        afterInd2 = atInd + 2
        .(before1 = ent[beforeInd1],
          at = ent[atInd],
          after1 = ent[afterInd1],
          after2 = ent[afterInd2])
    }, by = .(convId)]
    dt.bm = melt(dt.b, id=1, measures=2:5, variable.name='position', value.name='ent')
    dt.bm
}

# SWBD
dt.swbd1 = fread('data/SWBD_entropy_crossvalidate_samepos_dp.csv')
dt.swbd2 = fread('data/SWBD_entropy_crossvalidate_samepos_mcsopt.csv')
dt.swbd3 = fread('data/SWBD_entropy_crossvalidate_samepos_tt.csv')
dt.swbd1.b = getBound(dt.swbd1)
dt.swbd2.b = getBound(dt.swbd2)
dt.swbd3.b = getBound(dt.swbd3)
dt.swbd1.b[, Algorithm := 'BayesianSeg'][, ent := ent+1]
dt.swbd2.b[, Algorithm := 'MinCutSeg']
dt.swbd3.b[, Algorithm := 'TextTiling'][, ent := ent-1]
dt.swbd.b = rbindlist(list(dt.swbd1.b, dt.swbd2.b, dt.swbd3.b))

d.rec = data.table(x1=1.75, x2=2.25, y1=6, y2=12)
p = ggplot(dt.swbd.b, aes(x=position, y=ent, group=Algorithm)) +
    stat_summary(fun.data = mean_cl_boot, geom='ribbon', alpha=.5, aes(fill=Algorithm)) +
    stat_summary(fun.y = mean, geom='point', size=2.5, aes(shape=Algorithm)) +
    stat_summary(fun.y = mean, geom='line', aes(lty=Algorithm)) +
    annotate('text', x=2, y=10, label='Topic shift', color='#B22222', size=5) +
    geom_rect(data=d.rec, aes(xmin=x1, xmax=x2, ymin=y1, ymax=y2, fill=T), fill='grey', alpha=.5, inherit.aes=F) +
    labs(x = 'Relative sentence position from topic boundary', y = 'Sentence information (bit)') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() +
    theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'), legend.position=c(.7, .2))
pdf('figs/algo_compare_bound_SWBD.pdf', 5, 5)
plot(p)
dev.off()

# t-test between `before1` and `at`
t.test(dt.swbd.b[Algorithm=='BayesianSeg' & position=='before1' & !is.na(ent), ent],
    dt.swbd.b[Algorithm=='BayesianSeg' & position=='at' & !is.na(ent), ent])
# t = 3.511, df = 9685.1, p-value = 0.0004485
t.test(dt.swbd.b[Algorithm=='MinCutSeg' & position=='before1' & !is.na(ent), ent],
    dt.swbd.b[Algorithm=='MinCutSeg' & position=='at' & !is.na(ent), ent])
# t = 6.7258, df = 8124.1, p-value = 1.864e-11


# BNC
dt.bnc1 = fread('data/BNC_entropy_crossvalidate_samepos_dp.csv')
dt.bnc2 = fread('data/BNC_entropy_crossvalidate_samepos_mcsopt.csv')
dt.bnc3 = fread('data/BNC_entropy_db.csv')
dt.bnc1.b = getBound(dt.bnc1)
dt.bnc2.b = getBound(dt.bnc2)
dt.bnc3.b = getBound(dt.bnc3)
dt.bnc1.b[, Algorithm := 'BayesianSeg'][, ent := ent+1]
dt.bnc2.b[, Algorithm := 'MinCutSeg']
dt.bnc3.b[, Algorithm := 'TextTiling'][, ent := ent-1]
dt.bnc.b = rbindlist(list(dt.bnc1.b, dt.bnc2.b, dt.bnc3.b))

d.rec = data.table(x1=1.75, x2=2.25, y1=9.5, y2=14.5)
p = ggplot(dt.bnc.b, aes(x=position, y=ent, group=Algorithm)) +
    stat_summary(fun.data = mean_cl_boot, geom='ribbon', alpha=.5, aes(fill=Algorithm)) +
    stat_summary(fun.y = mean, geom='point', size=2.5, aes(shape=Algorithm)) +
    stat_summary(fun.y = mean, geom='line', aes(lty=Algorithm)) +
    annotate('text', x=2, y=12, label='Topic shift', color='#B22222', size=5) +
    geom_rect(data=d.rec, aes(xmin=x1, xmax=x2, ymin=y1, ymax=y2, fill=T), fill='grey', alpha=.5, inherit.aes=F) +
    labs(x = 'Relative sentence position from topic boundary', y = 'Sentence information (bit)') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() +
    theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'), legend.position=c(.7, .45))
pdf('figs/algo_compare_bound_BNC.pdf', 5, 5)
plot(p)
dev.off()

# t-test between `before1` and `at`
t.test(dt.bnc.b[Algorithm=='BayesianSeg' & position=='before1' & !is.na(ent), ent],
    dt.bnc.b[Algorithm=='BayesianSeg' & position=='at' & !is.na(ent), ent])
# t = 3.4934, df = 10665, p-value = 0.0004789
t.test(dt.bnc.b[Algorithm=='MinCutSeg' & position=='before1' & !is.na(ent), ent],
    dt.bnc.b[Algorithm=='MinCutSeg' & position=='at' & !is.na(ent), ent])
# t = 4.7961, df = 10378, p-value = 1.64e-06
