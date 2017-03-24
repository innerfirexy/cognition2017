# Analyze how the normalized LLA measure (from length effect) changes within topic episode and dialogue
# Yang Xu
# 2/28/2017

library(data.table)
library(tidyverse)
library(stringr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggalt)


# LLA function
LLA = function(prime, target, normalizer='sum') {
    # examine arguments
    stopifnot(normalizer %in% c('sum', 'prod', 'sqrtprod'))

    if (length(prime)==0 | length(target)==0) {
        return(NaN)
    }
    repeatcount = 0
    for (w in target) {
        if (w %in% prime) {
            repeatcount = repeatcount + 1
        }
    }
    switch(normalizer,
        sum = repeatcount / (length(prime) + length(target)),
        prod = repeatcount / (length(prime) * length(target)),
        sqrtprod = repeatcount / sqrt(length(prime) * length(target))
        )
}

# the function that computes LLA (local linguistic alignment) between speaking turns
# This is a modified function from the one in alignment_withintopic.R
# It outputes the sum of prime and target length as well, in addition to LLA
compute_LLA_sumLen = function(data) {
    d1 = copy(data)
    setkey(d1, convId, turnId)
    # join text within the same turn
    d2 = d1[, {
            # words = sapply(rawWord, function(text) { str_split(text, ' ')[[1]] })
            .(turnText = str_c(rawWord, collapse = ' '), speaker = speaker[1])
        }, by = .(convId, turnId)]
    # previous info
    prevConvId = shift(d2[, last(convId), by = .(convId, turnId)]$V1)
    prevSpeaker = shift(d2[, last(speaker), by = .(convId, turnId)]$V1)
    prevTurnText = shift(d2[, last(turnText), by = .(convId, turnId)]$V1)
    # LLA
    d.align = d2[, {
            currWords = str_split(turnText, ' ')[[1]]
            if (is.na(prevConvId[.GRP])) {
                lla_sum = NaN
                lla_prod = NaN
                lla_sqrtprod = NaN
                sumLen = NaN
            } else if (convId != prevConvId[.GRP]) {
                lla_sum = NaN
                lla_prod = NaN
                lla_sqrtprod = NaN
                sumLen = NaN
            } else if (speaker == prevSpeaker[.GRP]) {
                lla_sum = NaN
                lla_prod = NaN
                lla_sqrtprod = NaN
                sumLen = NaN
            } else {
                prime = str_split(prevTurnText[.GRP], ' ')[[1]]
                target = str_split(turnText, ' ')[[1]]
                lla_sum = LLA(prime, target, normalizer='sum')
                lla_prod = LLA(prime, target, normalizer='prod')
                lla_sqrtprod = LLA(prime, target, normalizer='sqrtprod')
                sumLen = as.numeric(length(prime) + length(target))
            }
            .(lla_sum = lla_sum, lla_prod = lla_prod, lla_sqrtprod = lla_sqrtprod, sumLen = sumLen)
        }, by = .(convId, turnId)]
    d.align
}


###
# Compute the normalzied alignment for Switchboard
dt.swbd = fread('data/SWBD_text_db.csv')
setkey(dt.swbd, convId)

system.time(dt.swbd.align <- compute_LLA_sumLen(dt.swbd)) # elapsed 27.734 sec

# compute the mean lla for each sumLen level
setkey(dt.swbd.align, sumLen)
dt.swbd.align.mean = dt.swbd.align[, {
    .(lla_sum_mean = mean(lla_sum[!is.nan(lla_sum)]),
      lla_prod_mean = mean(lla_prod[!is.nan(lla_prod)]),
      lla_sqrtprod_mean = mean(lla_sqrtprod[!is.nan(lla_sqrtprod)]))
    }, by = sumLen]
# join `lla_*_mean` columns back to dt.swbd.align
dt.swbd.align = dt.swbd.align[dt.swbd.align.mean, nomatch = 0]
# compute the normalized lla
dt.swbd.align[, lla_sum_norm := lla_sum / lla_sum_mean][, lla_prod_norm := lla_prod / lla_prod_mean][, lla_sqrtprod_norm := lla_sqrtprod / lla_sqrtprod_mean]

##
# Use models to check how lla_*_norm changes within dialogue and topic episode
m = lmer(lla_sum_norm ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      8.265e-04  1.555e-04 4.929e+04   5.316 1.07e-07 ***
# Yes! Alignment actually increases along dialogue

m = lmer(lla_prod_norm ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      8.048e-04  1.517e-04 4.299e+04   5.305 1.13e-07 ***

m = lmer(lla_sqrtprod_norm ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      8.183e-04  1.507e-04 4.814e+04   5.429 5.68e-08 ***

##
# check how lla_* (w/o normalizing) change within dialogue
m = lmer(lla_sum ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      2.547e-06  4.162e-06 5.647e+04   0.612    0.541 n.s.
m = lmer(lla_prod ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      3.002e-05  3.561e-06 1.599e+04    8.43   <2e-16 ***
m = lmer(lla_sqrtprod ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      -4.433e-06  9.433e-06  6.011e+04   -0.47    0.638 n.s.


# Read topic information data and join with alignment data
dt.swbd.topic = fread('data/SWBD_entropy_db.csv')
setkey(dt.swbd.topic, convId, turnId)
setkey(dt.swbd.align, convId, turnId)
dt.swbd.comb = dt.swbd.topic[dt.swbd.align, nomatch=0]

# shrink inTopicId column by computing the mean
dt.swbd.comb = dt.swbd.comb[, {
        .(topicId = topicId[1], inTopicId = mean(inTopicId),
        lla_sum = lla_sum[1], lla_sum_norm = lla_sum_norm[1],
        lla_prod = lla_prod[1], lla_prod_norm = lla_prod_norm[1],
        lla_sqrtprod = lla_sqrtprod[1], lla_sqrtprod_norm = lla_sqrtprod_norm[1],
        ent = mean(ent))
    }, by = .(convId, turnId)]
# add uniqueTopicId
dt.swbd.comb[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# models
# lla_*_norm ~ inTopicId
m = lmer(lla_sum_norm ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   9.899e-03  1.563e-03 4.167e+04   6.333 2.43e-10 ***
# lla_sum_norm increases within topic episode!
m = lmer(lla_prod_norm ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   7.342e-03  1.532e-03 3.891e+04   4.793 1.65e-06 ***
# lla_prod_norm increases
m = lmer(lla_sqrtprod_norm ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   8.815e-03  1.516e-03 4.134e+04   5.816 6.07e-09 ***
# lla_sqrtprod_norm increases

# lla_* ~ inTopicId
m = lmer(lla_prod ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   2.182e-04  3.360e-05 2.761e+04   6.493 8.54e-11 ***
m = lmer(lla_sqrtprod ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   -3.575e-04  9.770e-05  4.087e+04   -3.66 0.000253 ***
# Haha, opposite direction when we use square root as normalizer


m = lmer(llaNorm ~ ent + (1|convId), dt.swbd.comb)
summary(m)
# ent         4.326e-02  1.827e-03 7.636e+04   23.67   <2e-16 ***
# llaNorm is also sensitive to entropy

# add shifted entropy column
shiftedEnt = shift(dt.swbd.comb$ent)
dt.swbd.comb$shiftedEnt = shiftedEnt
dt.swbd.tmp = dt.swbd.comb[, .SD[2:.N,], by=convId]

m = lmer(llaNorm ~ shiftedEnt + (1|convId), dt.swbd.tmp)
summary(m)
# shiftedEnt  4.935e-02  1.828e-03 7.636e+04   27.00   <2e-16 ***
# llaNorm is correlated with the entropy of previous utterance


##
# How does llaNorm change across topic boundaries
dt.swbd.bound = dt.swbd.comb[, {
        # find the positions where topic shift happens
        beforeInd = which(diff(topicId)==1)
        atInd = which(c(0, diff(topicId))==1)
        afterInd1 = atInd + 1
        afterInd2 = atInd + 2
        .(lla_prod_norm_before = lla_prod_norm[beforeInd],
          lla_prod_norm_at = lla_prod_norm[atInd],
          lla_prod_norm_after1 = lla_prod_norm[afterInd1],
          lla_prod_norm_after2 = lla_prod_norm[afterInd2])
    }, by = .(convId)]
# melt
dt.swbd.bound.melt = melt(dt.swbd.bound, id=1, measures=2:4, variable.name='position', value.name='llaNorm')
# plot
p = ggplot(dt.swbd.bound.melt, aes(x=position, y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    annotate('text', x=2, y=1.01, label='Topic shift', color='#B22222', size=5) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Normalized LLA') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))
pdf('figs/llaNorm_acrossBound_SWBD.pdf', 5, 5)
plot(p)
dev.off()
#
# It shows that llaNorm decreases across topic boundary


##
# Plot llaNorm against inTopicId, with facet_wrap by topicId
mean(dt.swbd.comb[, max(inTopicId), by=uniqueTopicId]$V1) # 9
# create the `topicId_text` for facet_wrap
dt.swbd.comb[, topicId_text := paste0('Episode ', topicId)]
p = ggplot(dt.swbd.comb[topicId<=6 & inTopicId>=2 & inTopicId<=9,], aes(x=floor(inTopicId-1), y=lla_prod_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5) +
    stat_summary(fun.y = mean, geom = 'line') +
    facet_wrap(~topicId_text, nrow = 1) +
    xlab('Utterance position within topic episode') + ylab('Normalized LLA')
pdf('figs/llaNorm_vs_inTopicId_SWBD.pdf', 9, 2.5)
plot(p)
dev.off()



######
# The following analysis applies to BNC
dt.bnc = fread('data/BNC_text_db100.csv')
setkey(dt.bnc, convId)

system.time(dt.bnc.align <- compute_LLA_sumLen(dt.bnc)) # elapsed 11 sec

# compute the mean lla for each sumLen level
setkey(dt.bnc.align, sumLen)
dt.bnc.align.mean = dt.bnc.align[, {
        .(lla_sum_mean = mean(lla_sum[!is.nan(lla_sum)]),
          lla_prod_mean = mean(lla_prod[!is.nan(lla_prod)]),
          lla_sqrtprod_mean = mean(lla_sqrtprod[!is.nan(lla_sqrtprod)]))
    }, by = sumLen]
# join `llaMean` column back to dt.bnc.align
dt.bnc.align = dt.bnc.align[dt.bnc.align.mean, nomatch = 0]
# compute the normalized lla
dt.bnc.align[, lla_sum_norm := lla_sum / lla_sum_mean][, lla_prod_norm := lla_prod / lla_prod_mean][, lla_sqrtprod_norm := lla_sqrtprod / lla_sqrtprod_mean]

# Use models to check how llaNorm changes within dialogue and topic episode
# lla_*_norm ~ turnId
m = lmer(lla_sum_norm ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      -5.764e-04  5.217e-04  1.375e+04  -1.105    0.269 n.s., decrease
m = lmer(lla_prod_norm ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      -7.061e-04  5.168e-04  1.300e+04  -1.366    0.172 n.s., decrease
m = lmer(lla_sqrtprod_norm ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      -6.338e-04  5.121e-04  1.358e+04  -1.238    0.216 n.s., decrease

# lla_* ~ turnId
m = lmer(lla_sum ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      -3.221e-05  2.212e-05  1.951e+04  -1.456    0.145 n.s., decrease
m = lmer(lla_prod ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      1.199e-06  1.974e-05 1.344e+04   0.061    0.952 n.s.
m = lmer(lla_sqrtprod ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# turnId      -6.926e-05  5.053e-05  2.423e+04  -1.371     0.17 n.s., decrease


# Read topic information data and join with alignment data
dt.bnc.topic = fread('data/BNC_entropy_db.csv')
setkey(dt.bnc.topic, convId, turnId)
setkey(dt.bnc.align, convId, turnId)
dt.bnc.comb = dt.bnc.topic[dt.bnc.align, nomatch=0]

# shrink inTopicId column by computing the mean
dt.bnc.comb = dt.bnc.comb[, {
        .(topicId = topicId[1], inTopicId = mean(inTopicId),
        lla_sum = lla_sum[1], lla_sum_norm = lla_sum_norm[1],
        lla_prod = lla_prod[1], lla_prod_norm = lla_prod_norm[1],
        lla_sqrtprod = lla_sqrtprod[1], lla_sqrtprod_norm = lla_sqrtprod_norm[1],
        ent = mean(ent))
    }, by = .(convId, turnId)]
# add uniqueTopicId
dt.bnc.comb[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# models
# lla_*_norm ~ inTopicId
m = lmer(lla_sum_norm ~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   7.106e-03  1.754e-03 1.976e+04   4.051 5.12e-05 ***
# Yes, llaNorm increases within topic episode
m = lmer(lla_prod_norm ~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   7.569e-03  1.735e-03 1.907e+04   4.362  1.3e-05 ***
m = lmer(lla_sqrtprod_norm ~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   7.305e-03  1.719e-03 1.962e+04   4.249 2.16e-05 ***

# lla_* ~ inTopicId
m = lmer(lla_sum ~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   -1.200e-04  7.394e-05  2.035e+04  -1.623    0.105 n.s.
m = lmer(lla_prod~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   4.592e-04  6.502e-05 1.813e+04   7.062 1.71e-12 ***
m = lmer(lla_sqrtprod~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   -6.773e-04  1.688e-04  2.093e+04  -4.013 6.01e-05 *** decrease
# again, inconsistent with lla_prod



m = lmer(llaNorm ~ ent + (1|convId), dt.bnc.comb)
summary(m)
# ent         2.033e-02  2.156e-03 3.144e+04   9.428   <2e-16 ***

# add shifted entropy column
shiftedEnt = shift(dt.bnc.comb$ent)
dt.bnc.comb$shiftedEnt = shiftedEnt
dt.bnc.tmp = dt.bnc.comb[, .SD[2:.N,], by=convId]

m = lmer(llaNorm ~ shiftedEnt + (1|convId), dt.bnc.tmp)
summary(m)
# shiftedEnt  2.442e-02  2.174e-03 3.145e+04   11.23   <2e-16 ***
# llaNorm correlates with entropy of previous utterance


##
# How does llaNorm change across topic boundaries
dt.bnc.bound = dt.bnc.comb[, {
        # find the positions where topic shift happens
        beforeInd = which(diff(topicId)==1)
        atInd = which(c(0, diff(topicId))==1)
        afterInd1 = atInd + 1
        afterInd2 = atInd + 2
        .(lla_prod_norm_before = lla_prod_norm[beforeInd],
          lla_prod_norm_at = lla_prod_norm[atInd],
          lla_prod_norm_after1 = lla_prod_norm[afterInd1],
          lla_prod_norm_after2 = lla_prod_norm[afterInd2])
    }, by = .(convId)]
# melt
dt.bnc.bound.melt = melt(dt.bnc.bound, id=1, measures=2:4, variable.name='position', value.name='llaNorm')
# dt.bnc.bound.melt$position = as.numeric(dt.bnc.bound.melt$position)
# plot
p = ggplot(dt.bnc.bound.melt, aes(x=position, y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Normalized LLA') +
    annotate('text', x=2, y=.98, label='Topic shift', color='#B22222', size=5) +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))
pdf('figs/llaNorm_acrossBound_BNC.pdf', 5, 5)
plot(p)
dev.off()


##
# Plot llaNorm against inTopicId, with facet_wrap by topicId
mean(dt.bnc.comb[, max(inTopicId), by=uniqueTopicId]$V1) # 9
# create column for facet_wrap
dt.bnc.comb[, topicId_text := paste0('Episode ', topicId)]
p = ggplot(dt.bnc.comb[topicId<=6 & inTopicId>=2 & inTopicId<=9,], aes(x=floor(inTopicId-1), y=lla_prod_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5) +
    stat_summary(fun.y = mean, geom = 'line') +
    facet_wrap(~topicId, nrow = 1) +
    xlab('Utterance position within topic episode') + ylab('Normalized LLA')
pdf('figs/llaNorm_vs_inTopicId_BNC.pdf', 9, 2.5)
plot(p)
dev.off()


##
# Plot lla_prod_norm ~ inTopicId for SWBD and BNC together
dt.swbd.tmp = dt.swbd.comb[, .(topicId, inTopicId, lla_prod_norm, lla_prod, topicId_text)]
dt.swbd.tmp[, Corpus := 'SWBD']
dt.bnc.tmp = dt.bnc.comb[, .(topicId, inTopicId, lla_prod_norm, lla_prod, topicId_text)]
dt.bnc.tmp[, Corpus := 'BNC']
dt.comb = rbindlist(list(dt.swbd.tmp, dt.bnc.tmp))

p = ggplot(dt.comb[topicId<=6 & inTopicId>=2 & inTopicId<=9,], aes(x=floor(inTopicId-1), y=lla_prod_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill=Corpus)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Corpus)) +
    facet_wrap(~topicId_text, nrow = 1) +
    xlab('Utterance position within topic episode') + ylab('Normalized LLA') +
    theme_light() + theme(legend.position=c(.9, .7))
pdf('figs/nLLA_vs_inTopicId_together.pdf', 9, 2.5)
plot(p)
dev.off()
