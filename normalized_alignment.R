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
        sqrtprod = repeatcount / sqrt(length(prime) + length(target))
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
# turnId      8.265e-04  1.555e-04 4.929e+04   5.316 1.07e-07 ***

##
# check how lla_* (w/o normalizing) change within dialogue


# Read topic information data and join with alignment data
dt.swbd.topic = fread('data/SWBD_entropy_db.csv')
setkey(dt.swbd.topic, convId, turnId)
setkey(dt.swbd.align, convId, turnId)
dt.swbd.comb = dt.swbd.topic[dt.swbd.align, nomatch=0]

# shrink inTopicId column by computing the mean
dt.swbd.comb = dt.swbd.comb[, {
        .(topicId = topicId[1], inTopicId = mean(inTopicId), llaNorm = llaNorm[1], ent = mean(ent))
    }, by = .(convId, turnId)]
# add uniqueTopicId
dt.swbd.comb[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# models
m = lmer(llaNorm ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   9.899e-03  1.563e-03 4.167e+04   6.333 2.43e-10 ***
# llaNorm increases within topic episode!

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
        afterInd = atInd + 1
        .(llaNormBefore = llaNorm[beforeInd], llaNormAt = llaNorm[atInd], llaNormAfter = llaNorm[afterInd])
    }, by = .(convId)]
# melt
dt.swbd.bound.melt = melt(dt.swbd.bound, id=1, measures=2:4, variable.name='position', value.name='llaNorm')
# plot
p = ggplot(dt.swbd.bound.melt, aes(x=position, y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar')
pdf('figs/llaNorm_acrossBound_SWBD.pdf', 5, 5)
plot(p)
dev.off()
#
# It shows that llaNorm decreases across topic boundary


##
# Plot llaNorm against inTopicId, with facet_wrap by topicId
mean(dt.swbd.comb[, max(inTopicId), by=uniqueTopicId]$V1) # 9
p = ggplot(dt.swbd.comb[topicId<=6 & inTopicId>=2 & inTopicId<=9,], aes(x=floor(inTopicId-1), y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5) +
    stat_summary(fun.y = mean, geom = 'line') +
    facet_wrap(~topicId, nrow = 1) +
    xlab('within topic position of utterance') + ylab('LLA normalized by length')
pdf('figs/llaNorm_vs_inTopicId_SWBD.pdf', 9, 2.5)
plot(p)
dev.off()



######
# The following analysis applies to BNC
dt.bnc = fread('data/BNC_text_db100.csv')
setkey(dt.bnc, convId)

dt.bnc.align = compute_LLA_sumLen(dt.bnc) # takes a few seconds

# compute the mean lla for each sumLen level
setkey(dt.bnc.align, sumLen)
dt.bnc.align.mean = dt.bnc.align[, {.(llaMean = mean(lla[!is.nan(lla)]))}, by = sumLen]
# join `llaMean` column back to dt.bnc.align
dt.bnc.align = dt.bnc.align[dt.bnc.align.mean, nomatch = 0]
# compute the normalized lla
dt.bnc.align[, llaNorm := lla / llaMean]

# Use models to check how llaNorm changes within dialogue and topic episode
m = lmer(llaNorm ~ turnId + (1|convId), dt.bnc.align)
summary(m)
# n.s.
m = lmer(llaNorm ~ log(turnId) + (1|convId), dt.bnc.align)
summary(m)
# n.s.

# Read topic information data and join with alignment data
dt.bnc.topic = fread('data/BNC_entropy_db.csv')
setkey(dt.bnc.topic, convId, turnId)
setkey(dt.bnc.align, convId, turnId)
dt.bnc.comb = dt.bnc.topic[dt.bnc.align, nomatch=0]

# shrink inTopicId column by computing the mean
dt.bnc.comb = dt.bnc.comb[, {
        .(topicId = topicId[1], inTopicId = mean(inTopicId), llaNorm = llaNorm[1], ent = mean(ent))
    }, by = .(convId, turnId)]
# add uniqueTopicId
dt.bnc.comb[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# models
m = lmer(llaNorm ~ inTopicId + (1|uniqueTopicId), dt.bnc.comb)
summary(m)
# inTopicId   7.558e-03  1.767e-03 1.958e+04   4.277  1.9e-05 ***
# Yes, llaNorm increases within topic episode

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
        afterInd = atInd + 1
        .(llaNormBefore = llaNorm[beforeInd], llaNormAt = llaNorm[atInd], llaNormAfter = llaNorm[afterInd])
    }, by = .(convId)]
# melt
dt.bnc.bound.melt = melt(dt.bnc.bound, id=1, measures=2:4, variable.name='position', value.name='llaNorm')
# plot
p = ggplot(dt.bnc.bound.melt, aes(x=position, y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar')
pdf('figs/llaNorm_acrossBound_BNC.pdf', 5, 5)
plot(p)
dev.off()


##
# Plot llaNorm against inTopicId, with facet_wrap by topicId
mean(dt.bnc.comb[, max(inTopicId), by=uniqueTopicId]$V1) # 9
p = ggplot(dt.bnc.comb[topicId<=6 & inTopicId>=2 & inTopicId<=9,], aes(x=floor(inTopicId-1), y=llaNorm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5) +
    stat_summary(fun.y = mean, geom = 'line') +
    facet_wrap(~topicId, nrow = 1) +
    xlab('within topic position of utterance') + ylab('LLA normalized by length')
pdf('figs/llaNorm_vs_inTopicId_BNC.pdf', 9, 2.5)
plot(p)
dev.off()
