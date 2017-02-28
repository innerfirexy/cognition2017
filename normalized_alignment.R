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
LLA = function(prime, target) {
    if (length(prime)==0 | length(target)==0) {
        return(NaN)
    }
    repeatcount = 0
    for (w in target) {
        if (w %in% prime) {
            repeatcount = repeatcount + 1
        }
    }
    repeatcount / (length(prime) + length(target))
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
                lla = NaN
                sumLen = NaN
            } else if (convId != prevConvId[.GRP]) {
                lla = NaN
                sumLen = NaN
            } else if (speaker == prevSpeaker[.GRP]) {
                lla = NaN
                sumLen = NaN
            } else {
                prime = str_split(prevTurnText[.GRP], ' ')[[1]]
                target = str_split(turnText, ' ')[[1]]
                lla = LLA(prime, target)
                sumLen = as.numeric(length(prime) + length(target))
            }
            .(lla = lla, sumLen = sumLen)
        }, by = .(convId, turnId)]
    d.align
}


###
# Compute the normalzied alignment for Switchboard
dt.swbd = fread('data/SWBD_text_db.csv')
setkey(dt.swbd, convId)

dt.swbd.align = compute_LLA_sumLen(dt.swbd)

# compute the mean lla for each sumLen level
setkey(dt.swbd.align, sumLen)
dt.swbd.align.mean = dt.swbd.align[, {.(llaMean = mean(lla[!is.nan(lla)]))}, by = sumLen]
# join `llaMean` column back to dt.swbd.align
dt.swbd.align = dt.swbd.align[dt.swbd.align.mean, nomatch = 0]
# compute the normalized lla
dt.swbd.align[, llaNorm := lla / llaMean]

##
# Use models to check how llaNorm changes within dialogue and topic episode
m = lmer(llaNorm ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# turnId      8.265e-04  1.555e-04 4.929e+04   5.316 1.07e-07 ***
# Yes! Alignment actually increases along dialogue

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
