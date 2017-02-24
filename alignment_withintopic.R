# Compute the lexical alignment between speakers, and correlate it with the within-topic positions
# Yang Xu
# 2/24/2017

library(data.table)
library(tidyverse)
library(stringr)
library(lme4)
library(lmerTest)


# read Switchboard text data
dt.swbd = fread('data/SWBD_text_db.csv')
setkey(dt.swbd, convId)
# glimpse()
# Observations: 155,372
# Variables: 5
# $ convId   <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
# $ turnId   <int> 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15,...
# $ speaker  <chr> "A", "A", "B", "B", "A", "A", "B", "B", "B", "A", "B", "B", "A", "B", "B", "B", "B", "B", "A", "...
# $ globalId <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 2...
# $ rawWord  <chr> "okay", "uh first um I need to know uh how do you feel about uh about sending uh an elderly uh f...

##
# the function that computes LLA (local linguistic alignment) between speaking turns
compute_LLA = function(data) {
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
            } else if (convId != prevConvId[.GRP]) {
                lla = NaN
            } else if (speaker == prevSpeaker[.GRP]) {
                lla = NaN
            } else {
                prime = str_split(prevTurnText[.GRP], ' ')[[1]]
                target = str_split(turnText, ' ')[[1]]
                lla = LLA(prime, target)
            }
            .(lla = lla)
        }, by = .(convId, turnId)]
    d.align
}

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


###
# Test with compute_LLA
system.time(dt.swbd.align <- compute_LLA(dt.swbd))
 #   user  system elapsed
 # 16.575   0.185  16.656
nrow(dt.swbd.align[is.nan(lla),]) # 4180
nrow(dt.swbd.align[is.nan(lla) & turnId==1,]) # 1112
nrow(dt.swbd.align[!is.nan(lla) & turnId==1,]) # 0
nrow(dt.swbd.align[turnId==1,]) # 1112


###
# Use linear models to test lla ~ turnId
m = lmer(lla ~ turnId + (1|convId), dt.swbd.align)
summary(m)
# Fixed effects:
#              Estimate Std. Error        df t value Pr(>|t|)
# (Intercept) 2.962e-02  4.202e-04 2.406e+03  70.493   <2e-16 ***
# turnId      2.547e-06  4.162e-06 5.647e+04   0.612    0.541

m = lmer(lla ~ log(turnId) + (1|convId), dt.swbd.align)
summary(m)
# Fixed effects:
#               Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)  3.694e-02  8.293e-04  2.480e+04  44.546   <2e-16 ***
# log(turnId) -1.922e-03  2.031e-04  8.704e+04  -9.462   <2e-16 ***

# plot lla ~ log(turnId)
p = ggplot(dt.swbd.align, aes(x = log(turnId), y = lla)) +
    geom_smooth(method='lm')


####
# Okay, how does LLA change within topic episode
dt.swbd.topic = fread('data/SWBD_entropy_db.csv')
setkey(dt.swbd.topic, convId, turnId)
# join
dt.swbd.comb = dt.swbd.topic[dt.swbd.align, nomatch=0]
# shrink inTopicId column by computing the mean
dt.swbd.comb = dt.swbd.comb[, {
        .(topicId = topicId[1], inTopicId = mean(inTopicId), lla = lla[1], ent = mean(ent))
    }, by = .(convId, turnId)]
# add uniqueTopicId
dt.swbd.comb[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# models
m = lmer(lla ~ inTopicId + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# inTopicId   -7.193e-05  4.264e-05  3.943e+04  -1.687   0.0916 .

m = lmer(lla ~ log(inTopicId) + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# log(inTopicId) -7.733e-04  2.827e-04  6.104e+04  -2.736  0.00623 **

m = lmer(lla ~ log(inTopicId) + log(turnId) + (1|uniqueTopicId), dt.swbd.comb)
summary(m)
# log(inTopicId)  1.498e-04  2.868e-04  5.922e+04   0.522    0.601
# log(turnId)    -5.358e-03  3.058e-04  1.109e+04 -17.519   <2e-16 ***

# OK, looks like LLA DECREASES within topic and along the whole dialogue!
# (kind of consistent with PG Healey's  PLOS one paper)

##
# How is lla correlated with entropy
m = lmer(lla ~ ent + (1|convId), dt.swbd.comb)
summary(m)
# ent         1.201e-03  4.953e-05 7.635e+04   24.26   <2e-16 ***

# shift `ent` column, so as to correlate lla with the entropy of preceding utterance
shiftedEnt = shift(dt.swbd.comb$ent)
dt.swbd.comb$shiftedEnt = shiftedEnt
# remove the first row of each conversation
dt.tmp = dt.swbd.comb[, .SD[2:.N,], by=convId]

m = lmer(lla ~ shiftedEnt + (1|convId), dt.swbd.comb)
summary(m)
# Fixed effects:
#              Estimate Std. Error        df t value Pr(>|t|)
# (Intercept) 1.727e-02  5.518e-04 5.814e+03   31.30   <2e-16 ***
# shiftedEnt  1.403e-03  4.953e-05 7.635e+04   28.32   <2e-16 ***

# Thus, higher entropy utterances tend to attract more alignment


###
# Does lla increase across the boundaries of topic shift
dt.swbd.bound = dt.swbd.comb[, {
        # find the positions where topic shift happens
        beforeInd = which(diff(topicId)==1)
        atInd = which(c(0, diff(topicId))==1)
        afterInd = atInd + 1
        .(llaBefore = lla[beforeInd], llaAt = lla[atInd], llaAfter = lla[afterInd])
    }, by = .(convId)]
# compare
t.test(dt.swbd.bound$llaBefore, dt.swbd.bound$llaAt)
# t = 5.0904, df = 18514, p-value = 3.609e-07
t.test(dt.swbd.bound$llaAt, dt.swbd.bound$llaAfter)
# t = 16.796, df = 18525, p-value < 2.2e-16
t.test(dt.swbd.bound$llaBefore, dt.swbd.bound$llaAfter)
# t = 21.338, df = 17992, p-value < 2.2e-16
# Alignment before boundaries is larger then alignment after boundaries
