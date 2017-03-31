# To address Reviewer 2's point of comparing inTopciId vs. globalId on entropy
# Also some code check the entropy near topic boundaries
# especially for topic initiators
# Yang Xu
# 3/29/2017

library(data.table)
library(lme4)
library(lmerTest)

# Use the findInitiators function from find_topic_initiator.R
findInitiators = function(data, thrhld) {
    d1 = copy(data)
    setkey(d1, convId, topicId)

    # add the `previous topicId` and the `last speaker of previous topic` columns
    prevSpeaker = shift(d1[, last(speaker), by = .(convId, topicId)]$V1)
    prevTopicId = shift(d1[, last(topicId), by = .(convId, topicId)]$V1)
    prevConvId = shift(d1[, last(convId), by = .(convId, topicId)]$V1)
    # d1[, `:=`(prevSpeaker = shiftedSpeaker[.GRP], prevTopicId = shiftedTopicId[.GRP]), by = .(convId, topicId)]

    # get the initiator column
    d2 = d1[, {
            if (is.na(prevConvId[.GRP])) { # first conversation
                initiator = speaker[1]
            } else if (convId != prevConvId[.GRP]) { # gap between conversations
                initiator = speaker[1]
            } else { # within same conversation
                if (speaker[1] == prevSpeaker[.GRP]) {
                    initiator = speaker[1]
                } else {
                    # choose the initiator according to threshold
                    initiator = speaker[which(wordNum >= thrhld)][1]
                }
            }
            .(initiator = initiator)
        }, by = .(convId, topicId)]
    d2
}


##
# Original TextTiling segmentation results

# SWBD
dt = fread('data/SWBD_entropy_crossvalidate_samepos_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
# add uniqueTopicId
dt.found[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# whether entropy decreases with inTopicId for initiator
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt.found[group=='initiator',])
summary(m)
# inTopicId   -1.086e-01  9.179e-03  4.304e+04  -11.83   <2e-16 ***

# alternative model with globalId
m1 = lmer(ent ~ globalId + (1|convId), dt.found[group=='initiator',])
summary(m1)
# full model
m2 = lmer(ent ~ inTopicId + globalId + (1|uniqueTopicId) + (1|convId), dt.found[group=='initiator',])
summary(m2)
# inTopicId   -1.449e-01  9.453e-03  3.977e+04  -15.33   <2e-16 ***
# globalId     3.067e-02  1.969e-03  9.692e+03   15.58   <2e-16 ***

# anova
anova(m, m1)
# Models:
# object: ent ~ inTopicId + (1 | uniqueTopicId)
# ..1: ent ~ globalId + (1 | convId)
#        Df    AIC    BIC  logLik deviance Chisq Chi Df Pr(>Chisq)
# object  4 425711 425747 -212852   425703
# ..1     4 425783 425818 -212887   425775     0      0          1

anova(m1, m2)
# Models:
# object: ent ~ globalId + (1 | convId)
# ..1: ent ~ inTopicId + globalId + (1 | uniqueTopicId) + (1 | convId)
#        Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)
# object  4 425783 425818 -212887   425775
# ..1     6 425219 425273 -212604   425207 567.47      2  < 2.2e-16 ***

anova(m, m2)
# Models:
# object: ent ~ inTopicId + (1 | uniqueTopicId)
# ..1: ent ~ inTopicId + globalId + (1 | uniqueTopicId) + (1 | convId)
#        Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)
# object  4 425711 425747 -212852   425703
# ..1     6 425219 425273 -212604   425207 496.27      2  < 2.2e-16 ***


# BNC
dt = fread('data/BNC_entropy_crossvalidate_samepos_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
# add uniqueTopicId
dt.found[, uniqueTopicId := .GRP, by = .(convId, topicId)]

# whether entropy decreases with inTopicId for initiator
m = lmer(ent ~ inTopicId + (1|uniqueTopicId), dt.found[group=='initiator',])
summary(m)
# inTopicId   -1.061e-01  1.328e-02  2.836e+04   -7.99 1.33e-15 ***

# alternative model with globalId
m1 = lmer(ent ~ globalId + (1|convId), dt.found[group=='initiator',])
summary(m1)
# full model
m2 = lmer(ent ~ inTopicId + globalId + (1|uniqueTopicId) + (1|convId), dt.found[group=='initiator',])
summary(m2)

# anova
anova(m, m1)
# Models:
# object: ent ~ inTopicId + (1 | uniqueTopicId)
# ..1: ent ~ globalId + (1 | convId)
#        Df    AIC    BIC  logLik deviance Chisq Chi Df Pr(>Chisq)
# object  4 244919 244953 -122456   244911
# ..1     4 243205 243239 -121599   243197  1714      0  < 2.2e-16 ***

# NOTE:
# So, in BNC, globalId is a better predictor than inTopicId
# But this does NOT undermine our theoretical contribution.



####
# Check the entropy at topic boundaries

# Pseudo episodes, SWBD
dt = fread('data/SWBD_entropy_crossvalidate_samepos_pseudofixed.csv')
dt.bound = dt[, {
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
# melt
dt.bound.melt = melt(dt.bound, id=1, measures=2:5, variable.name='position', value.name='ent')
# plot
p = ggplot(dt.bound.melt, aes(x=position, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    # annotate('text', x=3, y=1.01, label='Topic shift', color='#B22222', size=5) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Per-word information content') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))


# Pseudo episodes, BNC
dt = fread('data/BNC_entropy_crossvalidate_samepos_pseudofixed.csv')
dt.bound = dt[, {
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
# melt
dt.bound.melt = melt(dt.bound, id=1, measures=2:5, variable.name='position', value.name='ent')
# plot
p = ggplot(dt.bound.melt, aes(x=position, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    # annotate('text', x=3, y=1.01, label='Topic shift', color='#B22222', size=5) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Per-word information content') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))
##
# T-test for position 0 vs. position 1
t.test(dt.bound.melt[position=='at', ent], dt.bound.melt[position=='after1', ent])
# t = 1.5394, df = 9114.1, p-value = 0.1237
# n.s., which is what we want


##
# Texttiling boundary, SWBD
dt = fread('data/SWBD_entropy_crossvalidate_samepos_topic.csv')
dt.bound = dt[, {
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
# melt
dt.bound.melt = melt(dt.bound, id=1, measures=2:5, variable.name='position', value.name='ent')
# plot
p = ggplot(dt.bound.melt, aes(x=position, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    # annotate('text', x=3, y=1.01, label='Topic shift', color='#B22222', size=5) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Per-word information content') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))

##
# Texttiling boundary, BNC
dt = fread('data/BNC_entropy_crossvalidate_samepos_topic.csv')
dt.bound = dt[, {
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
# melt
dt.bound.melt = melt(dt.bound, id=1, measures=2:5, variable.name='position', value.name='ent')
# plot
p = ggplot(dt.bound.melt, aes(x=position, y=ent)) +
    stat_summary(fun.data = mean_cl_boot, geom='errorbar', width=.2) +
    stat_summary(fun.y = mean, geom='point', size=3) +
    stat_summary(fun.y = mean, geom='line', lty=2, group=1) +
    # annotate('text', x=3, y=1.01, label='Topic shift', color='#B22222', size=5) +
    labs(x = 'Relative utterance position from topic boundary', y = 'Per-word information content') +
    scale_x_discrete(labels = c('-1', '0', '1', '2')) +
    theme_light() + theme(axis.text.x = element_text(size=12, color='#B22222', face='bold'))
