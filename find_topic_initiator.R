# Given the topic segmentation information, find the topic initiators and responders for each topic episode
# Yang Xu
# 2/23/2017

library(data.table)

# read the input file
dt = fread('data/SWBD_entropy_db.csv')
setkey(dt, convId, topicId)
# head(dt)
#    convId turnId speaker globalId     ent wordNum topicId inTopicId
# 1:      1      1       A        1 1.48789       1       1         1
# 2:      1      1       A        2 8.64813      26       1         2
# 3:      1      2       B        3 6.30240      23       1         3
# 4:      1      2       B        4 7.12943      19       1         4
# 5:      1      3       A        5 6.24289       1       2         1
# 6:      1      3       A        6 4.01994       1       2         2

# function
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
# test with findInitiators
for (i in 1:10) {
    dt.found = dt[findInitiators(dt, thrhld=i), nomatch = 0]
    fwrite(dt.found, paste0('data/SWBD_entropy_db_initiators_thrhld', i, '.csv'))
}

# plot
for (i in 1:10) {
    dt.found = fread(paste0('data/SWBD_entropy_db_initiators_thrhld', i, '.csv'))
    dt.found[, group := 'initiator']
    dt.found[speaker != initiator, group := 'responder']

    p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
        stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
        stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
        stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
        scale_x_continuous(breaks = 1:10) +
        theme(legend.position = c(.75, .2)) +
        xlab('within-episode position') + ylab('entropy')
    pdf(paste0('figs/e_vs_inPos_role_SWBD_thrhld', i, '.pdf'), 4, 4)
    plot(p)
    dev.off()
}
##
# from the figs (thrhld 1 ~ 10)
# Once thrhld > 1, the convergence patterns are significant


##
# find topic initiators for SWBD_entropy_crossvalidate_topic.csv
dt = fread('data/SWBD_entropy_crossvalidate_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   2.579e-03  3.804e-04 9.983e+04    6.78 1.21e-11 ***
# NOTE: YES, although the entropy computed this way decreases with globalId
# It STILL increases within topic episode.


##
# find topic initiators for BNC_entropy_crossvalidate_topic.csv
dt = fread('data/BNC_entropy_crossvalidate_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -1.040e-03  4.484e-04  5.084e+04  -2.319   0.0204 *
# NOTE: this is inconsistent with Switchboard


##
# find topic initiators for SWBD_entropy_fromBNC_topic.csv
dt = fread('data/SWBD_entropy_fromBNC_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   1.471e-03  4.044e-04 9.789e+04   3.638 0.000275 ***
# NOTE: entropy increases with inTopicId


##
# find topic initiators for BNC_entropy_fromSWBD_topic.csv
dt = fread('data/BNC_entropy_fromSWBD_topic.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -1.235e-03  5.162e-04  4.963e+04  -2.393   0.0167 *
# NOTE: entropy decreases with inTopicId


##
# find topic initiators for SWBD_entropy_db_dp.csv
dt = fread('data/SWBD_entropy_db_dpconfig.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -6.610e-03  7.489e-04  3.602e+04  -8.825   <2e-16 ***
# NOTE: entropy decreases with inTopicId


##
# find topic initiators for SWBD_entropy_crossvalidate_dp.csv
dt = fread('data/SWBD_entropy_crossvalidate_dp.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -1.883e-03  3.282e-04  5.669e+04  -5.739 9.56e-09 ***
# NOTE: entropy decreases with inTopicId


##
# find topic initiators for SWBD_entropy_crossvalidate_samepos_dp.csv
dt = fread('data/SWBD_entropy_crossvalidate_samepos_dp.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -1.139e-02  1.901e-03  4.036e+04  -5.994 2.07e-09 ***
# NOTE: entropy decreases with inTopicId
m = lmer(ent ~ inTopicId + (1|convId), dt.found[inTopicId<=4,])
summary(m)


##
# find topic initiators for BNC_entropy_crossvalidate_dp.csv
dt = fread('data/BNC_entropy_crossvalidate_dp.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   -2.854e-03  1.565e-03  2.814e+04  -1.823   0.0683 .
# NOTE: entropy slightly decreases with inTopicId


##
# find topic initiators for BNC_entropy_crossvalidate_samepos_dp.csv
dt = fread('data/BNC_entropy_crossvalidate_samepos_dp.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   2.541e-02  4.180e-03 3.770e+04   6.079 1.22e-09 ***
# NOTE: entropy increases with inTopicId


##
# find topic initiators for SWBD_entropy_crossvalidate_samepos_mcsopt.csv
dt = fread('data/SWBD_entropy_crossvalidate_samepos_mcsopt.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   7.713e-03  1.819e-03 6.300e+04    4.24 2.23e-05 ***
# NOTE: entropy increases with inTopicId


##
# find topic initiators for BNC_entropy_crossvalidate_samepos_mcsopt.csv
dt = fread('data/BNC_entropy_crossvalidate_samepos_mcsopt.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# NOTE: in MinCut, we do not see a decrease in topic initiator
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   5.730e-02  3.821e-03 4.205e+04   15.00   <2e-16 ***
# NOTE: entropy increases with inTopicId

# Add `topicLen` column
dt.found[, topicLen := .N, by = .(convId, topicId)]
summary(dt.found$topicLen) # Well, this tells nothing
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.00   23.00   57.00   81.01  106.00  817.00
# filter out short topics
p = ggplot(dt.found[inTopicId <= 13 & topicId > 1 & topicLen>=30,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')


##
# find topic initiators for BNC_entropy_crossvalidate_samepos_tt.csv
dt = fread('data/BNC_entropy_crossvalidate_samepos_tt.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   4.128e-02  9.676e-03 5.157e+04   4.266    2e-05 ***
# NOTE: entropy increases with inTopicId


##
# for SWBD_entropy_crossvalidate_samepos_pseudofixed.csv
dt = fread('data/SWBD_entropy_crossvalidate_samepos_pseudofixed.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found)
summary(m)
# inTopicId   3.292e-02  1.206e-02 1.025e+05    2.73  0.00634 **
# NOTE: still increase, but not as strong effect

##
# for BNC_entropy_crossvalidate_samepos_pseudofixed.csv
dt = fread('data/BNC_entropy_crossvalidate_samepos_pseudofixed.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found[group=='initiator'])
summary(m)
# inTopicId   -3.567e-02  2.605e-02  3.018e+04   -1.37    0.171
# NOTE: Okay, does not decrease



##
# for SWBD_entropy_crossvalidate_samepos_pseudorand.csv
dt = fread('data/SWBD_entropy_crossvalidate_samepos_pseudorand.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found[group=='initiator'])
summary(m)
# inTopicId   -2.672e-01  1.733e-02  5.629e+04  -15.42   <2e-16 ***


##
# for BNC_entropy_crossvalidate_samepos_pseudorand.csv
dt = fread('data/BNC_entropy_crossvalidate_samepos_pseudorand.csv')
setkey(dt, convId, topicId)
dt.found = dt[findInitiators(dt, thrhld=5), nomatch=0]
# plot
dt.found[, group := 'initiator']
dt.found[speaker != initiator, group := 'responder']
p = ggplot(dt.found[inTopicId <= 10 & topicId > 1,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = group)) +
    scale_x_continuous(breaks = 1:10) +
    theme(legend.position = c(.75, .2)) +
    xlab('within-episode position') + ylab('entropy')
# model test if entropy increases within topic episode
m = lmer(ent ~ inTopicId + (1|convId), dt.found[group=='initiator'])
summary(m)
# inTopicId   -5.991e-02  2.185e-02  3.029e+04  -2.742  0.00611 **
