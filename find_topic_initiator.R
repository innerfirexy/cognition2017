# Given the topic segmentation information, find the topic initiators and responders for each topic episode
# Yang Xu
# 2/23/2017

library(data.table)

# read the input file
dt = fread('data/SWBD_entropy_db.csv')
setkey(dt, convId, topicId)
# head(dt)
#    convId turnId speaker globalId     ent topicId inTopicId
# 1:      1      1       A        1 1.48789       1         1
# 2:      1      1       A        2 8.64813       1         2
# 3:      1      2       B        3 6.30240       1         3
# 4:      1      2       B        4 7.12943       1         4
# 5:      1      3       A        5 6.24289       2         1
# 6:      1      3       A        6 4.01994       2         2


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
