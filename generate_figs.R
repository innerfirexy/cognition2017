# Generate the figures in Cognition 2017 paper
# Author: Yang Xu
# Time: 2/22/2017

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)

# The palette with grey:
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


#######################
# Plot Figure 1
# combine the original Fig 1 & 2
#######################

dt.swbd = fread('data/SWBD_entropy_db.csv')
dt.swbd.norm = fread('data/SWBD_entropy_db_norm.csv')
dt.bnc = fread('data/BNC_entropy_db.csv')
dt.bnc.norm = fread('data/BNC_entropy_db_norm.csv')

# sent info
dt.swbd[, Corpus := 'SWBD']
dt.bnc[, Corpus := 'BNC']
dt.si = rbindlist(list(dt.swbd[, c(4, 5, 9)], dt.bnc[, c(4, 6, 10)]))
p = ggplot(dt.si[globalId<=100], aes(x = globalId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Corpus, lty=Corpus), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Corpus)) +
    scale_x_continuous(breaks = c(1,25,50,75,100)) +
    labs(x = 'Sentence position within dialogue', y = 'Sentence information (bits)') +
    theme_light() + theme(legend.position=c(.1,.8))
pdf('figs/si_vs_global.pdf', 5, 5)
plot(p)
dev.off()

# norm sent info
dt.swbd.norm[, Corpus := 'SWBD']
dt.bnc.norm[, Corpus := 'BNC']
dt.nsi = rbindlist(list(dt.swbd.norm[, c(4, 10, 11)], dt.bnc.norm[, c(4, 11, 12)]))
dt.nsi[Corpus=='BNC', ent_norm := ent_norm+.05] # place offset
dt.nsi[Corpus=='SWBD', ent_norm := ent_norm-.05]
p = ggplot(dt.nsi[globalId<=100], aes(x = globalId, y = ent_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', aes(fill=Corpus, lty=Corpus), alpha=.5) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty=Corpus)) +
    scale_x_continuous(breaks = c(1,25,50,75,100)) +
    labs(x = 'Sentence position within dialogue', y = 'Normalized sentence information') +
    theme_light() + theme(legend.position=c(.1,.8))
pdf('figs/nsi_vs_global.pdf', 5, 5)
plot(p)
dev.off()


##
# Statistical tests for Fig.1 (a), sentence information
# sent info increases with global position
m = lmer(log(ent) ~ globalId + (1|convId), dt.swbd)
summary(m)
# globalId    3.947e-04  5.946e-05 1.036e+05   6.638  3.2e-11 ***
m = lmer(log(ent) ~ globalId + (1|convId), dt.bnc)
summary(m)
# globalId    1.424e-03  8.542e-05 4.212e+04   16.67   <2e-16 ***

# Extra model for SWBD (after the early boost)
m = lmer(log(ent) ~ globalId + (1|convId), dt.swbd[globalId>=10])
summary(m)
# globalId    2.060e-04  6.848e-05 9.367e+04   3.008  0.00263 **


##
# Statistical tests for Fig.2 (b), normalied sentence information
# norm sent info increases withi global position
m = lmer(log(ent_norm) ~ globalId + (1|convId), dt.swbd.norm)
summary(m)
# globalId     5.727e-04  4.598e-05  1.036e+05   12.46   <2e-16 ***
m = lmer(log(ent_norm) ~ globalId + (1|convId), dt.bnc.norm)
summary(m)
# globalId     1.392e-03  8.221e-05  4.083e+04   16.93   <2e-16 ***

# Extra model for SWBD (after the early boost)
m = lmer(log(ent_norm) ~ globalId + (1|convId), dt.swbd.norm[globalId>=10])
summary(m)
# globalId     2.979e-04  5.239e-05  9.370e+04   5.687  1.3e-08 ***



#######################
# Plot Fig 2
# SI and NSI against within-topic position facet_wrap by topic episode
#######################

# get ggplot default colors
gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}
my_colors = gg_color_hue(2)

# read data, rename and combine
dt.swbd = fread('data/SWBD_entropy_db.csv')
dt.swbd.norm = fread('data/SWBD_entropy_db_norm.csv')
dt.bnc = fread('data/BNC_entropy_db.csv')
dt.bnc.norm = fread('data/BNC_entropy_db_norm.csv')

# sent info
dt.swbd[, Corpus:='SWBD']
dt.bnc[, Corpus:='BNC']
dt.si = rbindlist(list(dt.swbd[,c('ent', 'topicId', 'inTopicId', 'Corpus')],
    dt.bnc[,c('ent', 'topicId', 'inTopicId', 'Corpus')]))
dt.si$episodeId = paste('Episode', as.character(dt.si$topicId))

p = ggplot(dt.si[inTopicId <= 10 & topicId<=6,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Corpus)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Corpus)) +
    facet_wrap(~episodeId, nrow = 1) +
    scale_x_continuous(breaks = 1:10) +
    scale_fill_manual(values = c('BNC' = my_colors[1], 'SWBD' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC' = 1, 'SWBD' = 3)) +
    theme_light() + theme(legend.position = c(.85, .1), legend.direction='horizontal') +
    xlab('Relative sentence position within topic episode') + ylab('Sentence information (bits)')
pdf('figs/si_vs_inPos_facet.pdf', 9, 2.5)
plot(p)
dev.off()

# norm sent info
dt.swbd.norm[, Corpus:='SWBD']
dt.bnc.norm[, Corpus:='BNC']
dt.nsi = rbindlist(list(dt.swbd.norm[,c('ent_norm', 'topicId', 'inTopicId', 'Corpus')],
    dt.bnc.norm[,c('ent_norm', 'topicId', 'inTopicId', 'Corpus')]))
dt.nsi$episodeId = paste('Episode', as.character(dt.nsi$topicId))
# add offsets
dt.nsi[Corpus=='SWBD', ent_norm := ent_norm - .05]
dt.nsi[Corpus=='BNC', ent_norm := ent_norm + .05]

p = ggplot(dt.nsi[inTopicId <= 10 & topicId<=6,], aes(x = inTopicId, y = ent_norm)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Corpus)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Corpus)) +
    facet_wrap(~episodeId, nrow = 1) +
    scale_x_continuous(breaks = 1:10) +
    scale_fill_manual(values = c('BNC' = my_colors[1], 'SWBD' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC' = 1, 'SWBD' = 3)) +
    theme_light() + theme(legend.position = c(.85, .1), legend.direction='horizontal') +
    xlab('Relative sentence position within topic episode') + ylab('Normalized sentence information')
pdf('figs/nsi_vs_inPos_facet.pdf', 9, 2.5)
plot(p)
dev.off()



#######################
# Plot Fig.3
#######################

# read
dt.swbd = readRDS('data/swbd.leader.new.rds')
dt.bnc = readRDS('data/bnc.leader.tdbf.rds')
setnames(dt.swbd, c('topicID','inTopicID'), c('topicId', 'inTopicId'))
setnames(dt.bnc, c('topicID','inTopicID'), c('topicId', 'inTopicId'))

# change byLeader column to character
dt.swbd$byLeader = as.character(dt.swbd$byLeader)
dt.swbd[dt.swbd$byLeader == 'TRUE',]$byLeader = 'initiator'
dt.swbd[dt.swbd$byLeader == 'FALSE',]$byLeader = 'responder'
setnames(dt.swbd, 'byLeader', 'role')
dt.bnc$byLeader = as.character(dt.bnc$byLeader)
dt.bnc[dt.bnc$byLeader == 'TRUE',]$byLeader = 'initiator'
dt.bnc[dt.bnc$byLeader == 'FALSE',]$byLeader = 'responder'
setnames(dt.bnc, 'byLeader', 'role')

## plot
dt.swbd.tmp = dt.swbd[, .(ent, entc, wordNum, td, bf, inTopicId, role)][, corpus := 'SWBD'][, Group := '']
dt.bnc.tmp = dt.bnc[, .(ent, entc, wordNum, td, bf, inTopicId, role)][, corpus := 'BNC'][, Group := '']
dt.all = rbindlist(list(dt.swbd.tmp, dt.bnc.tmp))
dt.all[corpus == 'SWBD' & role == 'initiator', Group := 'SWBD: initiator']
dt.all[corpus == 'SWBD' & role == 'responder', Group := 'SWBD: responder']
dt.all[corpus == 'BNC' & role == 'initiator', Group := 'BNC: initiator']
dt.all[corpus == 'BNC' & role == 'responder', Group := 'BNC: responder']

p = ggplot(dt.all[inTopicId <= 10,], aes(x = inTopicId, y = ent)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = Group)) +
    scale_x_continuous(breaks = 1:10) +
    theme_light() + theme(legend.position = c(.75, .2)) +
    xlab('Relative sentence position within topic episode') + ylab('Sentence information (bits)') +
    scale_fill_manual(values = c('BNC: initiator' = my_colors[1], 'BNC: responder' = my_colors[1],
        'SWBD: initiator' = my_colors[2], 'SWBD: responder' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 3, 'SWBD: initiator' = 1, 'SWBD: responder' = 3)) +
    scale_shape_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 1, 'SWBD: initiator' = 4, 'SWBD: responder' = 4))
pdf('figs/si_vs_inPos_role.pdf', 4, 4)
plot(p)
dev.off()

p = ggplot(dt.all[inTopicId <= 10,], aes(x = inTopicId, y = entc)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = Group)) +
    scale_x_continuous(breaks = 1:10) +
    theme_light() + theme(legend.position = c(.75, .2)) +
    xlab('Relative sentence position within topic episode') + ylab('Normalized sentence information') +
    scale_fill_manual(values = c('BNC: initiator' = my_colors[1], 'BNC: responder' = my_colors[1],
        'SWBD: initiator' = my_colors[2], 'SWBD: responder' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 3, 'SWBD: initiator' = 1, 'SWBD: responder' = 3)) +
    scale_shape_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 1, 'SWBD: initiator' = 4, 'SWBD: responder' = 4))
pdf('figs/nsi_vs_inPos_role.pdf', 4, 4)
plot(p)
dev.off()



#######################
# Plot Fig.5
#######################
# Use the dt.all from last step
p.sl = ggplot(subset(dt.all, inTopicId <= 10), aes(x = inTopicId, y = wordNum)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = Group)) +
    scale_x_continuous(breaks = 1:10) +
    theme_light() + theme(legend.position = c(.8, .15)) +
    xlab('Relative sentence position') + ylab('Sentence length (number of words)') +
    guides(fill = guide_legend(title = 'Group'),
        lty = guide_legend(title = 'Group'),
        shape = guide_legend(title = 'Group')) +
    scale_fill_manual(values = c('BNC: initiator' = my_colors[1], 'BNC: responder' = my_colors[1],
        'SWBD: initiator' = my_colors[2], 'SWBD: responder' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 3, 'SWBD: initiator' = 1, 'SWBD: responder' = 3)) +
    scale_shape_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 1, 'SWBD: initiator' = 4, 'SWBD: responder' = 4))

p.td = ggplot(subset(dt.all, inTopicId <= 10), aes(x = inTopicId, y = td)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = Group)) +
    scale_x_continuous(breaks = 1:10) +
    theme_light() + theme(legend.position = c(.8, .15)) +
    xlab('Relative sentence position') + ylab('Tree depth') +
    guides(fill = guide_legend(title = 'Group'),
        lty = guide_legend(title = 'Group'),
        shape = guide_legend(title = 'Group')) +
    scale_fill_manual(values = c('BNC: initiator' = my_colors[1], 'BNC: responder' = my_colors[1],
        'SWBD: initiator' = my_colors[2], 'SWBD: responder' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 3, 'SWBD: initiator' = 1, 'SWBD: responder' = 3)) +
    scale_shape_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 1, 'SWBD: initiator' = 4, 'SWBD: responder' = 4))

p.bf = ggplot(subset(dt.all, inTopicId <= 10), aes(x = inTopicId, y = bf)) +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = .5, aes(fill = Group)) +
    stat_summary(fun.y = mean, geom = 'line', aes(lty = Group)) +
    stat_summary(fun.y = mean, geom = 'point', aes(shape = Group)) +
    scale_x_continuous(breaks = 1:10) +
    theme_light() + theme(legend.position = c(.8, .15)) +
    xlab('Relative sentence position') + ylab('Branching factor') +
    guides(fill = guide_legend(title = 'Group'),
        lty = guide_legend(title = 'Group'),
        shape = guide_legend(title = 'Group')) +
    scale_fill_manual(values = c('BNC: initiator' = my_colors[1], 'BNC: responder' = my_colors[1],
        'SWBD: initiator' = my_colors[2], 'SWBD: responder' = my_colors[2])) +
    scale_linetype_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 3, 'SWBD: initiator' = 1, 'SWBD: responder' = 3)) +
    scale_shape_manual(values = c('BNC: initiator' = 1, 'BNC: responder' = 1, 'SWBD: initiator' = 4, 'SWBD: responder' = 4))


library(gridExtra)
# the function that gets the legend of a plot (for multiple plotting that shares one legend)
g_legend = function(p) {
    tmp = ggplotGrob(p)
    leg = which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend = tmp$grobs[[leg]]
    legend
}

p.sl = p.sl + theme(legend.position = 'bottom') #plot.title = element_text(size = 12)
lgd = g_legend(p.sl)

pdf('figs/sltdbf.pdf', 10, 4)
grid.arrange(arrangeGrob(p.sl + theme(legend.position = 'none'),
                        p.td + theme(legend.position = 'none'),
                        p.bf + theme(legend.position = 'none'), ncol = 3),
            lgd, nrow = 2, heights = c(9, 1))
dev.off()







##
# What are mean topic length
dt.swbd = fread('data/SWBD_entropy_db.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 9.22
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 5.66
summary(dt.swbd[, .N, keyby=.(convId, topicId)]$N)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.000   5.000   8.000   9.222  11.000  49.000

dt.bnc = fread('data/BNC_entropy_db.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 10.9
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 7.3
summary(dt.bnc[, .N, by=.(convId, topicId)]$N)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.00    5.00   10.00   10.85   15.00   52.00

# other topic segmentation results
# dp (bayes-seg)
dt.swbd = fread('data/SWBD_text_db_dp.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 19.7
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 27.0

dt.bnc = fread('data/BNC_text_dbfull_mlrcut_dp.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 13.1
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 35.1

# mcsopt (mincut)
dt.swbd = fread('data/SWBD_text_db_mcsopt.csv')
mean(dt.swbd[, .N, keyby=.(convId, topicId)]$N) # 19.7
sd(dt.swbd[, .N, by=.(convId, topicId)]$N) # 26.8

dt.bnc = fread('data/BNC_text_dbfull_mlrcut_mcsopt.csv')
mean(dt.bnc[, .N, keyby=.(convId, topicId)]$N) # 13.1
sd(dt.bnc[, .N, by=.(convId, topicId)]$N) # 29.9
