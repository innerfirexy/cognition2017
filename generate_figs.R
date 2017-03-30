# Generate the figures in Cognition 2017 paper
# Author: Yang Xu
# Time: 2/22/2017

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)


##
# Plot Figure 1
dt.swbd = fread('data/SWBD_entropy_db.csv')

# models
m = lmer(ent ~ globalId + (1|convId), dt.swbd)
summary(m)
# globalId    4.225e-03  4.888e-04 1.036e+05   8.643   <2e-16 ***
# NOTE: entropy increases within dialogue
m = lmer(wordNum ~ globalId + (1|convId), dt.swbd)
summary(m)
# globalId    -8.455e-03  1.262e-03  1.036e+05  -6.698 2.12e-11 ***
# NOTE: wordNum decreases with globalId



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
