# Sanity check for the results from compute_entropy.py
# Yang Xu
# 3/2/2017

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)


## Switchboard data trained by cross-validation
dt = fread('data/SWBD_entropy_crossvalidate.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -4.283e-05  3.129e-05  8.168e+04  -1.369    0.171
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) -5.825e-03  1.778e-03  1.295e+05  -3.276  0.00105 **


## BNC data trained by cross-validation
dt = fread('data/BNC_entropy_crossvalidate.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -4.952e-05  1.109e-04  4.158e+04  -0.447    0.655
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) -1.047e-02  2.791e-03  3.570e+04  -3.752 0.000176 ***


## Switchboard data trained from BNC
dt = fread('data/SWBD_entropy_fromBNC.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -3.650e-04  3.272e-05  6.264e+04  -11.16   <2e-16 ***
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) -3.225e-02  1.867e-03  1.172e+05  -17.27   <2e-16 ***

# plot
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


## BNC data trained from Switchboard
dt = fread('data/BNC_entropy_fromSWBD.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -2.878e-04  1.273e-04  3.706e+04  -2.261   0.0238 *
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) -2.088e-02  3.194e-03  3.195e+04  -6.536 6.41e-11 ***


## Switchboard data trained from LMs of same position
dt = fread('data/SWBD_entropy_crossvalidate_samepos.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -4.283e-05  3.129e-05  8.168e+04  -1.369    0.171
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) 3.967e-03  2.124e-03 1.036e+05   1.868   0.0618 .

p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


## BNC data trained from LMs of same position
dt = fread('data/BNC_entropy_crossvalidate_samepos.csv')

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -2.040e-04  8.937e-05  3.546e+04  -2.283   0.0225 *
m = lmer(ent ~ log(globalId) + (1|convId), dt)
summary(m)
# log(globalId) -9.336e-03  2.242e-03  3.095e+04  -4.165 3.13e-05 ***

p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


##
# Check old entropy results
dt = fread('data/results_swbd_nltk_CV.csv')
setnames(dt, c('convId', 'globalId', 'localId', 'ent'))
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    4.225e-03  4.888e-04 1.036e+05   8.643   <2e-16 ***

dt = fread('data/results_BNC-DEM_CV_fullInfo_copy_new.csv')
setkey(dt, xmlId, divId)
dt[, convId := .GRP, by = .(xmlId, divId)]
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    1.514e-02  8.835e-04 3.547e+04   17.14   <2e-16 ***
