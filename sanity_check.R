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
