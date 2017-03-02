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
