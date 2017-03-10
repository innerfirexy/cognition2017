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
# globalId    5.510e-04  1.586e-04 7.118e+04   3.475 0.000511 ***
# NOTE: YES! consistent with the samepos results!
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)

## BNC data trained by cross-validation
dt = fread('data/BNC_entropy_crossvalidate.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    2.461e-03  9.029e-04 3.849e+04   2.726  0.00642 **
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)



## Switchboard data trained from BNC
dt = fread('data/SWBD_entropy_fromBNC.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -5.733e-03  4.817e-04  8.894e+04   -11.9   <2e-16 ***
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


## BNC data trained from Switchboard
dt = fread('data/BNC_entropy_fromSWBD.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    2.818e-03  1.827e-03 2.436e+04   1.542    0.123
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)



## Swtichboard trained from BNC, invocab unigrams only
dt = fread('data/SWBD_entropy_fromBNC_invocab.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -6.678e-04  6.251e-05  6.095e+04  -10.68   <2e-16 ***
nrow(dt[is.na(ent),])/nrow(dt)
# only 10% sentences cannot be calculated
mean(dt$inVocabProp)
# 82.9%

## BNC trained from Switchboard, invocab unigrams only
dt = fread('data/BNC_entropy_fromSWBD_invocab.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -9.322e-04  2.718e-04  2.598e+04   -3.43 0.000605 ***
nrow(dt[is.na(ent),])/nrow(dt)
# 0.04661983, less than 5%
mean(dt$inVocabProp)
# 88.2%


## Swtichboard trained from BNC, using sentence from same positions
dt = fread('data/SWBD_entropy_fromBNC_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    2.383e-02  1.830e-03 1.036e+05   13.02   <2e-16 ***

## BNC trained from Switchboard, using sentence from same positions
dt = fread('data/BNC_entropy_fromSWBD_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    9.522e-03  2.858e-03 2.999e+04   3.332 0.000864 ***
# NOTE: this is what we want



## Switchboard data trained from LMs of same position
dt = fread('data/SWBD_entropy_crossvalidate_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    1.265e-02  1.232e-03 1.036e+05   10.27   <2e-16 ***
# NOTE: consistent with nltk results!!
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


## BNC data trained from LMs of same position
dt = fread('data/BNC_entropy_crossvalidate_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    6.802e-02  2.377e-03 4.378e+04   28.61   <2e-16 ***
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


## SWBD trained from WSJ
dt = fread('data/SWBD_entropy_fromWSJ.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -7.569e-03  5.535e-04  9.171e+04  -13.68   <2e-16 ***

## BNC trained from WSJ
dt = fread('data/BNC_entropy_fromWSJ.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    1.586e-03  1.601e-03 2.840e+04   0.991    0.322
# n.s.


## SWBD and BNC trained from WSJ, with sentences of same position
dt = fread('data/SWBD_entropy_fromWSJ_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId      -0.18219    0.05183 9910.00000  -3.515 0.000441 ***
p = ggplot(dt, aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)

dt = fread('data/BNC_entropy_fromWSJ_samepos.csv')
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -3.096e-01  4.525e-02  1.132e+04  -6.843 8.18e-12 ***



## WSJ trained by cross-validation, using SRILM
dt = fread('data/wsj_entropy.csv')
m = lmer(ent ~ sentId + (1|fileId), dt)
summary(m)
# sentId      7.940e-03  1.215e-03 1.118e+04   6.537 6.55e-11 ***
# Yes, it means that SRILM can correctly replicate the entropy increase reported by Genzel et al 2002 & 2003
p = ggplot(dt, aes(x=sentId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)



## SWBD trained by cross-validation, but with NLTK code and SRILM method
dt = fread('data/results_swbd_srilm_CV.txt')
setnames(dt, c('convId', 'globalId', 'localId', 'ent'))
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -1.335e-04  7.006e-05  1.035e+05  -1.905   0.0568 .
# OKay, although we use the same cross-validation code that generates the nltk NgramModel data
# The current data generated by SRILM seem not consistent with it
# Needs to further look into how SRILM computes entropy differs from nltk NgramModel
# Maybe, starting from simpler estimation (e.g., unigram) of informaiton content is a good choice!


## SWBD infocont by unigram frequency
# NOTE: unigram information content decrease at first and then do not change
dt = fread('data/SWBD_infocont_unifreq.csv')
m = lmer(infoCont ~ globalId + (1|convId), dt)
summary(m)
# globalId    -6.718e-05  5.659e-05  8.945e+04  -1.187    0.235
m = lmer(infoCont ~ globalId + (1|convId), dt[globalId<=100,])
summary(m)
# globalId    -1.179e-03  1.324e-04  1.036e+05  -8.908   <2e-16 ***
m = lmer(infoCont ~ globalId + (1|convId), dt[globalId<=100 & globalId>=25,])
summary(m)
# n.s.
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=infoCont)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)


##
# NOTE: Entropy does increase with sentence position
# when we compute sentence entropy correctly.
## SWBD infocont by unigram probability, estimated by SRILM
dt = fread('data/SWBD_infocont_unisrilm.csv')
m = lmer(infoCont ~ globalId + (1|convId), dt)
summary(m)
# globalId    9.060e-04  2.079e-04 5.396e+04   4.359 1.31e-05 ***
m = lmer(infoCont ~ globalId + (1|convId), dt[globalId<=100,])
summary(m)
# globalId    1.194e-03  4.862e-04 1.035e+05   2.455   0.0141 *
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=infoCont)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)

# ppl ~ globalId
m = lmer(ppl ~ globalId + (1|convId), dt)
summary(m)
# globalId    9.829e-02  4.192e-02 2.098e+04   2.345    0.019 *
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ppl)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)

# ent ~ globalId
m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    4.904e-04  1.569e-04 7.026e+04   3.125  0.00178 **
p = ggplot(dt[globalId<=100,], aes(x=globalId, y=ent)) +
    stat_summary(fun.y = mean, geom = 'line') +
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha=.5)




########
# Check old entropy results, computed from nltk NgramModel
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

## Old train by bin results
dt = fread('data/results_swbd_nltk_trainByBin.txt')
setnames(dt, c('globalId', 'localId', 'ent'))
summary(lm(ent ~ globalId, dt))
# globalId    0.001531   0.001429   1.071    0.284
