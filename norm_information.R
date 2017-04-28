# Compute the normalized entropy
# Yang Xu
# 4/28/2017

library(data.table)

##
# the func that computes the normalized sentence information
normEnt = function(d) {
    stopifnot(all(c('ent', 'wordNum') %in% colnames(d)))
    d.tmp = copy(d)
    d.tmp[, ent_mean := mean(ent), keyby = wordNum] # add the column for the mean entropy of sentences of the same length
    d.tmp[, ent_norm := ent / ent_mean]
    d.tmp
}


# For original computed sentence information
dt.swbd = fread('data/SWBD_entropy_db.csv')
dt.swbd.norm = normEnt(dt.swbd)
fwrite(dt.swbd.norm, 'data/SWBD_entropy_db_norm.csv')

dt.bnc = fread('data/BNC_entropy_db.csv')
dt.bnc.norm = normEnt(dt.bnc)
fwrite(dt.bnc.norm, 'data/BNC_entropy_db_norm.csv')
