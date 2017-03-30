# Combine the entropy results output by compute_entropy.py
# with the data that contain topic structure information, e.g., SWBD_entropy_db.csv etc.
# Yang Xu
# 3/2/2017

library(data.table)

# the function that combines two data sets
combineTopic = function(d1, d2) {
    # check columns
    stopifnot(all.equal(c('convId', 'globalId', 'ent') %in% colnames(d1), rep(T, 3)))
    stopifnot(all.equal(c('convId', 'globalId', 'speaker', 'wordNum', 'topicId', 'inTopicId') %in% colnames(d2), rep(T, 6)))

    # combine
    setkey(d1, convId, globalId)
    setkey(d2, convId, globalId)
    d.comb = d1[, .(convId, globalId, ent)][d2[, .(convId, globalId, speaker, wordNum, topicId, inTopicId)],]
    d.comb
}

# combine Switchboard entropy trained by cross-validation with topic columns
d1 = fread('data/SWBD_entropy_crossvalidate.csv')
d2 = fread('data/SWBD_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_topic.csv')

# combine BNC entropy trained by cross-validation with topic columns
d1 = fread('data/BNC_entropy_crossvalidate.csv')
d2 = fread('data/BNC_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_topic.csv')

# combine SWBD entropy from cross-validation samepos with TextTiling topic columns
d1 = fread('data/SWBD_entropy_crossvalidate_samepos.csv')
d2 = fread('data/SWBD_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_samepos_topic.csv')

# combine SWBD entropy from cross-validation samepos with TextTiling topic columns
d1 = fread('data/BNC_entropy_crossvalidate_samepos.csv')
d2 = fread('data/BNC_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_samepos_topic.csv')


# combine Switchboard entropy trained from BNC with topic columns
d1 = fread('data/SWBD_entropy_fromBNC.csv')
d2 = fread('data/SWBD_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_fromBNC_topic.csv')

# combine BNC entropy trained from Switchboard with topic columns
d1 = fread('data/BNC_entropy_fromSWBD.csv')
d2 = fread('data/BNC_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_fromSWBD_topic.csv')


##
# combine SWBD entropy db with topic ids from dp.config
d1 = fread('data/SWBD_entropy_db.csv')
d2 = fread('data/SWBD_text_db_dp.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_db_dp.csv')

##
# combine SWBD entropy cross-validation with topic ids from dp.config
d1 = fread('data/SWBD_entropy_crossvalidate.csv')
d2 = fread('data/SWBD_text_db_dp.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_dp.csv')

##
# combine SWBD entropy cross-validation samepos with topic ids from dp.config
d1 = fread('data/SWBD_entropy_crossvalidate_samepos.csv')
d2 = fread('data/SWBD_text_db_dp.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_samepos_dp.csv')

##
# combine BNC entropy cross-validation with topic ids from dp.config
d1 = fread('data/BNC_entropy_crossvalidate.csv')
d2 = fread('data/BNC_text_dbfull_mlrcut_dp.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_dp.csv')

##
# combine BNC entropy cross-validation samepos with topic ids from dp.config
d1 = fread('data/BNC_entropy_crossvalidate_samepos.csv')
d2 = fread('data/BNC_text_dbfull_mlrcut_dp.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_samepos_dp.csv')

##
# combine SWBD entropy cross-validation samepos with topic ids from mcsopt.ai.config
d1 = fread('data/SWBD_entropy_crossvalidate_samepos.csv')
d2 = fread('data/SWBD_text_db_mcsopt.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_samepos_mcsopt.csv')

##
# combine BNC entropy cross-validation samepos with topic ids from mcsopt.ai.config
d1 = fread('data/BNC_entropy_crossvalidate_samepos.csv')
d2 = fread('data/BNC_text_dbfull_mlrcut_mcsopt.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_samepos_mcsopt.csv')


##
# combine BNC entropy cross-validation samepos with topic ids from texttiling in db
d1 = fread('data/BNC_entropy_crossvalidate_samepos.csv')
d2 = fread('data/BNC_entropy_db.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_samepos_tt.csv')


##
# combine SWBD entropy cross-validation samepos with pseudo ids (fixed seglen)
d1 = fread('data/SWBD_entropy_crossvalidate_samepos.csv')
d2 = fread('data/SWBD_text_db_pseudofixed.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/SWBD_entropy_crossvalidate_samepos_pseudofixed.csv')

##
# combine BNC entropy cross-validation samepos with pseudo ids (fixed seglen)
d1 = fread('data/BNC_entropy_crossvalidate_samepos.csv')
d2 = fread('data/BNC_text_dbfull_mlrcut_pseudofixed.csv')
d = combineTopic(d1, d2)
fwrite(d, 'data/BNC_entropy_crossvalidate_samepos_pseudofixed.csv')
