# Quick check of how entropy of sentences with dysfluencies removed change
# Yang Xu
# 3/3/2017

library(RMySQL)
library(data.table)
library(lme4)
library(lmerTest)

# ssh yvx5085@brain.ist.psu.edu -i ~/.ssh/id_rsa -L 1234:localhost:3306
conn = dbConnect(MySQL(), host = '127.0.0.1', user = 'yang', port = 1234, password = "05012014", dbname = 'swbd')
sql = 'SELECT convID, globalID, ent FROM entropy_disf WHERE ent IS NOT NULL'
df = dbGetQuery(conn, sql)

dt = data.table(df)
setnames(dt, c('convId', 'globalId', 'ent'))

m = lmer(ent ~ globalId + (1|convId), dt)
summary(m)
# globalId    -4.833e-04  7.939e-05  2.600e+04   -6.087 1.16e-09 ***
##
# NOTE: WOW~ when dysfluencies are removed, sentence entropy still drops in dialogue
## corresponding to Doyle's work on Twitter entropy. 
