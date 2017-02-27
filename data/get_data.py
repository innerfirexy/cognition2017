# Get data from database
# Yang Xu
# 2/22/2017

import MySQLdb
import csv

# get db connection
def db_conn(db_name):
    # db init: ssh yvx5085@brain.ist.psu.edu -i ~/.ssh/id_rsa -L 1234:localhost:3306
    conn = MySQLdb.connect(host = "127.0.0.1",
                    user = "yang",
                    port = 1234,
                    passwd = "05012014",
                    db = db_name)
    return conn

##
# select Switchboard entropy data
def select_SWBD_entropy():
    conn = db_conn('swbd')
    cur = conn.cursor()
    sql = 'select convID, turnID, speaker, globalID, ent, wordNum, tileID, inTileID from entropy where ent IS NOT NULL'
    cur.execute(sql)
    data = cur.fetchall()
    with open('SWBD_entropy_db.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'turnId', 'speaker', 'globalId', 'ent', 'wordNum', 'topicId', 'inTopicId'])
        for row in data:
            csvwriter.writerow(row)
    conn.close()

##
# select Switchboard text from db
def select_SWBD_text():
    conn = db_conn('swbd')
    cur = conn.cursor()
    sql = 'select convID, turnID, speaker, globalID, rawWord from entropy where rawWord <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('SWBD_text_db.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'turnId', 'speaker', 'globalId', 'rawWord'])
        for row in data:
            csvwriter.writerow(row)
    conn.close()

##
# Select BNC entropy data
# The `entropy_DEM100` table contains 2398 rows where `topicID` and `inTopicID` columns are NULL
# which are perhaps due to the TextTiling segmentation
def select_BNC_entropy():
    conn = db_conn('bnc')
    cur = conn.cursor()
    sql = 'select convID, speaker, speakerOriginal, globalID, turnID, ent, wordNum, topicID, inTopicID from entropy_DEM100 \
        where topicID IS NOT NULL AND inTopicID IS NOT NULL'
    cur.execute(sql)
    data = cur.fetchall()
    with open('BNC_entropy_db.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'speaker', 'speakerOriginal', 'globalId', 'turnId', 'ent', 'wordNum', 'topicId', 'inTopicId'])
        for row in data:
            csvwriter.writerow(row)
    conn.close()

##
# Select BNC text data from db
# The `entropy_DEM100` table only contains the first 100 sentences from 1043 conversations
# The conversations that have less than 100 sentences are excluded
def select_BNC_text100():
    conn = db_conn('bnc')
    cur = conn.cursor()
    sql = 'select convID, turnID, speaker, speakerOriginal, globalID, rawWord from entropy_DEM100 \
        where rawWord <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('BNC_text_db100.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'turnId', 'speaker', 'speakerOriginal', 'globalId', 'rawWord'])
        for row in data:
            csvwriter.writerow(row)
    conn.close()

##
# Select BNC full text data from db
# The `entropy_DEM_full` table contains all sentences from the 1346 conversations
def select_BNC_textfull():
    conn = db_conn('bnc')
    cur = conn.cursor()
    sql = 'select convId, speakerOriginal, globalId, turnId, localId, strLower, wordNum, episodeId, inEpisodeId \
        from entropy_DEM_full where strLower <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('BNC_text_dbfull.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'speakerOriginal', 'globalId', 'turnId', 'localId', 'rawWord', 'wordNum', \
            'topicId', 'inTopicId'])
        for row in data:
            csvwriter.writerow(row)
    conn.close()



##
# main
if __name__ == '__main__':
    # select_SWBD_entropy()
    # select_SWBD_text()
    # select_BNC_entropy()
    # select_BNC_text100()
    select_BNC_textfull()
