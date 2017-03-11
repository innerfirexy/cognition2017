#!/usr/local/bin/python3
# Get data from database
# Yang Xu
# 2/22/2017

import MySQLdb
import csv
import sys
import string

# get db connection
def db_conn(db_name):
    # db init: ssh yvx5085@brain.ist.psu.edu -i ~/.ssh/id_rsa -L 1234:localhost:3306
    conn = MySQLdb.connect(host = "127.0.0.1",
                    user = "yang",
                    port = 3306,
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
            text = row[-1].lower()
            newrow = list(row[:-1])
            newrow.append(text)
            csvwriter.writerow(newrow)
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
            text = row[-4].lower()
            newrow = list(row[:-4])
            newrow.append(text)
            newrow += list(row[-3:])
            csvwriter.writerow(newrow)
    conn.close()


##
# Get text from CSN db as training corpus
def select_CSN():
    conn = db_conn('csn')
    cur = conn.cursor()
    sql = 'select tokenized from csntokenizednew where tokenized <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('CSN_text.txt', 'w') as f:
        for i, row in enumerate(data):
            text = row[0]
            # remove double quotes
            text = text[1:-1]
            # split
            tokens = text.split('^^~^^')
            # traverse tokens and form sentences by detecting sentence-ending symbols
            # and omit punctuations in the middle of a sentence
            end_puncts = ['.', '?', '!']
            other_puncts = [p for p in list(string.punctuation) if p not in end_puncts]
            sent = []
            for j, t in enumerate(tokens):
                if j == len(tokens)-1:
                    if t in other_puncts:
                        if len(sent) > 0:
                            f.write(' '.join(sent) + '\n')
                    elif t in end_puncts:
                        if len(sent) > 0:
                            f.write(' '.join(sent) + '\n')
                    else:
                        sent.append(t.lower())
                        f.write(' '.join(sent) + '\n')
                else:
                    if t in other_puncts:
                        continue
                    elif t in end_puncts:
                        if len(sent) > 0:
                            f.write(' '.join(sent) + '\n')
                            sent = []
                    else:
                        sent.append(t.lower())
            # print process
            if (i+1) % 1000 == 0:
                sys.stdout.write('\r{}/{} rows written'.format(i+1, len(data)))
                sys.stdout.flush()
    conn.close()


##
# select BNC written text data from db
def select_BNC_written():
    conn = db_conn('bnc')
    cur = conn.cursor()
    sql = 'select rawWord from entropy_nsp where rawWord <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('BNC_writtentext_db.txt', 'w') as f:
        for row in data:
            text = row[0]
            f.write(text.lower() + '\n')
    conn.close()

##
# select disfluencies-removed Switchboard text from db
def select_SWBD_disf():
    conn = db_conn('bnc')
    cur = conn.cursor()
    sql = 'select convID, turnID, speaker, globalID, rawWord from entropy_disf where rawWord <> \"\"'
    cur.execute(sql)
    data = cur.fetchall()
    with open('SWBD_text_disfrmvd.csv', 'w', newline='') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'turnId', 'speaker', 'globalId', 'rawWord'])
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
    # select_BNC_textfull()
    # select_CSN()
    # select_BNC_written()
    select_SWBD_disf()
