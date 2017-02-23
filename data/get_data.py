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
# select entropy data
def select_entropy():
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

##
# main
if __name__ == '__main__':
    select_entropy()
