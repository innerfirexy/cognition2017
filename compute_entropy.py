# NOTE: Must use Python2.7 provided by macOS system, i.e., /usr/bin/python
# Compute per-word entropy for the SWBD_ / BNC_ text_db.csv files
# Yang Xu
# 3/1/2017

import sys
from random import shuffle
from srilm import *
from __future__ import print_function


##
# compute the entropy of Switchboard by 10-fold cross-validation
def SWBD_crossvalidate():
    # read all text data
    alldata = {}
    with open('SWBD_text_db.csv', 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            if cid in alldata:
                alldata[cid].append((gid, text))
            else:
                alldata[cid] = [(gid, text)]

    # prepare folds
    convIds = alldata.keys()
    shuffle(convIds)
    foldN = 10
    foldLen = len(convIds) / foldN
    foldIds = {}
    for i in range(0, foldN):
        if i < foldN-1:
            foldIds[i] = convIds[i*foldLen : (i+1)*foldLen]
        else:
            foldIds[i] = convIds[i*foldLen:]

    # defind the function that read text from alldata for a list of convIds
    def readtext(data, conv_ids):
        text = []
        for cid in conv_ids:
            for row in data[cid]:
                if row[1] != '':
                    text.append(row[1])
        return text

    # conduct cross-validation
    
    pass

##
# compute the entropy of Switchboard by using the model trained from BNC
def SWBD_fromBNC():
    pass

##
# main
if __name__ == '__main__':
