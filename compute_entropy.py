# NOTE: Must use Python2.7 provided by macOS system, i.e., /usr/bin/python
# Compute per-word entropy for the SWBD_ / BNC_ text_db.csv files
# Yang Xu
# 3/1/2017

from __future__ import print_function

import sys
import subprocess
import csv
import math

from random import shuffle
from srilm import *



##
# compute the entropy of Switchboard by 10-fold cross-validation
def crossvalidate(inputfile, outputfile):
    # read all text data
    alldata = {}
    with open(inputfile, 'r') as fr:
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

    # the function that read text from alldata for a list of convIds
    def readtext(data, conv_ids):
        text = []
        for cid in conv_ids:
            for row in data[cid]:
                if row[1] != '':
                    text.append(row[1])
        return text

    # conduct cross-validation
    entropy_results = []
    for i in range(0, foldN):
        traintext = []
        for j in range(0, i) + range(i+1, foldN):
            traintext += readtext(alldata, foldIds[j])
        # write traintext to file
        trainfile = 'data/lm/train_fold%s.txt' % i
        with open(trainfile, 'w') as fw:
            for row in traintext:
                fw.write(row + '\n')
        # train the lm
        lmfile = 'data/lm/lm_fold%s.lm' % i
        srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
        train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
        return_code = subprocess.check_call(train_cmd)
        if return_code != 0:
            raise Exception('trainning failed for fold %s' % i)
        print('training done for fold %s' % i)
        # compute entropy
        lm = initLM(3)
        readLM(lm, lmfile)
        for cid in foldIds[i]:
            for row in alldata[cid]:
                gid, text = row[0], row[1]
                try:
                    ppl = getSentencePpl(lm, text, len(text.split()))
                except Exception as e:
                    print('convId: %s' % cid)
                    print('globalId: %s' % gid)
                    raise
                else:
                    ent = math.log(ppl, 10)
                    entropy_results.append((cid, gid, ent))
        print('computing done for fold %s' % i)

    # write entropy_results to file
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in entropy_results:
            csvwriter.writerow(row)


##
# compute the entropy using LM trained from an external file
def externalLM(testfile, trainfile, outputfile):
    # read text from trainfile, and write to a temporary file
    with open(trainfile, 'r') as fr, open('data/lm/train.txt', 'w') as fw:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            text = items[4]
            fw.write(text + '\n')
    # train the LM
    lmfile = 'data/lm/train.lm'
    srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
    train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', 'data/lm/train.txt', '-lm', lmfile]
    return_code = subprocess.check_call(train_cmd)
    if return_code != 0:
        raise Exception('trainning failed')

    # read text from testfile and compute entropy
    lm = initLM(3)
    readLM(lm, lmfile)
    entropy_results = []
    with open(testfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            try:
                ppl = getSentencePpl(lm, text, len(text.split()))
            except Exception as e:
                print('convId: %s' % cid)
                print('globalId: %s' % gid)
                raise
            else:
                ent = math.log(ppl, 10)
                entropy_results.append((cid, gid, ent))
    # write entropy_results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in entropy_results:
            csvwriter.writerow(row)


##
# compute entropy using 10-fold cross-validation
# Training text is the set of sentences from the same position (globalId) as test text
def crossvalidate_samepos(inputfile, outputfile, sentence_n=100):
    # read all text data
    alldata = {}
    with open(inputfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            if gid <= sentence_n:
                if cid in alldata:
                    alldata[cid][gid] = text
                else:
                    alldata[cid] = {gid : text}

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

    # the function that read text from alldata
    # for a list of convIds, and a given globalId
    def readtext(data, conv_ids, gid):
        text = []
        for cid in conv_ids:
            if gid in data[cid]:
                text.append(data[cid][gid])
        return text

    # conduct cross-validation
    
    pass


##
# main
if __name__ == '__main__':
    # SWBD_crossvalidate(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_entropy_crossvalidate.csv')

    # Cut the `speakerOriginal` column in BNC_text_db100.csv
    # $ mlr --csv cut -f convId,turnId,speaker,globalId,rawWord BNC_text_db100.csv > BNC_text_db100_mlrcut.csv
    # crossvalidate(inputfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/BNC_entropy_crossvalidate.csv')

    # compute Switchboard using LM trained from BNC
    # externalLM(testfile='data/SWBD_text_db.csv', trainfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/SWBD_entropy_fromBNC.csv')

    # compute BNC using LM trained from Switchboard
    # externalLM(testfile='data/BNC_text_db100_mlrcut.csv', trainfile='data/SWBD_text_db.csv', outputfile='data/BNC_entropy_fromSWBD.csv')

    # using LM trained from Penn Treebank WSJ corpus
