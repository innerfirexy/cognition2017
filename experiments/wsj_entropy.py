# Examine if SRILM can correctly replicate the entropy increase found in WSJ corpus
# NOTE: See ../sanity_check.R for results
# Yang Xu
# 3/3/2017

from __future__ import print_function

import sys
sys.path.append('..')

import subprocess
import csv
import math
import os

from random import shuffle
from srilm import *


##
# compute entropy by 10-fold cross-validation
def crossvalidate(inputfile, outputfile):
    # read all text data
    alldata = {}
    with open(inputfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = items[0], int(items[1]), items[2]
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
        trainfile = '../data/lm/train_fold%s.txt' % i
        with open(trainfile, 'w') as fw:
            for row in traintext:
                fw.write(row + '\n')
        # train the lm
        lmfile = '../data/lm/lm_fold%s.lm' % i
        srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
        train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
        FNULL = open(os.devnull, 'w') # suppress stdout and stderr
        return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
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
                    print('fileId: %s' % cid)
                    print('sentId: %s' % gid)
                    raise
                else:
                    ent = math.log(ppl, 10)
                    entropy_results.append((cid, gid, ent))
        print('computing done for fold %s' % i)

    # write entropy_results to file
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['fileId', 'sentId', 'ent'])
        for row in entropy_results:
            csvwriter.writerow(row)


##
# main
if __name__ == '__main__':
    crossvalidate(inputfile='../data/lm/wsj_gt10_full.csv', outputfile='../data/wsj_entropy.csv')
