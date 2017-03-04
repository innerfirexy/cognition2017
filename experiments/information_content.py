#!/usr/bin/python
# NOTE: use macOS default python2
# Try alternative methods rather than Ngram models to estimate the information content of sentences
# Yang Xu
# 3/3/2017

from __future__ import print_function
from nltk.probability import FreqDist
from random import shuffle
import math
import numpy as np
import sys
import csv
import itertools

##
# Estimate information content using negative log probability of unigram
# as used by Priva, 2016 (Not so fast ...)
# The probability of unigram is estimated by the observed frequency
# with add-1 smooth for zero counts
def unigram_freq(inputfile, outputfile):
    # read all data from inputfile
    alldata = {}
    with open(inputfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            if cid in alldata:
                if text != '':
                    alldata[cid].append((gid, text))
            else:
                if text != '':
                    alldata[cid] = [(gid, text)]

    # the function that returns all unigrams for a list of convIds
    def get_unigrams(data, conv_ids):
        words = []
        for cid in conv_ids:
            for item in data[cid]:
                words.append(item[1].split())
        return itertools.chain.from_iterable(words)

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

    # estimate information content using cross-validation
    results = []
    for i in range(0, foldN):
        train_cids = []
        for j in range(0, i) + range(i+1, foldN):
            train_cids += foldIds[j]
        fdist = FreqDist(get_unigrams(alldata, train_cids))
        # compute negative log probability
        for cid_idx, cid in enumerate(foldIds[i]):
            for item in alldata[cid]:
                gid, text = item[0], item[1]
                neglogprob = []
                for w in text.split():
                    if w in fdist:
                        p = float(fdist[w]+1) / (fdist.N()+1)
                        neglogprob.append(-math.log(p))
                    else:
                        p = 1 / float(fdist.N()+1)
                        neglogprob.append(-math.log(p))
                meanval = float(sum(neglogprob)) / len(neglogprob)
                results.append((cid, gid, meanval))
            # print
            sys.stdout.write('\r%s/%s of fold %s done' % (cid_idx, len(foldIds[i]), i+1))
            sys.stdout.flush()
        print('fold %s done.' % (i+1))

    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'infoCont'])
        for row in results:
            csvwriter.writerow(row)


##
# main
if __name__ == '__main__':
    unigram_freq(inputfile='../data/SWBD_text_db.csv', outputfile='../data/SWBD_infocont_unifreq.csv')
