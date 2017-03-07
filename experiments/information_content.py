#!/usr/bin/python
# NOTE: use macOS default python2
# Try alternative methods rather than Ngram models to estimate the information content of sentences
# Yang Xu
# 3/3/2017

from __future__ import print_function
from nltk.probability import FreqDist
from nltk.util import ngrams
from random import shuffle

import math
import numpy as np
import sys
import os
import csv
import itertools
import subprocess

sys.path.append('..')
from srilm import *


# the function that reads text data
def read_text_data(datafile, cid_col=0, gid_col=3, text_col=4):
    data = {}
    with open(datafile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[cid_col]), int(items[gid_col]), items[text_col]
            if cid in data:
                if text != '':
                    data[cid].append((gid, text))
            else:
                if text != '':
                    data[cid] = [(gid, text)]
    return data

# the function that get all unigrams from data for a list of convIds
def get_unigrams(data, conv_ids):
    words = []
    for cid in conv_ids:
        for item in data[cid]:
            words.append(item[1].split())
    return itertools.chain.from_iterable(words)

# the function that get all sentences from data
def get_sentences(data, conv_ids):
    sents = []
    for cid in conv_ids:
        for item in data[cid]:
            sents.append(item[1])
    return sents

# compute sentence entropy
# adding <s> to the left, and </s> to the right
def sentence_entropy(lm, text):
    words = text.split()
    trigrams = list(ngrams(words, 3, pad_left=True, left_pad_symbol='<s>', pad_right=True, right_pad_symbol='</s>'))
    trigrams = trigrams[1:-1]
    probs = [-getTrigramProb(lm, ' '.join(gram)) for gram in trigrams]
    ent = float(sum(probs)) / len(probs)
    return ent



##
# Estimate information content using negative log probability of unigram
# as used by Priva, 2016 (Not so fast ...)
# The probability of unigram is estimated by the observed frequency
# with add-1 smooth for zero counts
def unigram_freq(inputfile, outputfile):
    # read data
    alldata = read_text_data(inputfile)

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
# Estimate information content using negative log probability of unigram
# and the probability is estimated by the unigram language model trained via SRILM
def unigram_srilm(inputfile, outputfile):
    # read all data from inputfile
    alldata = read_text_data(inputfile)

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
        train_sents = get_sentences(alldata, train_cids)
        # write train_sents to trainfile
        trainfile = '../data/lm/train.txt'
        with open(trainfile, 'w') as fw:
            for s in train_sents:
                fw.write(s + '\n')
        # train the LM
        lmfile = '../data/lm/train.lm'
        srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
        train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
        FNULL = open(os.devnull, 'w') # suppress stdout and stderr
        return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
        if return_code != 0:
            raise Exception('trainning failed')
        # compute mean information content for each sentence in test set
        lm = initLM(3)
        readLM(lm, lmfile)
        for k, cid in enumerate(foldIds[i]):
            for item in alldata[cid]:
                gid, text = item[0], item[1]
                neglogprob = []
                for w in text.split():
                    neglogprob.append(-getUnigramProb(lm, w))
                meanval = float(sum(neglogprob)) / len(neglogprob)
                ppl = getSentencePpl(lm, text, len(text.split()))
                ent = sentence_entropy(lm, text)
                results.append((cid, gid, meanval, ppl, ent))
            # print
            sys.stdout.write('\r%s/%s of fold %s done' % (k+1, len(foldIds[i]), i+1))
            sys.stdout.flush()
        print('fold %s done.' % (i+1))

    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'infoCont', 'ppl', 'ent'])
        for row in results:
            csvwriter.writerow(row)


##
# main
if __name__ == '__main__':
    # unigram_freq(inputfile='../data/SWBD_text_db.csv', outputfile='../data/SWBD_infocont_unifreq.csv')

    unigram_srilm(inputfile='../data/SWBD_text_db.csv', outputfile='../data/SWBD_infocont_unisrilm.csv')
