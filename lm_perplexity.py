#!/usr/bin/python
# NOTE: Must use Python2.7 provided by macOS system, i.e., /usr/bin/python

# Evaluate the language models used to compute sentence entropy
# Yang Xu
# 3/20/2017

from __future__ import print_function

import sys
import subprocess
import csv
import math
import os

from random import shuffle
from srilm import *

from comp_info_cont import readtext_2list, readtext_2dict, get_sents_fromlist, get_sents_fromdict


##
# Compute the perplexity of a LM (model_file) on a testing set (test_file)
def model_ppl(lm_file, test_file, lm_order=3):
    # load lm
    lm = initLM(3)
    readLM(lm, lm_file)
    # compute perplexity
    try:
        ppl = getCorpusPpl(lm, test_file)
    except Exception as e:
        raise
    else:
        deleteLM(lm)
        return ppl

##
# get the perplexity of cross-validation
# return 10 values
def cv_ppl(data_file, output_file):
    # read data
    alldata = readtext_2list(data_file)

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

    # conduct cross-validation
    results = []
    for i in range(0, foldN):
        traintext = []
        for j in range(0, i) + range(i+1, foldN):
            traintext += get_sents_fromlist(alldata, foldIds[j])
        # write traintext to file
        trainfile = 'data/lm/train.txt'
        with open(trainfile, 'w') as f:
            for row in traintext:
                f.write(row + '\n')
        # write testtext to file
        testtext = get_sents_fromlist(alldata, foldIds[i])
        testfile = 'data/lm/test.txt'
        with open(testfile, 'w') as f:
            for row in testtext:
                f.write(row + '\n')
        # train the lm
        lmfile = 'data/lm/train.lm'
        srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
        train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
        FNULL = open(os.devnull, 'w') # suppress stdout and stderr
        return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
        if return_code != 0:
            raise Exception('trainning failed for fold %s' % i)
        print('training done for fold %s' % i)
        # compute perplexity
        lm = initLM(3)
        readLM(lm, lmfile)
        ppl = getCorpusPpl(lm, testfile)
        results.append(ppl)

    # write results to output_file
    with open(output_file, 'w') as f:
        for row in results:
            f.write(str(row) + '\n')


##
# main
if __name__ == '__main__':
    # LM trained on BNC, tested on SWBD
    # print('LM trained on BNC, tested on SWBD:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/BNC_full_order3.lm', test_file='data/lm/SWBD.txt')))
    # Perplexity: 179.795227051

    # LM trained on SWBD, tested on BNC:
    # print('LM trained on SWBD, tested on BNC:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/SWBD_order3.lm', test_file='data/lm/BNC_full.txt')))
    # Perplexity: 266.232452393

    # SWBD cross-validation perplexity
    # cv_ppl(data_file='data/SWBD_text_db.csv', output_file='data/lm/SWBD_cv_ppl.txt')
    # mean = 77.38471, sd = 1.898807
    # BNC cross-validation perplexity
    # cv_ppl(data_file='data/BNC_text_dbfull_mlrcut.csv', output_file='data/lm/BNC_cv_ppl.txt')
    # mean = 107.4193, sd = 15.49336

    # LM trained on CSN, tested on SWBD
    # print('LM trained on CSN, tested on SWBD:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/SWBD.txt')))
    # Perplexity: 244.400680542

    # LM trained on CSN, tested on BNC
    # print('LM trained on CSN, tested on BNC:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/BNC_full.txt')))
    # Perplexity: 266.107971191
