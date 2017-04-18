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
import re

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

def model_oov(lm_file, test_file, lm_order=3, verbose=False):
    # run command
    srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
    test_cmd = [srilm_dir + 'ngram', '-lm', lm_file, '-ppl', test_file]
    try:
        output = subprocess.check_output(test_cmd, stderr=subprocess.STDOUT)
        if verbose:
            print(output)
        m = re.search(r'[0-9]+(?= OOVs)', output)
        oovn = int(m.group(0))
    except Exception as e:
        raise
    else:
        return oovn

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
        # compute perplexity and OOVs number
        lm = initLM(3)
        readLM(lm, lmfile)
        ppl = getCorpusPpl(lm, testfile)
        oovn = model_oov(lm_file=lmfile, test_file=testfile)
        results.append((ppl, oovn))

    # write results to output_file
    with open(output_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['ppl', 'oovn'])
        for row in results:
            csvwriter.writerow(row)

##
# Get the perplexity of running cross-validation over sentences of same positions
# The output file has three columns: fold_id, sentence_id, perplexity
def cv_samepos_ppl(input_file, output_file, sent_n=100):
    # read data
    alldata = readtext_2dict(input_file, sent_n=sent_n)

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

    # cross-validation
    results = []
    for i in range(0, foldN):
        # for each sentence position
        for j in range(1, sent_n+1):
            # collect all sentences at position j in other convIds than foldIds[i]
            traintext = []
            for k in range(0, i) + range(i+1, foldN):
                text = get_sents_fromdict(alldata, foldIds[k], j)
                traintext += text
            # write traintext to file
            trainfile = 'data/lm/train.txt'
            with open(trainfile, 'w') as f:
                for text in traintext:
                    f.write(text + '\n')
            # train the LM
            lmfile = 'data/lm/train.lm'
            srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
            train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
            FNULL = open(os.devnull, 'w') # suppress stdout and stderr
            return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
            if return_code != 0:
                raise Exception('trainning failed for fold %s at sentence position %s' % (i, j))
            # compute perplexity and oovn
            testtext = get_sents_fromdict(alldata, foldIds[i], j)
            testfile = 'data/lm/test.txt'
            with open(testfile, 'w') as f:
                for text in testtext:
                    f.write(text + '\n')
            lm = initLM(3)
            readLM(lm, lmfile)
            ppl = getCorpusPpl(lm, testfile)
            oovn = model_oov(lm_file=lmfile, test_file=testfile)
            results.append((i, j, ppl, oovn))
            # print process within a fold
            sys.stdout.write('\rfold %s, %s/%s sentences done' % (i, j, sent_n))
            sys.stdout.flush()
        print('fold %s done' % i)

    # write results to outputfile
    with open(output_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['foldId', 'sentenceId', 'ppl', 'oovn'])
        for row in results:
            csvwriter.writerow(row)


##
# main
if __name__ == '__main__':
    # LM trained on BNC, tested on SWBD
    # print('LM trained on BNC, tested on SWBD:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/BNC_full_order3.lm', test_file='data/lm/SWBD.txt')))
    # Perplexity: 179.795227051
    # print('OOVs number: {}'.format(model_oov(lm_file='data/lm/BNC_full_order3.lm', test_file='data/lm/SWBD.txt')))
    # OOVN: 141724

    # LM trained on SWBD, tested on BNC:
    # print('LM trained on SWBD, tested on BNC:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/SWBD_order3.lm', test_file='data/lm/BNC_full.txt')))
    # Perplexity: 266.232452393
    # print('OOVs number: {}'.format(model_oov(lm_file='data/lm/SWBD_order3.lm', test_file='data/lm/BNC_full.txt')))
    # OOVN: 74075

    # SWBD cross-validation perplexity
    # cv_ppl(data_file='data/SWBD_text_db.csv', output_file='data/lm/SWBD_cv_ppl.txt')
    # Perplexity, mean = 77.38471, sd = 1.898807
    # OOVN: sum = 13009, mean = 1300.9, sd = 85.93079
    # BNC cross-validation perplexity
    # cv_ppl(data_file='data/BNC_text_dbfull_mlrcut.csv', output_file='data/lm/BNC_cv_ppl.txt')
    # Perplexity, mean = 107.4193, sd = 15.49336
    # OOVN: 14417, mean = 1441.7, sd = 605.2599

    # LM trained on CSN, tested on SWBD
    # print('LM trained on CSN, tested on SWBD:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/SWBD.txt')))
    # Perplexity: 244.400680542
    # print('OOVN: {}'.format(model_oov(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/SWBD.txt')))
    # OOVN: 108360

    # LM trained on CSN, tested on BNC
    # print('LM trained on CSN, tested on BNC:')
    # print('Perplexity: {}'.format(model_ppl(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/BNC_full.txt')))
    # Perplexity: 266.107971191
    # print('OOVN: {}'.format(model_oov(lm_file='data/lm/CSN_order3.lm', test_file='data/lm/BNC_full.txt')))
    # OOVN: 7129

    # SWBD cv samepos ppl
    cv_samepos_ppl(input_file='data/SWBD_text_db.csv', output_file='data/lm/SWBD_cv_samepos_ppl.txt')
    # Perplexity: mean = 88.9816, sd = 9.595562
    # OOVN: 114950
    # BNC cv samepos ppl
    cv_samepos_ppl(input_file='data/BNC_text_dbfull_mlrcut.csv', output_file='data/lm/BNC_cv_samepos_ppl.txt')
    # Perplexity: mean = 84.92237, sd = 15.58643
    # OOVN: 55438
