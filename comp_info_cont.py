#!/usr/bin/python
# NOTE: Must use Python2.7 provided by macOS system, i.e., /usr/bin/python
# Compute the informaiton content using variant methods
# Yang Xu
# 3/8/2017

from __future__ import print_function

import sys
import subprocess
import csv
import math
import os

from random import shuffle
from nltk.util import ngrams
from srilm import *


# the function that reads text into a dict object
# key is convId, and value is a list
def readtext_2list(datafile, cid_col=0, gid_col=3, text_col=4):
    """
    return: dict(convId -> [(globalId, sentence_text)])
    """
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

# reads text into a dict
# key is convId, and value is a dict
# the number of sentences per convId is limited by sent_n
def readtext_2dict(datafile, cid_col=0, gid_col=3, text_col=4, sent_n=100):
    """
    sent_n: the maximum number of sentences read from per convId
    return: dict(convId -> {globalId -> sentence_text})
    """
    data = {}
    with open(datafile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[cid_col]), int(items[gid_col]), items[text_col]
            if gid <= sent_n:
                if cid in data:
                    data[cid][gid] = text
                else:
                    data[cid] = {gid : text}
    return data


# the function that gets all words from data for a list of convIds
def get_words_fromlist(data, conv_ids):
    words = []
    for cid in conv_ids:
        for item in data[cid]:
            words.append(item[1].split())
    return itertools.chain.from_iterable(words)

# the function that get all sentences from data
def get_sents_fromlist(data, conv_ids):
    sents = []
    for cid in conv_ids:
        for item in data[cid]:
            sents.append(item[1])
    return sents

# the function that gets words from dict, at a certain sentence position: gid
def get_sents_fromdict(data, conv_ids, gid):
    sents = []
    for cid in conv_ids:
        if gid in data[cid]:
            sents.append(data[cid][gid])
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
# Compute the information content of sentence using cross-validation
def crossvalidate(inputfile, outputfile):
    # read data
    alldata = readtext_2list(inputfile)

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
        with open(trainfile, 'w') as fw:
            for row in traintext:
                fw.write(row + '\n')
        # train the lm
        lmfile = 'data/lm/train.lm'
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
                ent = sentence_entropy(lm, text)
                results.append((cid, gid, ent))
        print('computing done for fold %s' % i)

    # write results to file
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in results:
            csvwriter.writerow(row)


##
# Compute the information content of sentence using cross-validation
# LMs are trained per sentence position, i.e., 100 models trained for the first 100 sentences respectively
def crossvalidate_samepos(inputfile, outputfile, sent_n=100):
    # read data
    alldata = readtext_2dict(inputfile, sent_n=sent_n)

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
    entropy_results = []
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
            with open(trainfile, 'w') as fw:
                for text in traintext:
                    fw.write(text + '\n')
            # train the LM
            lmfile = 'data/lm/train.lm'
            srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
            train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', trainfile, '-lm', lmfile]
            FNULL = open(os.devnull, 'w') # suppress stdout and stderr
            return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
            if return_code != 0:
                raise Exception('trainning failed for fold %s at sentence position %s' % (i, j))
            # compute sentence entropy
            lm = initLM(3)
            readLM(lm, lmfile)
            for cid in foldIds[i]:
                if j in alldata[cid]:
                    gid = j
                    text = alldata[cid][j]
                    ent = sentence_entropy(lm, text)
                    results.append((cid, j, ent))
            # print process within a fold
            sys.stdout.write('\rfold %s, %s/%s sentences done' % (i, j, sent_n))
            sys.stdout.flush()
        # print when a fold is done
        print('\nDone for fold %s' % i)

    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in results:
            csvwriter.writerow(row)


##
# compute the entropy using LM trained from an external file
def externalTrain(testfile, trainfile, outputfile):
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
    FNULL = open(os.devnull, 'w') # suppress stdout and stderr
    return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if return_code != 0:
        raise Exception('trainning failed')

    # read text from testfile and compute entropy
    lm = initLM(3)
    readLM(lm, lmfile)
    results = []
    with open(testfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            ent = sentence_entropy(lm, text)
            results.append((cid, gid, ent))

    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in results:
            csvwriter.writerow(row)


##
# Compute infomation content using only unigrams existed in the training vocabulary
def externalTrain_invocab(testfile, trainfile, outputfile):
    # read text from trainfile, and insert frequency into a dict
    wordsdict = {}
    with open(trainfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            for w in items[4].split():
                if w in wordsdict:
                    wordsdict[w] += 1
                else:
                    wordsdict[w] = 1
    # get total frequency
    wordsN = sum(val for val in wordsdict.itervalues())
    # compute
    results = []
    with open(testfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            probs = []
            for w in text.split():
                if w in wordsdict:
                    p = float(wordsdict[w]) / wordsN
                    probs.append(-math.log(p))
            if len(probs) > 0:
                ent = sum(probs) / len(probs)
                results.append((cid, gid, ent, float(len(probs))/len(text.split())))
            else:
                results.append((cid, gid, 'NA', 0))
    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent', 'inVocabProp'])
        for row in results:
            csvwriter.writerow(row)


##
# Train LM using external sentences of same position
def externalTrain_samepos(testfile, trainfile, outputfile):
    # read text from trainfile into a dict
    # and key is sentence position, and value is text
    traintext = {}
    with open(trainfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            gid, text = int(items[3]), items[4]
            if gid in traintext:
                traintext[gid].append(text)
            else:
                traintext[gid] = [text]

    # read text from testfile into a dict
    # where key is sentence position, and value is a dict {cid -> text}
    testtext = {}
    with open(testfile, 'r') as fr:
        fr.next()
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            if gid in testtext:
                testtext[gid][cid] = text
            else:
                testtext[gid] = {cid: text}

    # train LM and compute entropy
    results = []
    for gid in range(1, 101):
        # train the LM
        tmpfile = 'data/lm/train.txt'
        with open(tmpfile, 'w') as fw:
            for text in traintext[gid]:
                fw.write(text + '\n')
        lmfile = 'data/lm/train.lm'
        srilm_dir = '/Users/yangxu/projects/srilm-1.7.1/bin/macosx/'
        train_cmd = [srilm_dir + 'ngram-count', '-order', '3', '-text', tmpfile, '-lm', lmfile]
        FNULL = open(os.devnull, 'w') # suppress stdout and stderr
        return_code = subprocess.check_call(train_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
        if return_code != 0:
            raise Exception('trainning failed')
        # compute
        lm = initLM(3)
        readLM(lm, lmfile)
        for cid, text in testtext[gid].iteritems():
            ent = sentence_entropy(lm, text)
            results.append((cid, gid, ent))
        # print progress
        sys.stdout.write('\r%s/%s sentence positions done' % (gid, 100))
        sys.stdout.flush()

    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in results:
            csvwriter.writerow(row)
    # print
    print('\nDone for %s' % testfile)



##
# Compute entropy using already trained LM
def externalLM(testfile, lmfile, outputfile):
    # load the LM, read text from testfile, and compute entropy
    lm = initLM(3)
    readLM(lm, lmfile)
    results = []
    with open(testfile, 'r') as fr:
        fr.next()
        for line in fr:
            items = line.strip().split(',')
            cid, gid, text = int(items[0]), int(items[3]), items[4]
            ent = sentence_entropy(lm, text)
            results.append((cid, gid, ent))
    # write results to outputfile
    with open(outputfile, 'w') as fw:
        csvwriter = csv.writer(fw, delimiter=',')
        csvwriter.writerow(['convId', 'globalId', 'ent'])
        for row in results:
            csvwriter.writerow(row)


##
# main
if __name__ == '__main__':
    # cross-validation
    # crossvalidate(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_entropy_crossvalidate.csv')
    # crossvalidate(inputfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/BNC_entropy_crossvalidate.csv')

    # cross-validation by same sentence position
    # crossvalidate_samepos(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_entropy_crossvalidate_samepos.csv')
    # crossvalidate_samepos(inputfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/BNC_entropy_crossvalidate_samepos.csv')

    # compute Switchboard using LM trained from BNC, and compute BNC using LM trained from Switchboard
    # externalTrain(testfile='data/SWBD_text_db.csv', trainfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/SWBD_entropy_fromBNC.csv')
    # externalTrain(testfile='data/BNC_text_db100_mlrcut.csv', trainfile='data/SWBD_text_db.csv', outputfile='data/BNC_entropy_fromSWBD.csv')

    # compute Switchboard using LM trained from BNC, and compute BNC using LM trained from Switchboard
    # using invocab unigrams only
    # externalTrain_invocab(testfile='data/SWBD_text_db.csv', trainfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/SWBD_entropy_fromBNC_invocab.csv')
    # externalTrain_invocab(testfile='data/BNC_text_db100_mlrcut.csv', trainfile='data/SWBD_text_db.csv', outputfile='data/BNC_entropy_fromSWBD_invocab.csv')

    # compute Switchboard using LM trained from BNC, and compute BNC using LM trained from Switchboard
    # using sentences from same position
    # externalTrain_samepos(testfile='data/SWBD_text_db.csv', trainfile='data/BNC_text_db100_mlrcut.csv', outputfile='data/SWBD_entropy_fromBNC_samepos.csv')
    # externalTrain_samepos(testfile='data/BNC_text_db100_mlrcut.csv', trainfile='data/SWBD_text_db.csv', outputfile='data/BNC_entropy_fromSWBD_samepos.csv')

    # using LM trained from WSJ corpus
    # externalLM(testfile='data/SWBD_text_db.csv', lmfile='data/lm/wsj_gt10_text.lm', outputfile='data/SWBD_entropy_fromWSJ.csv')
    # externalLM(testfile='data/BNC_text_db100_mlrcut.csv', lmfile='data/lm/wsj_gt10_text.lm', outputfile='data/BNC_entropy_fromWSJ.csv')

    # Using external LM trained from WSJ, using sentences from same position
    externalTrain_samepos(testfile='data/SWBD_text_db.csv', trainfile='data/lm/wsj_gt10_full_addblank.csv', outputfile='data/SWBD_entropy_fromWSJ_samepos.csv')
    externalTrain_samepos(testfile='data/BNC_text_db100_mlrcut.csv', trainfile='data/lm/wsj_gt10_full_addblank.csv', outputfile='data/BNC_entropy_fromWSJ_samepos.csv')
