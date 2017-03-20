#!/usr/local/bin/python3
# Conduct segmentation on text data
# Yang Xu
# 3/12/2017

import subprocess
import sys
import ast
import os
import pandas as pd
import numpy as np

##
# take a list of str as input, and output the sentence index of segment boundaries
def conduct_segment(inputlist, config = 'dp.config'):
    # NOTE: we found that the result returned by bayes-seg is 1-indexed
    # sentence positions
    """
    inputlist: a list of str
    config: one of ['dp.config', 'cue.config', 'mcsopt.ai.confi']
    """
    assert config in ['dp.config', 'cue.config', 'mcsopt.ai.config']

    # save inputlist to a temporary file
    tmp_file = 'data/tmp/text_to_seg.txt'
    with open(tmp_file, 'w') as f:
        for row in inputlist:
            f.write(row + '\n')
    tmp_file_full = os.path.join(os.getcwd(), tmp_file)

    # run segment script
    cmd = ['./conduct_segment.sh', tmp_file_full, config]
    proc = subprocess.Popen(cmd, stdout = subprocess.PIPE)

    try:
        last = proc.stdout.readlines()[-1].strip().decode('utf-8')
        res = ast.literal_eval(last)
    except Exception as e:
        raise
    else:
        return res


##
# make topicId and inTopicId from the result of conduct_segment
def make_topic_ids(bound_ind):
    """
    bound_ind: the sentence indice of topic boundaries, returned by conduct_segment
    return: a dict of two lists, {'topicId': [1,1,1,...], 'inTopicId': [1,2,3,...]}
    """
    ids1 = [] # topicId
    ids2 = [] # inTopicId
    for i, ind in enumerate(bound_ind):
        length = ind if i == 0 else ind - bound_ind[i-1]
        ids1 += [i+1] * length
        ids2 += list(range(1, length+1))
    return {'topicId': ids1, 'inTopicId': ids2}


##
# segment text data file: 'data/SWBD_text_db.csv', 'data/BNC_text_db100.csv', & 'data/BNC_text_dbfull.csv'
def seg_textdata(inputfile, outputfile, config = 'dp.config'):
    """
    config: one of ['dp.config', 'cue.config', 'mcsopt.ai.confi']
    """
    assert config in ['dp.config', 'cue.config', 'mcsopt.ai.config']

    # read textdata into a pandas dataframe
    df = pd.read_csv(inputfile)
    # BTW, examine if `wodNum` column exists in df
    # if not, create it from `rawWord` column
    if 'rawWord' not in df.columns:
        raise Exception('rawWord column not existed in inputfile')
    if 'wordNum' not in df.columns:
        df['wordNum'] = df.rawWord.apply(lambda x: len(x.split()))

    ##
    # handle special cases (dialogues that are too short) for BNC
    if inputfile == 'data/BNC_text_dbfull_mlrcut.csv':
        df_ref = pd.read_csv('data/BNC_convs_gt10sents.csv')
        included_cids = df_ref.convId.unique()
        df = df[df.convId.isin(included_cids)]
        # remove NaNs in rawWord
        df = df[df.rawWord.notnull()]


    # call conduct_segment for each convId
    df_ids = pd.DataFrame()
    for i, cid in enumerate(df.convId.unique()):
        sentlist = list(df[df.convId == cid].rawWord)
        try:
            res = conduct_segment(inputlist=sentlist, config=config)
        except Exception as e:
            print('problematic convId: {}'.format(cid))
            print('length of sentlist: {}'.format(len(sentlist)))
            raise
        ids = make_topic_ids(res)
        df_tmp = pd.DataFrame(ids)
        # DEBUG code
        if df_tmp.shape[0] != df[df.convId == cid].shape[0]:
            raise Exception('inconsistent length')
        # end DEBUG
        # combine
        df_ids = pd.concat([df_ids, df_tmp], axis=0)
        # print progress
        sys.stdout.write('\r{0}/{1} convIds segmented'.format(i+1, len(df.convId.unique())))
        sys.stdout.flush()

    # save df_ids temporarily
    tmpfile = inputfile[:-4] + '_ids.csv'
    df_ids.to_csv(tmpfile, sep=',', index=False)
    # combine df and df_ids
    df1 = pd.concat([df.reset_index(drop=True), df_ids.reset_index(drop=True)], axis=1) # reset_index is necessary
    df1.to_csv(outputfile, sep=',', index=False)


##
# test conduct_segment
def test1():
    inputlist = []
    with open('/Users/yangxu/GitHub/bayes-seg/data/books/clinical/000.ref', 'r') as f:
        for line in f:
            inputlist.append(line.strip())
    print(conduct_segment(inputlist))

##
# test make_topic_ids
def test2():
    bInds = [3,7,11,20]
    # bInds = [0]
    sNum = 30
    res = make_topic_ids(sNum, bInds)
    print(bInds)
    print(res)
    print('length of topicId: {0}'.format(len(res['topicId'])))
    print('length of inTopicId: {0}'.format(len(res['inTopicId'])))


##
# main
if __name__ == '__main__':
    # test1()
    # test2()

    # segment SWBD
    # seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_dp.csv', config='dp.config')

    # segment BNC of full length
    # eplase 12 min
    seg_textdata(inputfile='data/BNC_text_dbfull_mlrcut.csv', outputfile='data/BNC_text_dbfull_mlrcut_dp.csv', config='dp.config')
