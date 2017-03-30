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
    config: one of ['dp.config', 'cue.config', 'mcsopt.ai.config', 'ui.config']
    """
    assert config in ['dp.config', 'cue.config', 'mcsopt.ai.config', 'ui.config']

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
        output = proc.stdout.readlines()
        last = output[-1].strip().decode('utf-8')
        res = ast.literal_eval(last)
    except Exception as e:
        print('Last line of output: \n{}'.format(last))
        print('All output: \n')
        for line in output:
            print(line)
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
    config: one of ['dp.config', 'cue.config', 'mcsopt.ai.confi', 'ui.config']
    """
    assert config in ['dp.config', 'cue.config', 'mcsopt.ai.config', 'ui.config']

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
            print('df_tmp.shape[0] == {}'.format(df_tmp.shape[0]))
            print('res:\n{}'.format(res))
            print('ids:\n{}'.format(ids))
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
# Pseudo segmentation using fixed length
def pseudo_seg_fixedlen(inputfile, outputfile, seglen=10):
    # read textdata into a pandas dataframe
    df = pd.read_csv(inputfile)
    # BTW, examine if `wodNum` column exists in df
    # if not, create it from `rawWord` column
    if 'rawWord' not in df.columns:
        raise Exception('rawWord column not existed in inputfile')
    if 'wordNum' not in df.columns:
        df['wordNum'] = df.rawWord.apply(lambda x: len(x.split()))

    # handle special cases (dialogues that are too short) for BNC
    if inputfile == 'data/BNC_text_dbfull_mlrcut.csv':
        df_ref = pd.read_csv('data/BNC_convs_gt10sents.csv')
        included_cids = df_ref.convId.unique()
        df = df[df.convId.isin(included_cids)]
        # remove NaNs in rawWord
        df = df[df.rawWord.notnull()]

    # For all convIds, assign pseudo segment Ids
    df_ids = pd.DataFrame()
    discarded_convIds = []
    for i, cid in enumerate(df.convId.unique()):
        nrow = df[df.convId == cid].shape[0]
        # discared conversations that are too short
        if nrow < seglen:
            discarded_convIds.append(cid)
            continue
        # create pseudo ids
        pseudo_ids = make_pseudo_ids(nrow, seglen)
        df_tmp = pd.DataFrame(pseudo_ids)
        # combine
        df_ids = pd.concat([df_ids, df_tmp], axis=0)
        # print progress
        sys.stdout.write('\r{0}/{1} convIds segmented'.format(i+1, len(df.convId.unique())))
        sys.stdout.flush()

    # combine df and df_ids
    if len(discarded_convIds) > 0:
        df = df[~df.convId.isin(discarded_convIds)]
    df1 = pd.concat([df.reset_index(drop=True), df_ids.reset_index(drop=True)], axis=1) # reset_index is necessary
    df1.to_csv(outputfile, sep=',', index=False)


##
# the func that creates pseudo seg_ids and in_seg_ids columns
def make_pseudo_ids(n, seglen):
    assert n >= seglen
    seg_ids = []
    in_seg_ids = []
    nseg = n // seglen
    nres = n % seglen
    for j in range(nseg):
        seg_ids += [j+1] * seglen
        in_seg_ids += list(range(1, seglen+1))
    if nres > 0:
        seg_ids += [nseg+1] * nres
        in_seg_ids += list(range(1, nres+1))
    return {'topicId': seg_ids, 'inTopicId': in_seg_ids}



##
# main
if __name__ == '__main__':
    # test1()
    # test2()

    # segment SWBD, using dp.config
    # seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_dp.csv', config='dp.config')

    # segment BNC, using dp.config
    # seg_textdata(inputfile='data/BNC_text_dbfull_mlrcut.csv', outputfile='data/BNC_text_dbfull_mlrcut_dp.csv', config='dp.config')
    # elapse 12 min

    # segment SWBD, using mcsopt.ai.config
    # seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_mcsopt.csv', config='mcsopt.ai.config')

    # segment BNC, using mcsopt.ai.config
    # seg_textdata(inputfile='data/BNC_text_dbfull_mlrcut.csv', outputfile='data/BNC_text_dbfull_mlrcut_mcsopt.csv', config='mcsopt.ai.config')
    # elapse 7:25.81 total

    # segment SWBD using cue.config, DO NOT work
    # seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_cue.csv', config='cue.config')
    # segment SWBD using ui.config, DO NOT work
    # seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_ui.csv', config='ui.config')

    # assign pseudo ids (fixed length) to SWBD
    # pseudo_seg_fixedlen(inputfile='data/SWBD_text_db.csv', outputfile='data/SWBD_text_db_pseudofixed.csv', seglen=10)

    # assign pseudo ids (fixed length) to BNC
    pseudo_seg_fixedlen(inputfile='data/BNC_text_dbfull_mlrcut.csv', outputfile='data/BNC_text_dbfull_mlrcut_pseudofixed.csv', seglen=10)
