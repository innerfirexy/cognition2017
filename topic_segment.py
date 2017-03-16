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
def make_topic_ids(sent_num, bound_ind):
    """
    sent_len: the number of sentences as the input of conduct_segment
    bound_ind: the sentence indice of topic boundaries, returned by conduct_segment
    return: a dict of two lists, {'topicId': [1,1,1,...], 'inTopicId': [1,2,3,...]}
    """
    ids1 = [] # topicId
    ids2 = [] # inTopicId
    for i, ind in enumerate(bound_ind):
        length = ind+1 if i == 0 else ind - bound_ind[i-1]
        ids1 += [i+1] * length
        ids2 += list(range(1, length+1))
    if sent_num > bound_ind[-1] + 1:
        ids1 += [ids1[-1]+1] * (sent_num - bound_ind[-1] - 1)
        ids2 += list(range(1, sent_num - bound_ind[-1]))
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

    # call conduct_segment for each convId
    for i, cid in enumerate(df.convId.unique()):
        sentlist = list(df[df.convId == cid].rawWord)
        res = conduct_segment(inputlist=sentlist, config=config)
        # print progress
        sys.stdout.write('\r{0}/{1} convIds segmented'.format(i+1, len(df.convId.unique())))
        sys.stdout.flush()



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
    sNum = 25
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
    seg_textdata(inputfile='data/SWBD_text_db.csv', outputfile='foo', config='dp.config')
