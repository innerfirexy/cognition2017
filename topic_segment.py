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
# test
def test():
    inputlist = []
    with open('/Users/yangxu/GitHub/bayes-seg/data/books/clinical/000.ref', 'r') as f:
        for line in f:
            inputlist.append(line.strip())
    print(conduct_segment(inputlist))


##
# main
if __name__ == '__main__':
    test()
