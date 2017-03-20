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
