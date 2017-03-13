#!/bin/bash

text_file=$1 # first argument is the full path to the text file
config_file="config/$2"

cd /Users/yangxu/GitHub/bayes-seg/

cat $text_file | ./segment $config_file
