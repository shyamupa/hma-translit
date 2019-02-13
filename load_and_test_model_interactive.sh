#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [[ "$#" -ne 2 ]]; then 	# number of args
    echo "USAGE: ${ME} <vocabfile> <model>"
    echo 
    exit
fi
vocabfile=$1
model=$2
time python -m seq2seq.predict \
     --vocabfile ${vocabfile} \
     --mono \
     --beam_width 1 \
     --restore ${model} \
     --interactive

if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
