#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [[ "$#" -ne 5 ]]; then 	# number of args
    echo "USAGE: ${ME} <ftrain> <ftest> <seed> <vocabfile> <aligned_file>"
    exit
fi
ftrain=$1
ftest=$2
seed=$3
vocabfile=$4
aligned_file=$5

time python -m seq2seq.prepare_data \
     --ftrain ${ftrain} \
     --ftest ${ftest} \
     --vocabfile ${vocabfile} \
     --aligned_file ${aligned_file} \
     --seed ${seed}





if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
