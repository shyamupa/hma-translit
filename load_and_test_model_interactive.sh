#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [ "$#" -ne 3 ]; then 	# number of args
    echo "USAGE: ${ME} <ftrain> <model> <seed>"
    echo 
    exit
fi
ftrain=$1
model=$2
seed=$3
time python -m seq2seq.main \
     --ftrain ${ftrain} \
     --mono \
     --beam_width 1 \
     --restore ${model} \
     --interactive \
     --seed ${seed}

if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
