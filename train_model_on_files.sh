#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [ "$#" -ne 4 ]; then 	# number of args
    echo "USAGE: ${ME} <ftrain> <ftest> <seed> <model_path>"
    exit
fi
ftrain=$1
ftest=$2
seed=$3
model=$4

time python -m seq2seq.main \
     --ftrain ${ftrain} \
     --ftest ${ftest} \
     --mono \
     --beam_width 1 \
     --save ${model} \
     --seed ${seed}





if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
