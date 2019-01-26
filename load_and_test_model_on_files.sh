#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [ "$#" -ne 5 ]; then 	# number of args
    echo "USAGE: <ftrain> <ftest> <model> <seed> <outfile>"
    echo "$ME"
    exit
fi
ftrain=$1
ftest=$2
model=$3
seed=$4
out=$5
time python -m seq2seq.main \
     --ftrain ${ftrain} \
     --ftest ${ftest} \
     --mono \
     --beam_width 1 \
     --restore ${model} \
     --seed ${seed} \
     --dump ${out}





if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
