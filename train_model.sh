#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [ "$#" -ne 3 ]; then 	# number of args
    echo "USAGE: <lang> <seed> <model_path>"
    echo "$ME"
    exit
fi
lang=$1
seed=$2
model=$3
time python -m seq2seq.main \
     --lang ${lang} \
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
