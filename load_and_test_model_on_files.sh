#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [[ "$#" -ne 4 ]]; then 	# number of args
    echo "USAGE: <vocabfile> <model> <ftest> <outfile>"
    echo "$ME"
    exit
fi
vocabfile=$1
model=$2
ftest=$3
outfile=$4
time python -m seq2seq.predict \
     --vocabfile ${vocabfile} \
     --ftest ${ftest} \
     --mono \
     --beam_width 1 \
     --restore ${model} \
     --dump ${outfile}





if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
