#!/usr/bin/env bash

if [ $# -ne 3 ]
then
    echo "Usage: <seed-file> <fold_num> <folds>"
    exit
fi


. tools/coref/setup.sh


export SAMPLES_CACHE_PATH=${out_path}/$2_fold/data.mp
mkdir ${out_path}/$2_fold
for seed in $(cat $1); do
    echo ${seed}
    mkdir ${out_path}/$2_fold/${seed}
    python3 -u tools/coref/param_search.py ${data_path} ${base_prop_path} ${lst_path} $2 $3 ${out_path}/$2_fold/${seed} ${seed} 2>> ${out_path}/$2_fold/${seed}/err.log | tee ${out_path}/$2_fold/${seed}/${seed}.log
done

