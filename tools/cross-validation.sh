#!/usr/bin/env bash

if [[ $# -lt 4 ]]
then
    echo "Usage: <resources directory name> <n folds> <ner|net|rel_ext> <n seeds> {conllu-file}"
    exit
fi

. tools/common/setup_variables.sh

input_path=resources/$1_generated
out_path=out

set -e
mkdir -p ${out_path}

unlabeled=
if [[ $# -gt 4 ]]
then
    unlabeled="-unlabeled $5"
fi

current_time=$(date +%Y-%m-%d--%H-%M-%S)

python3 -u tools/param_search.py -task $3 -props resources/prop.json -lst resources/lst.json \
    -seeds $4 -out ${out_path} ${unlabeled} \
    cross_validation -traindev ${input_path} -folds $2 \
    2>> err--${current_time}.log | tee out--${current_time}.log
