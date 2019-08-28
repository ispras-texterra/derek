#!/usr/bin/env bash

if [[ $# -lt 4 ]]
then
    echo "Usage: <bb|seed|chemprot|...> <train-on-dev|dev-on-train> <ner|net|rel_ext> <n seeds> {conllu-file}"
    exit
fi

. tools/common/setup_variables.sh

input_path=resources/$1_generated

strategy=$2
if [[ ${strategy} == "dev-on-train" ]]
then
    train_path=${input_path}/dev
    dev_path=${input_path}/train
else
    train_path=${input_path}/train
    dev_path=${input_path}/dev
    strategy=train-on-dev
fi

set -e
echo ${strategy}

out_path=out/${strategy}
mkdir -p ${out_path}

unlabeled=
if [[ $# -gt 4 ]]
then
    unlabeled="-unlabeled $5"
fi

logs_path=logs
mkdir -p ${logs_path}

current_time=$(date +%Y-%m-%d--%H-%M-%S)

python3 -u tools/param_search.py -task $3 -props resources/prop.json -lst resources/lst.json \
    -seeds $4 -out ${out_path} ${unlabeled} \
    holdout -train ${train_path} -dev ${dev_path} \
    2>${logs_path}/err--${current_time}.log | tee ${logs_path}/out--${current_time}.log
