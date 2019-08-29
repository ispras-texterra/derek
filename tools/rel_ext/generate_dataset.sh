#!/usr/bin/env bash

if [[ $# -lt 2 ]]
then
    echo "Usage: <bb|seed|chemprot> <segmenter args>"
    exit
fi

. tools/rel_ext/setup_variables.sh $1

echo generating $1 dataset...
shift
python3 tools/generate_dataset.py -name train -input ${in_train_path} -name dev -input ${in_dev_path} \
    -name test -input ${in_test_path} -o ${out_path} -transformer_props resources/transformers.json \
    ${reader} $@ && echo success
