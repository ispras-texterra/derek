#!/usr/bin/env bash

if [[ $# -lt 2 ]]
then
    echo "Usage: <resources directory name> <reader and/or segmenter args>"
    exit
fi

. tools/common/setup_variables.sh

input_path=resources/$1
output_path=resources/$1_generated
reader=BRAT

shift
python3 tools/generate_dataset.py -input ${input_path} -o ${output_path} -transformer_props resources/transformers.json ${reader} $@  && echo success
