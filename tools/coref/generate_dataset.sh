#!/usr/bin/env bash


. tools/coref/setup.sh

python3 tools/generate_dataset.py -input ${data_path} -o ${out_path} -transformer_props resources/transformers.json RuCor $@
