#!/usr/bin/env bash

if [[ $# -ne 1 ]]
then
    echo "Usage: <bb|seed|chemprot>"
    exit
fi

. tools/common/setup_variables.sh


if [[ $1 == 'bb' ]]
then
    in_dev_path=resources/bb3/BioNLP-ST-2016_BB-event_dev
    in_train_path=resources/bb3/BioNLP-ST-2016_BB-event_train
    in_test_path=resources/bb3/BioNLP-ST-2016_BB-event_test
    out_path=resources/bb_generated
    reader=BioNLP
elif [[ $1 == 'seed' ]]
then
    in_dev_path=resources/bb3/BioNLP-ST-2016_SeeDev-binary_dev
    in_train_path=resources/bb3/BioNLP-ST-2016_SeeDev-binary_train
    in_test_path=resources/bb3/BioNLP-ST-2016_SeeDev-binary_test
    out_path=resources/seed_generated
    reader=BioNLP
elif [[ $1 == 'chemprot' ]]
then
    in_dev_path=resources/ChemProt_Corpus/chemprot_development
    in_train_path=resources/ChemProt_Corpus/chemprot_training
    in_test_path=resources/ChemProt_Corpus/chemprot_test
    out_path=resources/chemprot_generated
    reader=ChemProt
fi
