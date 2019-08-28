#!/usr/bin/env bash
set -e
export PYTHONPATH=$PWD

python3 -m unittest discover -s tests/datamodel -p '*.py'
python3 -m unittest discover -s tests/feature_extraction -p '*.py'
python3 -m unittest discover -s tests/common -p '*.py'
python3 -m unittest discover -s tests/tools_tests -p '*.py'
python3 -m unittest discover -s tests/processing -p '*.py'

python3 -m unittest discover -s tests/managers -p '*.py'
