import json
import os
import sys

from derek.common.logger import init_logger
from tools.common.experiment_runner import BasicExperimentRunner
from tools.coref.coref_trainer import CorefTrainer


def main():
    if len(sys.argv) < 2:
        print("Usage: <working-dir>")
        return
    with open(os.path.join(sys.argv[1], 'base_props.json')) as f:
        base_props = json.load(f)
    with open(os.path.join(sys.argv[1], 'params.json')) as f:
        paths = json.load(f)

    runner = BasicExperimentRunner(base_props, paths)
    coref_trainer = CorefTrainer()

    runner.run(coref_trainer, sys.argv[1])


if __name__ == '__main__':
    init_logger('logger')
    main()
