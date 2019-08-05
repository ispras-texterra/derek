import json
import os


class BasicExperimentRunner:
    def __init__(self, base_props: dict, paths: dict):
        self.base_props = base_props
        self.paths = paths

    def run(self, trainer, working_dir: str):
        with open(os.path.join(working_dir, 'status'), 'w') as f:
            f.write('init')
        results = {}
        experiment_result = trainer.train(self.base_props, self.paths, working_dir)
        results.update(experiment_result)

        mode = 'r+' if os.path.exists(os.path.join(working_dir, 'result')) else 'w'
        with open(os.path.join(working_dir, 'result'), mode) as f:
            if mode == 'r+':
                prev_results = json.load(f)
                self._update_result(prev_results, results)
                f.seek(0)
            json.dump(results, f, indent=4, sort_keys=True)

        os.remove(os.path.join(working_dir, 'status'))

    def _update_result(self, prev_results: dict, results: dict):
        for metric, res in results.items():
            res.update(prev_results[metric])
