from ultralytics.utils import DEFAULT_CFG_DICT, RANK
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.engine.model import Model as _Model_

import pdb


class Model(_Model_):

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)

        model = model or self._smart_load('model')
        
        self.model = model(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg
        self.overrides['task'] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
