import pdb
from pathlib import Path
from typing import Union
# from ...engine.model import Model
from ...nn.tasks import DetectionModel
from .detect.train import DetectionTrainer

from ultralytics.engine.model import Model  # noqa
# from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class RTDetector(Model):
    def __init__(self, model, task=None) -> None:
        super().__init__(model, task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': DetectionTrainer,
                'validator': DetectionValidator,
                'predictor': DetectionPredictor, 
            },
        }
