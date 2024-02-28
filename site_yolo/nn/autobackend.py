from typing import Any
from torch._C import device
from ultralytics.nn.autobackend import AutoBackend as _AutoBackend_
import pdb


class AutoBackend(_AutoBackend_):

    def __call__(self, im, augment=False, visualize=False, embed=None):
        # pdb.set_trace()
        return super().forward(im, augment, visualize, embed)
