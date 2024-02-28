from ultralytics.nn.modules.conv import Conv, autopad
from torch import nn

class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.cv1 = Conv(c1, c1, k, s, p=autopad(k, p, d), g=c1, act=act)
        self.cv2 = Conv(c1, c2, 1, 1, p=0, act=act)
    
    def forward(self, x):
        return self.cv2(self.cv1(x))
