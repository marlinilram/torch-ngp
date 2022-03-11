import torch
import torch.nn as nn


class ExposureRefine(nn.Module):
    def __init__(self, num_frame):
        super().__init__()

        self.num_frame = num_frame
        self.vars = nn.Parameter(torch.zeros(self.num_frame))

    def get_exposure_scale(self, ids):
        return torch.exp(0.6931471805599453*self.vars[ids])
