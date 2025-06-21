import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        feats = img.view(img.size(0), -1) 
        return feats