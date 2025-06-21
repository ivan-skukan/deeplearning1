import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k // 2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        
        self.block1 = _BNReluConv(input_channels, emb_size, k=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block2 = _BNReluConv(emb_size, emb_size, k=3)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block3 = _BNReluConv(emb_size, emb_size, k=3)
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        return self.forward(img)

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        loss = F.triplet_margin_loss(a_x, p_x, n_x, margin=1.0, p=2)
        return loss

    def forward(self, x):
        x = self.block1(x)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)
        x = self.block3(x)
        x = self.mp3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x