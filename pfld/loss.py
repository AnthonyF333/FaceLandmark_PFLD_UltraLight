import torch
from torch import nn
import math


class LandmarkLoss(nn.Module):
    def __init__(self, n_landmark=98):
        super(LandmarkLoss, self).__init__()
        self.n_landmark = n_landmark

    def forward(self, landmark_gt, landmark_pred):
        loss = wing_loss(landmark_gt, landmark_pred, N_LANDMARK=self.n_landmark)
        return loss


def wing_loss(y_true, y_pred, N_LANDMARK, w=10.0, epsilon=2.0):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x / epsilon), absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss
