import torch.nn as nn
import torch

class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    def forward(self, output, target, input=None):
        return torch.mean((target - output)**2)

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss"""
    def forward(self, output, target, input=None):
        return torch.mean(torch.abs((target - output) / target))

class GaussianLikelihoodLoss(nn.Module):
    """Gaussian Likelihood Loss"""
    def forward(self, prediction, target, cov):
        batch_size, n, time_steps = prediction.size()[0], prediction.size()[1], prediction.size()[2]
        loss = 0.0
        eps = torch.tensor([1e-12])

        for t in range(time_steps):
            diff = torch.unsqueeze(prediction[:, :, t] - target[:, :, t], 2)
            inv_sigma = torch.linalg.inv(cov[:, :, :, t])
            curr_det = torch.linalg.det(cov[:, :, :, t])
            curr_loss = torch.bmm(torch.transpose(diff, 1, 2), inv_sigma)
            curr_loss = torch.bmm(curr_loss, diff)
            loss += torch.mean(curr_loss, dim=0) + torch.mean(torch.log(curr_det))
        return loss / time_steps