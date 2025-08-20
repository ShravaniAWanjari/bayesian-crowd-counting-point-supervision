import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLoss(nn.Module):
    def __init__(self, sigma = 8, d = 76.8):
        super(BayesianLoss, self).__init__()

        self.sigma = sigma
        self.d = d

    def forward(self, pred_density, points):
        if points.size(0) == 0:
            return torch.sum(pred_density.abs())
        
        batch_size, _, height, width = pred_density.size()

        if batch_size>1:
            raise NotImplementedError("This loss function is implemented for a batch size of 1.")
        
        pred_density = pred_density.squeeze(0).squeeze(0)

        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).to(pred_density.device).float()
        

        grid_flat= grid.reshape(-1,1,2)
        points_flat = points.unsqueeze(0)

        distances_sq = torch.sum((grid_flat - points_flat)**2, dim=-1)

        likelihoods = torch.exp(-distances_sq/(2*self.sigma**2))

        likelihood_sum = likelihoods.sum(dim=1, keepdim=True)

        likelihood_sum = torch.clamp(likelihood_sum, min=1e-8)

        posteriors = likelihoods/likelihood_sum

        pred_flat = pred_density.view(-1,1)
        expected_counts = torch.sum(posteriors* pred_flat, dim=0)

        gt_expected_counts = torch.ones_like(expected_counts)
        loss = F.l1_loss(expected_counts, gt_expected_counts, reduction='sum')

        min_distances_sq, _ = torch.min(distances_sq, dim=1, keepdim=True)
        min_distances = torch.sqrt(min_distances_sq)

        bg_likelihood = torch.exp(-(min_distances - self.d)**2/(2*self.sigma**2))

        denominator_plus = likelihood_sum + bg_likelihood
        posterior_plus = torch.cat((likelihoods, bg_likelihood), dim=1)/ denominator_plus

        expected_bg_count = torch.sum(posterior_plus[:, -1]*pred_flat.squeeze(), dim=0)

        loss += F.l1_loss(expected_bg_count, torch.zeros_like(expected_bg_count), reduction='sum')

        return loss
    