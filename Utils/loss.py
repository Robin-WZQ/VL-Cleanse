import torch
from torch.nn.functional import cosine_similarity

class SimilarityLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = cosine_similarity(input, target, dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class MSELoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.loss_fkt = torch.nn.MSELoss(reduction=reduction)
        self.flatten = flatten

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = self.loss_fkt(input, target)
        return loss

    
class AssimilationLoss(torch.nn.Module):
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor):
        loss_b = 0
        for i in range(input.shape[0]):
            input_c = input[i].squeeze(0)

            loss = cosine_similarity(input_c.unsqueeze(0), input_c.unsqueeze(1), dim=2)

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
                
            loss_b += loss
        
        return loss_b / input.shape[0]
    
class MaskL1Loss(torch.nn.Module):
    '''compute the L1 Norm of the mask'''
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor):
        loss = torch.mean(torch.abs(input))
        return loss
    
class UnifiedMaskLoss(torch.nn.Module):
    def __init__(self, lambda_size=0.1, lambda_tv=0.0005, p=2):
        super().__init__()
        self.lambda_size = lambda_size
        self.lambda_tv = lambda_tv
        self.p = p

    def forward(self, mask):  
        # 1. 面积约束 (Lp Norm) - 强制收缩空间
        loss_size = torch.pow(mask + 1e-8, self.p).mean()

        # 2. 连通性约束 (TV Loss) - 强制聚集成 Patch
        tv_h = torch.pow(mask[:, :, 1:, :] - mask[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(mask[:, :, :, 1:] - mask[:, :, :, :-1], 2).sum()
        loss_tv = (tv_h + tv_w)
        
        return loss_size + self.lambda_tv * loss_tv
        
        # mask_pred = torch.sigmoid(mask)  # [B, 1, H, W]
        # sparse_loss = mask_pred.mean()
        
        # return sparse_loss

class Loss_sim_inter(torch.nn.Module):
    '''Compute the inter-sample simialrity loss'''
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
        
    def forward(self, input: torch.Tensor):
        loss = 0
        for i in range(input.shape[0]):
            for j in range(i+1, input.shape[0]):
                loss += cosine_similarity(input[i].unsqueeze(0), input[j].unsqueeze(0), dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss