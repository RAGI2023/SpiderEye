import torch

def l_num_loss(pred: torch.Tensor, target: torch.Tensor, num: int = 1) -> torch.Tensor:
    """
    Generalized L_num loss for regression tasks.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        num (int): The power to which the absolute difference is raised. Default is 1.

    Returns:
        torch.Tensor: Computed L_num loss.
    """
    diff = torch.abs(pred - target)
    loss = diff.pow(num)
    return loss.mean()
