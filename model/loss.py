import torch
import torch.nn as nn
import torch.nn.functional as F

def l_num_loss(img1, img2, mask, l_num=1):
    diff = torch.abs(img1 - img2) * mask
    loss = torch.mean(diff**l_num)
    return loss

def cal_smooth_term_stitch(stitched_image:torch.Tensor, learned_mask:torch.Tensor):
    """

    Args:
        stitched_image (torch.Tensor): (B,3,H,Wc)
        learned_mask (torch.Tensor): (B,1,H,Wc)

    Returns:
        loss
    """

    DELTA = 1
    dh_mask = torch.abs(learned_mask[:,:,0:-1*DELTA,:] - learned_mask[:,:,DELTA:,:])
    dw_mask = torch.abs(learned_mask[:,:,:,0:-1*DELTA] - learned_mask[:,:,:,DELTA:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*DELTA,:] - stitched_image[:,:,DELTA:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*DELTA] - stitched_image[:,:,:,DELTA:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap

    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(diff_feature[:,:,0:-1*delta,:] + diff_feature[:,:,delta:,:])
    dw_diff_img = torch.abs(diff_feature[:,:,:,0:-1*delta] + diff_feature[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss
