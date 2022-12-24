import math
import time

import torch
from tqdm import tqdm
import model.tools.ssim as ssim_pth


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_losses(loss_type):
    loss_specifics = {}
    loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

def init_meters(loss_type):
    losses = init_losses(loss_type)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    lpips = AverageMeter()
    return losses, psnrs, ssims, lpips

def quantize(img, rgb_range=255.):
    return img.mul(255 / rgb_range).clamp(0, 255).round()

def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-255) arguments.
    '''
    diff = (pred - gt).div(255)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)


def calc_metrics(im_pred, im_gt, mask=None):
    q_im_pred = quantize(im_pred.data, rgb_range=1.)
    q_im_gt = quantize(im_gt.data, rgb_range=1.)
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())
    ssim = ssim_pth.ssim(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), data_range=255)
    return psnr, ssim

def eval_LPIPS(model, im_pred, im_gt):
    im_pred = 2.0 * im_pred - 1
    im_gt = 2.0 * im_gt - 1
    dist = model.forward(im_pred, im_gt)[0]
    return dist

def eval_metrics(output, gt, psnrs, ssims, lpips, lpips_model=None, mask=None, psnrs_masked=None, ssims_masked=None):
    # PSNR should be calculated for each image
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs.update(psnr)
        ssims.update(ssim)
        if mask is not None:
            psnr_masked, ssim_masked = calc_metrics(output[b], gt[b], mask[b])
            psnrs_masked.update(psnr_masked)
            ssims_masked.update(ssim_masked)
        if lpips_model is not None:
            _lpips = eval_LPIPS(lpips_model, output[b].unsqueeze(0), gt[b].unsqueeze(0))
            lpips.update(_lpips)


def test(model, criterion, test_loader, loss_type, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = init_meters(loss_type)
    model.eval()
    criterion.eval()

    t = time.time()
    with torch.no_grad():
        for i, (inputs, gt) in enumerate(tqdm(test_loader)):
            im1 = inputs[0]
            im2 = inputs[1]

            # Forward
            out, feats = model(im1, im2)

            # Save loss values
            loss = criterion(out, gt)
            losses['total'].update(loss.item())

            # Evaluate metrics
            eval_metrics(out, gt, psnrs, ssims, lpips)

    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg))

    return losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg

