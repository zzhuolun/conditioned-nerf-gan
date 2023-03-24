import torch
import numpy as np
import pytorch_fid
from pytorch_fid.inception import InceptionV3
from enum import Enum
import time
from pathlib import Path
from datasets import _read_resize_shapenet
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
import os

FEATURE_DIMS = 768


def get_model(dev):
    """Return model for oFID calculation"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIMS]
    model = InceptionV3([block_idx]).to(dev)
    return model


def call_model(model, inp):
    """Call model on input [B, 3, H, W] and return [17*17*B, 768] features"""
    assert inp.ndim == 4 and inp.shape[1] == 3
    B = inp.shape[0]

    with torch.inference_mode():
        out = model(inp)
        assert len(out) == 1
        out = out[0]
        assert list(out.shape) == [B, FEATURE_DIMS, 17, 17]

    return torch.reshape(out.permute(0, 2, 3, 1), [17 * 17 * B, FEATURE_DIMS])


class Implementation(Enum):
    NumpyExact = 1
    CudaApproximate = 2


def compute_ofid_from_batch(model, img_gt, img_pred, impl):
    """Compute object-based FID (oFID). The B images should be from the same object. img_* has shape [B, 3, 128, 128] and B >= 10 at least"""

    assert isinstance(
        impl, Implementation
    ), f"impl must be an Implementation but is {type(impl)}"

    assert img_gt.shape == img_pred.shape
    assert img_gt.ndim == 4 and img_gt.shape[-1] == 128 and img_gt.shape[-2] == 128
    B = img_gt.shape[0]
    if B < 10:
        raise RuntimeError(
            f"Please use at least 10 (currently {B}) images because otherwise the metric is not stable."
        )

    # Compute features and moments
    feat_gt = call_model(model, img_gt)
    feat_pred = call_model(model, img_pred)

    mu_gt, sigma_gt = _get_mu_sigma(feat_gt)
    mu_pred, sigma_pred = _get_mu_sigma(feat_pred)

    if impl == Implementation.NumpyExact:
        tonp = lambda x: x.detach().cpu().numpy()
        ofid = numpy_calculate_frechet_distance(
            tonp(mu_gt), tonp(sigma_gt), tonp(mu_pred), tonp(sigma_pred)
        ).item()
    elif impl == Implementation.CudaApproximate:
        ofid = torch_calculate_frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred)
    else:
        raise NotImplementedError(f"Impl {impl} not implemented")

    return ofid


def _get_mu_sigma(feat):
    assert feat.ndim == 2 and feat.shape[-1] == FEATURE_DIMS

    return torch.mean(feat, 0), torch_cov(feat)


### Below Code from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py ###


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    """Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    m = 1.0 * m
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?
def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        print("wat")
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    out = (
        diff.dot(diff)
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2 * torch.trace(covmean)
    )
    return out


def compute_psnr(a: torch.Tensor, b: torch.Tensor, max_val=1.0) -> float:
    """Compute psnr for a batch of images

    Args:
        a (torch.Tensor): BxCxHxW
        b (torch.Tensor): BxCxHxW
    """
    assert a.shape == b.shape
    assert a.max() <= max_val
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).cpu()
    psnr = 20 * torch.log10(torch.Tensor([max_val])) - 10 * torch.log10(mse)
    return psnr.mean().item()


def compute_lpips(loss_fn, a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute lpips distance for a batch of images. Assume the input images are in the range [0,1]."""
    assert a.shape == b.shape

    # Normalize the images to range [-1, 1], as described here https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/README.md?plain=1#L19
    a_normalized = (2 * a) - 1
    b_normalized = (2 * b) - 1
    dist = loss_fn.forward(a_normalized, b_normalized)
    return dist.mean().item()


def compute_metrics_from_dirs(
    real_dir: str,
    gen_dir: str,
    car_lst: str,
    device: torch.device,
    B: int = 24,
    isapproximate=False,
):

    with open(car_lst, "r") as f:
        cars = f.readlines()
    cars = [i[:-1] for i in cars]
    ofid_ls = []
    lpips_ls = []
    psnr_ls = []
    totensor = transforms.ToTensor()

    inception_model = get_model(device)

    # Linearly calibrated models (LPIPS)
    lpip_loss = lpips.LPIPS(net="alex", spatial=False).to(device)
    # Can also set net = 'squeeze' or 'vgg'
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
    
    with torch.no_grad():
        for car in tqdm(cars):
            img_real_ls = []
            img_gen_ls = []
            for i in range(B):
                img_real_path = os.path.join(real_dir, f"{car}_{i:04}.png")
                img_gen_path = os.path.join(gen_dir, f"{car}_{i:04}.png")
                img_gen_ls.append(totensor(Image.open(img_gen_path).convert("RGB")))
                img_real_ls.append(totensor(Image.open(img_real_path).convert("RGB")))
            img_real = torch.stack(img_real_ls, 0).to(device)
            img_gen = torch.stack(img_gen_ls, 0).to(device)
            # Note: CudaApproximate will often returns NaN, use NumpyExact instead (only ca. 3 more minutes per evaluation.)
            # compute object fid
            ofid_ls.append(
                compute_ofid_from_batch(
                    inception_model,
                    img_real,
                    img_gen,
                    impl=Implementation.CudaApproximate
                    if isapproximate
                    else Implementation.NumpyExact,
                )
            )
            # compute LPIPS
            lpips_ls.append(compute_lpips(lpip_loss, img_real, img_gen))
            # compute PSNR
            psnr_ls.append(compute_psnr(img_real, img_gen))
            torch.cuda.empty_cache()

    ofid = sum(ofid_ls) / len(ofid_ls)
    lpips_score = sum(lpips_ls) / len(lpips_ls)
    psnr = sum(psnr_ls) / len(psnr_ls)

    # compute FID
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir], 128, "cuda", dims=2048
    )
    torch.cuda.empty_cache()
    return fid, ofid, lpips_score, psnr