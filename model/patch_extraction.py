"""
Patch extraction + feature vector generation, adapted from
src/feature_fusion/patch_extraction.py and
src/feature_fusion/feature_vector_generation.py in the original repo.

Note: the pretrained CNN weights fully overwrite the architecture's
initial parameters (including the SRM filters) when loaded via
load_state_dict, so exact-match of the *initialization* code isn't
required for correct inference -- only matching layer shapes is.
"""
import math

import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.util import view_as_windows
from torch.autograd import Variable


def get_patches(image_mat: np.ndarray, stride: int = 1024):
    """
    Extract 128x128x3 patches from an image using a sliding window.
    For typical photo sizes this yields a single patch (a top-left
    crop), matching the original repo's behavior.
    """
    h, w = image_mat.shape[0], image_mat.shape[1]

    # The sliding window requires each dimension to be >= 128px.
    # Upscale (preserving aspect ratio) if the image is smaller.
    min_dim = min(h, w)
    if min_dim < 128:
        scale = 128 / min_dim
        new_w, new_h = max(128, int(round(w * scale))), max(128, int(round(h * scale)))
        import cv2
        image_mat = cv2.resize(image_mat, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    window_shape = (128, 128, 3)
    windows = view_as_windows(image_mat, window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches.append(windows[m][n][0])
    return patches


def get_yi(model, patch):
    with torch.no_grad():
        model.eval()
        return model(patch)


def get_y_hat(y: np.ndarray, operation: str = "mean"):
    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise ValueError("operation must be 'mean' or 'max'")


def get_patch_yi(model, image_bgr: np.ndarray) -> np.ndarray:
    """
    Full per-image feature extraction: split into patches, run each
    through the CNN, and mean-pool into a single 400-D vector.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    y = []
    patches = get_patches(image_bgr, stride=1024)

    for patch in patches:
        img_tensor = transform(patch.copy())
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor.double())
        yi = get_yi(model=model, patch=img_variable)
        y.append(yi)

    y = np.vstack(tuple(y))
    y_hat = get_y_hat(y=y, operation="mean")
    return y_hat
