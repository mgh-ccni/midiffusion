import torch, torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
from utils import histogram, MutualInformation
import torch
import torch.fft as fft

def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)

    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])

def normalize_channel_wise(tensor1, tensor2):
    # Calculate mean and standard deviation along the channel dimension
    max1 = tensor1.max()
    min1 = tensor1.min()
    max2 = tensor2.max()
    min2 = tensor2.min()

    # Normalize tensors channel-wise
    normalized_tensor1 = (tensor1 - min1) / (max1 - min1)
    normalized_tensor2 = (tensor2 - min2) / (max2 - min2)

    return normalized_tensor1, normalized_tensor2

def statistical_align_normal(source, target, patch_size, ch_sz=1, sig=2, normalize=True):
    MI = MutualInformation.MutualInformation(num_bins=patch_size**3, sigma=sig, normalize=normalize, device=source.device)

    if target.shape[1] == 1:
        if patch_size == 3:
            p1d = (1, 1, 1, 1)  # windows size used (1, 4, 1, 4)
        elif patch_size == 5:
            p1d = (2, 2, 2, 2)
        targettarget = F.pad(target, p1d, "constant", 0)  # effectively zero padding
        stride = 1  # patch stride
        patchestarget = targettarget.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        # Create searching tensor of target
        augmented_target1 = patchestarget.permute(0, 1, 2, 4, 3, 5).reshape(
            [source.shape[0], ch_sz, source.shape[-2] * patch_size, source.shape[-1] * patch_size])

        augmented_target_padded = F.pad(augmented_target1, p1d, "constant", 0)
        patchestarget = augmented_target_padded.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        augmented_target2 = patchestarget.permute(0, 1, 2, 4, 3, 5).reshape(
            [augmented_target1.shape[0], ch_sz, augmented_target1.shape[-2] * patch_size,
             augmented_target1.shape[-1] * patch_size])

        # Create reference tensor from source
        sourcesource = F.pad(source, p1d, "constant", 0)  # effectively zero padding
        stride = 1  # patch stride
        patchessource = sourcesource.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        augmented_source1 = patchessource.permute(0, 1, 2, 4, 3, 5).reshape(
            [source.shape[0], ch_sz, source.shape[-2] * patch_size, source.shape[-1] * patch_size])

        augmented_patchessource = torch.tile(patchessource, (patch_size, patch_size))
        src_shape = augmented_patchessource.shape
        augmented_source2 = augmented_patchessource.permute(0, 1, 2, 4, 3, 5).reshape(
            [src_shape[0], ch_sz, src_shape[2] * patch_size ** 2, src_shape[3] * patch_size ** 2])

        resy = augmented_source2.unfold(2, patch_size ** 2, patch_size ** 2).unfold(3, patch_size ** 2,
                                                                                    patch_size ** 2).unfold(4, patch_size,
                                                                                                            patch_size).unfold(
            5, patch_size, patch_size).reshape(source.shape[0] * 128 * 128 * patch_size * patch_size, 1, patch_size,
                                               patch_size)
        resx = augmented_target2.unfold(2, patch_size ** 2, patch_size ** 2).unfold(3, patch_size ** 2,
                                                                                    patch_size ** 2).unfold(4, patch_size,
                                                                                                            patch_size).unfold(
            5, patch_size, patch_size).reshape(source.shape[0] * 128 * 128 * patch_size * patch_size, 1, patch_size,
                                               patch_size)
        score = MI(resy, resx)
        uscore = score.reshape(source.shape[0], 128, 128, patch_size, patch_size)
        sim = uscore.permute(0, 1, 3, 2, 4).reshape(source.shape[0], 1, 128 * patch_size, 128 * patch_size)
        maxpool = nn.MaxPool2d(patch_size, patch_size, return_indices=True)
        pool_kl, indicesKL = maxpool(sim)
        output = retrieve_elements_from_indices(augmented_target1, indicesKL)
        output_sim = retrieve_elements_from_indices(sim, indicesKL)
    else:
        output = torch.zeros_like(source)
        output_sim = torch.zeros_like(source)
        for i in range(target.shape[1]):
            tensor1, tensor2= statistical_align(source, target[:, i, :, :].unsqueeze(1), patch_size, ch_sz, sig)

            output[:, i:i+1, :, :], output_sim[:, i:i+1, :, :] = normalize_channel_wise(tensor1, tensor2)

    return output, output_sim

def statistical_align(source, target, patch_size, ch_sz=1, sig=2, normalize=True):
    MI = MutualInformation.MutualInformation(num_bins=patch_size**3, sigma=sig, normalize=normalize, device=source.device)
    if source.shape[1] == 1:
        if patch_size == 3:
            p1d = (1, 1, 1, 1)  # windows size used (1, 4, 1, 4)
        elif patch_size == 5:
            p1d = (2, 2, 2, 2)
        targettarget = F.pad(target, p1d, "constant", 0)  # effectively zero padding
        stride = 1  # patch stride
        patchestarget = targettarget.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        # Create searching tensor of target
        augmented_target1 = patchestarget.permute(0, 1, 2, 4, 3, 5).reshape(
            [source.shape[0], ch_sz, source.shape[-2] * patch_size, source.shape[-1] * patch_size])

        augmented_target_padded = F.pad(augmented_target1, p1d, "constant", 0)
        patchestarget = augmented_target_padded.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        augmented_target2 = patchestarget.permute(0, 1, 2, 4, 3, 5).reshape(
            [augmented_target1.shape[0], ch_sz, augmented_target1.shape[-2] * patch_size,
             augmented_target1.shape[-1] * patch_size])

        # Create reference tensor from source
        sourcesource = F.pad(source, p1d, "constant", 0)  # effectively zero padding
        stride = 1  # patch stride
        patchessource = sourcesource.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        augmented_source1 = patchessource.permute(0, 1, 2, 4, 3, 5).reshape(
            [source.shape[0], ch_sz, source.shape[-2] * patch_size, source.shape[-1] * patch_size])

        augmented_patchessource = torch.tile(patchessource, (patch_size, patch_size))
        src_shape = augmented_patchessource.shape
        augmented_source2 = augmented_patchessource.permute(0, 1, 2, 4, 3, 5).reshape(
            [src_shape[0], ch_sz, src_shape[2] * patch_size ** 2, src_shape[3] * patch_size ** 2])

        resy = augmented_source2.unfold(2, patch_size ** 2, patch_size ** 2).unfold(3, patch_size ** 2,
                                                                                    patch_size ** 2).unfold(4, patch_size,
                                                                                                            patch_size).unfold(
            5, patch_size, patch_size).reshape(source.shape[0] * 128 * 128 * patch_size * patch_size, 1, patch_size,
                                               patch_size)
        resx = augmented_target2.unfold(2, patch_size ** 2, patch_size ** 2).unfold(3, patch_size ** 2,
                                                                                    patch_size ** 2).unfold(4, patch_size,
                                                                                                            patch_size).unfold(
            5, patch_size, patch_size).reshape(source.shape[0] * 128 * 128 * patch_size * patch_size, 1, patch_size,
                                               patch_size)
        score = MI(resy, resx)
        uscore = score.reshape(source.shape[0], 128, 128, patch_size, patch_size)
        sim = uscore.permute(0, 1, 3, 2, 4).reshape(source.shape[0], 1, 128 * patch_size, 128 * patch_size)
        maxpool = nn.MaxPool2d(patch_size, patch_size, return_indices=True)
        pool_kl, indicesKL = maxpool(sim)
        output = retrieve_elements_from_indices(augmented_target1, indicesKL)
        output_sim = retrieve_elements_from_indices(sim, indicesKL)
    else:
        output = torch.zeros_like(source)
        output_sim = torch.zeros_like(source)
        for i in range(source.shape[1]):
            output[:, i:i+1, :, :], output_sim[:, i:i+1, :, :] = statistical_align(source[:, i, :, :].unsqueeze(1),
                                                                        target[:, i, :, :].unsqueeze(1), patch_size, ch_sz, sig)
    return output, output_sim
