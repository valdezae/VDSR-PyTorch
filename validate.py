# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted

import config
import imgproc
from model import VDSR


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        window_size: Size of the gaussian filter window
        size_average: If True, average SSIM across all batches
        
    Returns:
        SSIM value
    """
    # Ensure same data type for both tensors
    dtype = img1.dtype
    
    # Check input dimensions
    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    elif len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    
    if len(img2.shape) == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
        
    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Generate gaussian kernel
    kernel_size = window_size
    sigma = 1.5
    
    # Create 1D gaussian kernel
    gauss = torch.Tensor([np.exp(-(x - kernel_size//2)**2/float(2*sigma**2)) for x in range(kernel_size)])
    gauss = gauss / gauss.sum()
    
    # Create 2D kernel by outer product
    _1D_kernel = gauss.unsqueeze(1)
    _2D_kernel = _1D_kernel.mm(_1D_kernel.t()).float().unsqueeze(0).unsqueeze(0)
    
    # Move kernel to same device as input AND convert to same dtype
    kernel = _2D_kernel.expand(img1.size(1), 1, kernel_size, kernel_size).to(img1.device).to(dtype)
    
    # Pad input for valid convolution
    padded_img1 = F.pad(img1, (window_size//2, window_size//2, window_size//2, window_size//2), mode='replicate')
    padded_img2 = F.pad(img2, (window_size//2, window_size//2, window_size//2, window_size//2), mode='replicate')
    
    # Calculate means
    mu1 = F.conv2d(padded_img1, kernel, groups=img1.shape[1])
    mu2 = F.conv2d(padded_img2, kernel, groups=img2.shape[1])
    
    # Calculate squares of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(padded_img1 * padded_img1, kernel, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(padded_img2 * padded_img2, kernel, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(padded_img1 * padded_img2, kernel, groups=img1.shape[1]) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def main() -> None:
    # Initialize the super-resolution model
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load VDSR model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create folder for comparison images
    comparison_dir = os.path.join(results_dir, "comparisons")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation indices
    total_psnr = 0.0
    total_ssim = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])
        comparison_image_path = os.path.join(comparison_dir, f"comparison_{file_names[index]}")

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Make high-resolution image
        hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0
        
        # Handle grayscale images (if any)
        if len(hr_image.shape) == 2 or hr_image.shape[2] == 1:
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_GRAY2BGR)
        
        hr_image_height, hr_image_width = hr_image.shape[:2]
        hr_image_height_remainder = hr_image_height % 12
        hr_image_width_remainder = hr_image_width % 12
        hr_image = hr_image[:hr_image_height - hr_image_height_remainder, :hr_image_width - hr_image_width_remainder, ...]

        # Make low-resolution image
        lr_image = imgproc.imresize(hr_image, 1 / config.upscale_factor)
        lr_image = imgproc.imresize(lr_image, config.upscale_factor)

        # Convert BGR image to YCbCr image
        lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

        # Split YCbCr image data
        lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
        hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert Y image data convert to Y tensor data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Cal PSNR
        total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))
        
        # Cal SSIM - ensure both tensors have the same dtype
        try:
            current_ssim = calculate_ssim(sr_y_tensor, hr_y_tensor).item()
            total_ssim += current_ssim
        except RuntimeError as e:
            print(f"SSIM calculation error: {e}")
            current_ssim = 0.0
            # Try with float32 tensors as fallback
            try:
                sr_y_float = sr_y_tensor.float()
                hr_y_float = hr_y_tensor.float()
                current_ssim = calculate_ssim(sr_y_float, hr_y_float).item()
                total_ssim += current_ssim
                print("SSIM calculated using float32 tensors")
            except Exception as e2:
                print(f"Float32 SSIM calculation also failed: {e2}")

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        
        # Save the SR image
        cv2.imwrite(sr_image_path, sr_image * 255.0)
        
        # Create side-by-side comparison image
        # Make sure all images are in the same range [0,1] before converting to uint8
        lr_display = np.clip(lr_image * 255.0, 0, 255).astype(np.uint8)
        hr_display = np.clip(hr_image * 255.0, 0, 255).astype(np.uint8)
        sr_display = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
        
        # Make sure images have the same dimensions for comparison
        # (1) Ensure same width
        min_width = min(lr_display.shape[1], hr_display.shape[1], sr_display.shape[1])
        lr_display = lr_display[:, :min_width]
        hr_display = hr_display[:, :min_width]
        sr_display = sr_display[:, :min_width]
        
        # (2) Ensure same height
        min_height = min(lr_display.shape[0], hr_display.shape[0], sr_display.shape[0])
        lr_display = lr_display[:min_height, :]
        hr_display = hr_display[:min_height, :]
        sr_display = sr_display[:min_height, :]
        
        # Add labels to each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Create label backgrounds
        label_height = 30
        lr_with_label = cv2.copyMakeBorder(lr_display, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        hr_with_label = cv2.copyMakeBorder(hr_display, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        sr_with_label = cv2.copyMakeBorder(sr_display, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Calculate metrics for current image
        current_psnr = 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2)).item()
        
        # Add text labels with metrics
        cv2.putText(lr_with_label, "LR", (10, 20), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(hr_with_label, "HR", (10, 20), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(sr_with_label, f"SR - PSNR: {current_psnr:.2f}dB SSIM: {current_ssim:.4f}", 
                   (10, 20), font, font_scale, (255, 255, 255), thickness)
        
        # Make sure all images have the same channels (in case some are grayscale)
        if len(lr_with_label.shape) == 2:
            lr_with_label = cv2.cvtColor(lr_with_label, cv2.COLOR_GRAY2BGR)
        if len(hr_with_label.shape) == 2:
            hr_with_label = cv2.cvtColor(hr_with_label, cv2.COLOR_GRAY2BGR)
        if len(sr_with_label.shape) == 2:
            sr_with_label = cv2.cvtColor(sr_with_label, cv2.COLOR_GRAY2BGR)
            
        # Concatenate images horizontally
        try:
            comparison_image = cv2.hconcat([lr_with_label, hr_with_label, sr_with_label])
            
            # Save comparison image
            cv2.imwrite(comparison_image_path, comparison_image)
        except Exception as e:
            print(f"Error creating comparison image: {e}")
            print(f"LR shape: {lr_with_label.shape}, HR shape: {hr_with_label.shape}, SR shape: {sr_with_label.shape}")

    # Calculate and print average metrics
    avg_psnr = total_psnr / total_files
    avg_ssim = total_ssim / total_files
    print(f"PSNR: {avg_psnr:4.2f}dB")
    print(f"SSIM: {avg_ssim:4.4f}")
    print(f"Side-by-side comparison images saved to '{os.path.abspath(comparison_dir)}'")


if __name__ == "__main__":
    main()
