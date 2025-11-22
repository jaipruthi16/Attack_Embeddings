"""
Defense module for CLIP adversarial attacks (FGSM / PGD)
Includes:
 - JPEG compression
 - Bit depth reduction
 - Random resize + padding defense
 - Total variation denoising
 - Combined defense function for CLIP inference
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from torchvision.transforms import functional as TF_F
from PIL import Image
import io

# ---------------------------------------
# 1. JPEG Compression Defense
# ---------------------------------------
def jpeg_compress(img_tensor, quality=50):
    """
    JPEG compress a normalized tensor image in [0,1]
    """
    img = img_tensor.clamp(0, 1).detach().cpu()
    pil = TF_F.to_pil_image(img)

    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    compressed = Image.open(buffer)

    return TF_F.to_tensor(compressed)


# ---------------------------------------
# 2. Bit-Depth Reduction (Quantization)
# ---------------------------------------
def bit_depth_reduction(img_tensor, bits=5):
    """
    Reduces the bit depth of the image (default 5 bits)
    """
    img = img_tensor.clamp(0, 1)
    levels = 2 ** bits
    return torch.round(img * levels) / levels


# ---------------------------------------
# 3. Random Resize + Padding (Feature-Squeezing Defense)
# ---------------------------------------
def random_resize_pad(img_tensor, min_size=192, max_size=224):
    """
    Resize image to random size then pad back to 224x224.
    Breaks gradient alignment used by PGD/FGSM.
    """
    target = torch.randint(min_size, max_size + 1, (1,)).item()
    img = TF_F.resize(img_tensor, (target, target))

    pad_h = 224 - target
    pad_w = 224 - target

    padded = TF_F.pad(img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2))
    return padded


# ---------------------------------------
# 4. Total Variation Minimization (TVM)
# ---------------------------------------
def tv_min(img, weight=0.1, iters=10):
    """
    Total Variation Minimization (Rudin-Osher-Fatemi Denoising).
    Reduces adversarial noise.
    """
    x = img.clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([x], lr=0.1)

    for _ in range(iters):
        tv = torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + \
             torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))

        optimizer.zero_grad()
        (weight * tv).backward()
        optimizer.step()

        x.data = x.data.clamp(0, 1)

    return x.detach()


# ---------------------------------------
# 5. Combined Defense Pipeline
# ---------------------------------------
def defend_image(img_tensor):
    """
    Apply ALL defenses in sequence (strongest combination).
    """
    x = img_tensor.clone().detach()

    # 1. JPEG compression
    x = jpeg_compress(x, quality=55)

    # 2. Bit depth reduction
    x = bit_depth_reduction(x, bits=5)

    # 3. Randomized resize + padding
    x = random_resize_pad(x)

    # 4. Optional: Mild TV denoising (can disable if speed needed)
    x = tv_min(x, weight=0.05, iters=5)

    return x.clamp(0,1)


# ---------------------------------------
# 6. CLIP Defense Wrapper (for inference)
# ---------------------------------------
def defended_forward(pipeline, pixel_values):
    """
    Use this instead of pipeline.model.get_image_features for defense.
    Example usage:
        defended_embeds = defended_forward(pipeline, pixel_values)
    """
    defended = defend_image(pixel_values.squeeze(0))
    defended = defended.unsqueeze(0).to(pipeline.model.device)

    embeds = pipeline.model.get_image_features(defended)
    return embeds / embeds.norm(dim=-1, keepdim=True)
