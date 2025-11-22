# defense.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from io import BytesIO
from PIL import Image

# ----------------------------------------------------
# JPEG COMPRESSION DEFENSE (preprocessing purification)
# ----------------------------------------------------
def jpeg_compress(x, quality=40):
    """
    x: tensor [B,C,H,W] in [0,1]
    returns: purified tensor
    """
    x = (x * 255).clamp(0,255).byte()
    out = []
    for img in x:
        pil_img = T.ToPILImage()(img)
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        pil_out = Image.open(buf)
        out.append(T.ToTensor()(pil_out))
    return torch.stack(out)


# ----------------------------------------------------
# FGSM + PGD attacks (for MAT / FAT)
# ----------------------------------------------------
def fgsm(model, images, labels, eps=0.1):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    adv = images + eps * images.grad.sign()
    return adv.clamp(0,1).detach()

def pgd(model, images, labels, eps=0.1, alpha=0.01, steps=5):
    ori = images.clone().detach()
    adv = ori.clone().detach()
    for _ in range(steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        eta = torch.clamp(adv - ori, -eps, eps)
        adv = (ori + eta).clamp(0,1).detach()
    return adv


# ----------------------------------------------------
# MAT — Mixed Adversarial Training
# ----------------------------------------------------
def train_mat(model, loader, device, eps=0.1, attack="fgsm", lr=1e-3, epochs=5):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if attack == "fgsm":
                adv = fgsm(model, x, y, eps)
            else:
                adv = pgd(model, x, y, eps)

            mixed_x = torch.cat([x, adv], dim=0)
            mixed_y = torch.cat([y, y], dim=0)

            optim.zero_grad()
            logits = model(mixed_x)
            loss = F.cross_entropy(logits, mixed_y)
            loss.backward()
            optim.step()

        print(f"[MAT] Epoch {ep+1}/{epochs} Loss={loss.item():.4f}")


# ----------------------------------------------------
# FAT — Friendly Adversarial Training (early-stop PGD)
# ----------------------------------------------------
def train_fat(model, loader, device, eps=0.1, alpha=0.01, early_steps=1, lr=1e-3, epochs=5):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Early-stopped PGD (1–2 steps only)
            adv = pgd(model, x, y, eps, alpha, steps=early_steps)

            optim.zero_grad()
            logits = model(adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()

        print(f"[FAT] Epoch {ep+1}/{epochs} Loss={loss.item():.4f}")


# ----------------------------------------------------
# SIMPLE WRAPPER: Purify + Predict
# ----------------------------------------------------
def defend_and_predict(model, x, quality=40):
    model.eval()
    purified = jpeg_compress(x, quality)
    with torch.no_grad():
        return model(purified)
