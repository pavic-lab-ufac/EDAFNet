import torch
import torch.nn as nn
import torchvision
import math
import piq
import torch.nn.functional as F

# -------------------------------------------------------------------
# Funções e Classes de Suporte (copiadas de loss1.py)
# -------------------------------------------------------------------

class FFTLoss(nn.Module):
    """Calcula a L1 loss no domínio da frequência."""
    def __init__(self, reduction="mean"):
        super(FFTLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Separa as partes real e imaginária para o cálculo da L1 loss
        pred_fft_stacked = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft_stacked = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        return F.l1_loss(pred_fft_stacked, target_fft_stacked)


def gram_matrix(x):
    """Calcula a matriz de Gram para a style loss."""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)


class VGGPerceptualLoss(nn.Module):
    """Calcula a VGG Perceptual Loss (content e style)."""
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        blocks = [
            vgg.features[:4].eval(),
            vgg.features[4:9].eval(),
            vgg.features[9:16].eval(),
            vgg.features[16:23].eval()
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss_content = 0.0
        loss_style = 0.0
        x, y = input, target
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_content += F.l1_loss(x, y)
            if i in style_layers:
                loss_style += F.l1_loss(gram_matrix(x), gram_matrix(y))
                
        return loss_content, loss_style


def range_compressor(hdr_img, mu=5000):
    """Aplica a compressão mu-law."""
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)

# -------------------------------------------------------------------
# Nova Classe de Loss (Combinação de loss1.py e loss.py)
# -------------------------------------------------------------------

class Loss(nn.Module):
    """
    Combina as funções de loss de `loss1.py` (L1 comprimido, VGG, SSIM, FFT)
    mas com o formato de saída de `loss.py`.
    
    Retorna:
        - total_loss (torch.Tensor): A loss total ponderada, para o backpropagation.
        - loss_dict (dict): Um dicionário com as losses individuais para logging.
    """
    def __init__(self, w_recon=1.0, w_vgg=0.1, w_ssim=0.2, w_fft=0.1, mu=5000):
        super(Loss, self).__init__()
        self.w_recon = w_recon
        self.w_vgg = w_vgg
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.mu = mu

        self.loss_l1 = nn.L1Loss()
        self.loss_vgg = VGGPerceptualLoss(resize=False)  # 'resize=False' é geralmente preferível se as imagens já têm o tamanho certo
        self.loss_ssim = piq.SSIMLoss(data_range=1.0, reduction='mean')
        self.loss_fft = FFTLoss()

    def forward(self, pred_linear, target_linear):
        # Dicionário para armazenar as losses para logging
        loss_dict = {}

        # --- Cálculos nas imagens com clamp [0, 1] para VGG, SSIM e FFT ---
        pred_clamped = torch.clamp(pred_linear, 0, 1)
        target_clamped = torch.clamp(target_linear, 0, 1)
        
        # VGG Perceptual Loss (apenas a de conteúdo)
        loss_vgg, _ = self.loss_vgg(pred_clamped, target_clamped)
        
        # SSIM Loss (estrutural)
        # piq.SSIMLoss retorna 1 - SSIM, então já é uma loss
        loss_ssim = self.loss_ssim(pred_clamped, target_clamped)

        # FFT Loss (frequência)
        loss_fft = self.loss_fft(pred_clamped, target_clamped)

        # --- Cálculo na imagem comprimida para a L1 de reconstrução ---
        pred_compressed = range_compressor(pred_linear, self.mu)
        target_compressed = range_compressor(target_linear, self.mu)
        loss_recon = self.loss_l1(pred_compressed, target_compressed)

        # --- Ponderação e soma das losses ---
        total_loss = (
            self.w_recon * loss_recon +
            self.w_vgg * loss_vgg +
            self.w_ssim * loss_ssim +
            self.w_fft * loss_fft
        )

        # Preenche o dicionário com as losses ponderadas para consistência no logging
        loss_dict['loss_recon'] = self.w_recon * loss_recon
        loss_dict['loss_vgg'] = self.w_vgg * loss_vgg
        loss_dict['loss_ssim'] = self.w_ssim * loss_ssim
        loss_dict['loss_fft'] = self.w_fft * loss_fft
        
        # Retorna a loss total (para .backward()) e o dicionário (para logging)
        return total_loss, loss_dict
