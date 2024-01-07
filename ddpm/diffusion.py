import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
from copy import deepcopy
from .ema import EMA
from .utils import extract




class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output: 
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        # classifier,
        # loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
        project_name = None
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        # self.classifier = classifier

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # if loss_type not in ["l1", "l2", "DB"]:
        #     raise ValueError("__init__() got unknown loss type")
        #
        # self.loss_type = loss_type
        self.num_timesteps = len(betas)
        self.project_name = project_name

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        # # "partial" can build a function
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, input, noise, t, y, use_ema=True):
        model_input = torch.cat((input, noise), dim= 1)
        return self.model(model_input, t)

    @torch.no_grad()
    def sample_noise(self, input, device, target, use_ema=True):
        noise = torch.randn(input.size(0), self.img_channels, *self.img_size, device=device)
        for t in range(self.num_timesteps - 1, -1, -1):
            # # number of time steps
            t_batch = torch.tensor([t], device=device).repeat(input.size(0))
            # # mean value of generate image
            noise = self.remove_noise(input, noise, t_batch, target, use_ema)
            if t > 0:
                # # adds variance on mean value
                noise += extract(self.sigma, t_batch, noise.shape) * torch.randn_like(input)
        return noise


    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   


    def get_predict(self, x, target, adv_perturb= None):

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        add_noise = perturbed_x - x
        if adv_perturb != None:
            perturbed_x += adv_perturb

        perturbed_x_in = torch.cat((perturbed_x, noise), dim= 1)
        predict_noise = self.model(perturbed_x_in, t)
        return predict_noise, add_noise

    def get_losses(self, x, t, target):
        predict_noise, noise = self.get_predict(x, target)
        loss = F.mse_loss(predict_noise, noise)
        return loss, predict_noise


    def forward(self, x, target):
        b, c, h, w = x.shape
        device = x.device
        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, target)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    alphas = []
    f0 = f(0, T)
    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    betas = []
    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

