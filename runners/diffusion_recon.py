import os
import numpy as np
from tqdm import tqdm
import logging
import torch
import torchvision.utils as tvu
import torch.utils.data as data
import torch.nn.functional as F
from models.diffusion import Model
from functions.process_data import *
from models.ema import EMAHelper
from models.unet import UNet
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

from datasets import get_dataset, data_transform, inverse_data_transform
from models.vae import VAE, AE

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    sample = mean + torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample



# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

mse_loss = torch.nn.MSELoss()

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.dataset, self.test_dataset = get_dataset(args, config)

        self.vae = UNet(6).to(self.device)
        self.vae = torch.nn.DataParallel(self.vae)

        for param in self.vae.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

    def loss_fn(self, recon_x, x, mu=0, logvar=0):
        BCE = F.mse_loss(recon_x, x)
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE# + KLD, BCE, KLD

    def training_reconstructor(self):
        print("Loading model")

        model = Model(self.config)
        train_loader = data.DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            print('NOT USING PRETRAINED MODEL')

        logging.info("Model loaded")
        ckpt_id = 0

        n = self.config.sampling.batch_size

        for param in model.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = True

        for epoch in range(5):
            for i, datas in enumerate(train_loader):

                #datas = next(iter(train_loader))

                # Modality A
                modA_len = self.dataset.channels[0]
                modA = slice(0, modA_len)

                # Modality B
                modB_len = self.dataset.channels[1]
                modB = slice(modA_len, modA_len + modB_len)

                if self.config.data.dataset == "BALVAN":
                    datas = datas.permute(0, 3, 1, 2)
                    x = datas[:, modA].float().to(self.device)  # used for training
                    y = datas[:, modB].float().to(self.device)  # using for sampling
                else:
                    (x, y) = datas

                # with torch.no_grad():
                y0 = x
                x0 = y

                x0 = (x0 - 0.5) * 2.
                y0 = (y0 - 0.5) * 2.

                self.optimizer.zero_grad()

                x = self.vae(x0)

                loss = self.loss_fn(x, y0)

                loss.backward()
                print('Loss is:', loss)
                self.optimizer.step()
                tvu.save_image(x0, os.path.join(self.args.image_folder,
                                                f'x0.png'))
                tvu.save_image(x, os.path.join(self.args.image_folder,
                                               f'x.png'))
                tvu.save_image(y0, os.path.join(self.args.image_folder,
                                                f'y.png'))

        for i, datas in enumerate(train_loader):
            modA_len = self.dataset.channels[0]
            modA = slice(0, modA_len)

            # Modality B
            modB_len = self.dataset.channels[1]
            modB = slice(modA_len, modA_len + modB_len)

            if self.config.data.dataset == "BALVAN":
                datas = datas.permute(0, 3, 1, 2)
                x = datas[:, modA].float().to(self.device)  # used for training
                y = datas[:, modB].float().to(self.device)  # using for sampling
            else:
                (x, y) = datas

            # with torch.no_grad():
            y0 = x
            x0 = y
            tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
            x0 = (x0 - 0.5) * 2.
            y0 = (y0 - 0.5) * 2.

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                insert_noise = int(total_noise_levels / 50)
                for param in self.vae.parameters():
                    param.requires_grad = False
                x0 = self.vae(x0)
                x = x0 * a[insert_noise - 1].sqrt() + e * (1.0 - a[insert_noise - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))

                logging.info('Ok')
                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):
                        t = (torch.ones(n) * i).to(self.device)

                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                        logvar=self.logvar,
                                                                        betas=self.betas)

                        x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        x = x_

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{it}.png'))

                        progress_bar.update(1)

            torch.save(x, os.path.join(self.args.image_folder,
                                       f'samples_{it}.pth'))
            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                       f'samples_{it}.png'))

            tvu.save_image(y0, os.path.join(self.args.image_folder,
                                            f'groudtruth_{it}.png'))
            avg_fid = []
            for index in range(x.shape[0]):
                fid = calculate_fid((x.cpu().detach().numpy()[index,0,:,:] + 1) * 0.5, y0.cpu().detach().numpy()[index,0,:,:])
                avg_fid.append(fid)
            avg_fid = np.mean(avg_fid)
            print('FID (different): %.3f' % avg_fid)


