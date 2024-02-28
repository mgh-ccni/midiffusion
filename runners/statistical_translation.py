import numpy as np
from tqdm import tqdm

import torchvision.utils as tvu
import torch.utils.data as data
import cv2

from models.diffusion import Model
from models.unet import create_model
from functions.process_data import *
from models.ema import EMAHelper

from datasets import get_dataset, data_transform, inverse_data_transform
from models.conditioner import Ridge, polynomial, statistical_align, statistical_align_normal
import matplotlib.pyplot as plt
import time
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

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


def image_editing_denoising_step_flexible_mask(x, x_s, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, x_s, t)

    x = x[:, 0, :, :].unsqueeze(1)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)
    #mean = model_output
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

        self.regressor = polynomial(degree=3, device=self.device) #Ridge(alpha=1e-3, fit_intercept=True, device=self.device)

    def evaluation_sample(self, logging):
        print("Loading model")

        if self.args.use_unet:
            model = create_model(self.config.data.image_size, self.config.model.in_channels, self.config.model.ch,
                                 self.config.model.num_res_blocks)
            print("Use UNet")
        else:
            print("Use ResNet")
            model = Model(self.config)

        train_loader = data.DataLoader(
            self.dataset,
            batch_size=self.config.sampling.batch_size,
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
        model.eval()
        fid_full_avg = []
        #logging.info("Prepare Dataset")
        #datas = next(iter(train_loader))
        print('TEST MODEL INIT PASS')

        try:
            for it, datas in enumerate(train_loader):

                # Modality A
                modA_len = self.dataset.channels[0]
                modA = slice(0, modA_len)

                # Modality B
                modB_len = self.dataset.channels[1]
                modB = slice(modA_len, modA_len + modB_len)

                logging.info('Modality A has ' + str(modA_len) + ' channels.')
                logging.info('Modality B has ' + str(modB_len) + ' channels.')
                logging.info("Dataset prepared")

                if self.config.data.dataset == "BALVAN" or "MRI":
                    datas = datas.permute(0, 3, 1, 2)
                    x = datas[:, modA].float().to(self.device)  # used for training
                    y = datas[:, modB].float().to(self.device)  # using for sampling
                elif self.config.data.dataset == "ZURICH" or self.config.data.dataset == "ELICEIRI":
                    datas = datas.permute(0, 3, 1, 2)
                    x = datas[:, modA].float().to(self.device)
                    y = datas[:, modB].float().to(self.device)
                    try:
                        assert x.shape[1] == y.shape[1] == 3
                    except AssertionError:
                        if x.shape[1] == 1:
                            x = x.repeat(1, 3, 1, 1)
                        if y.shape[1] == 1:
                            y = y.repeat(1, 3, 1, 1)
                else:
                    (x, y) = datas

                x = data_transform(self.config, x)
                y = data_transform(self.config, y)
                print('TEST DATA TRANSFORM PASS')

                with torch.no_grad():
                    # y--->x
                    y0 = x
                    x0 = y
                    #tvu.save_image(inverse_data_transform(self.config, x0), os.path.join(self.args.image_folder, f'original_{it}.png'))
                    cv2.imwrite(os.path.join(self.args.image_folder, f'original_{it}.png'), 255 * inverse_data_transform(self.config, x0)[0].permute(1, 2, 0).cpu().numpy())
                    print(y0.max(), y0.min())
                    e = torch.randn_like(x0)
                    total_noise_levels = self.args.t
                    a = (1 - self.betas).cumprod(dim=0)
                    insert_noise = total_noise_levels
                    #x0 = x0 * a[insert_noise - 1].sqrt()
                    x = e #* (1.0 - a[insert_noise - 1]).sqrt() #+ x0 * a[insert_noise - 1].sqrt()
                    if x0.shape[1] == 3:
                        x0_squeezed = (x0[:, 0, :, :] * 0.299 + x0[:, 1, :, :] * 0.587 + x0[:, 2, :, :] * 0.114).unsqueeze(1)
                        e = torch.randn_like(x0_squeezed)
                        x_squeezed = e
                        _, sim_self = statistical_align_normal(x0_squeezed, x0,
                                                     patch_size=self.config.model.geo_patch_sz, sig=self.config.model.MIsigma, normalize=self.config.model.MInormalization)  # x_: cross-correlation aligned x; sim: self-similarity matrix
                    else:
                        _, sim_self = statistical_align(x0, x0,
                                                patch_size=self.config.model.geo_patch_sz, sig=self.config.model.MIsigma, normalize=self.config.model.MInormalization)  # x_: cross-correlation aligned x; sim: self-similarity matrix
                    
                    first_run = True
                    with tqdm(total=total_noise_levels, desc="Iteration {}".format(len(train_loader))) as progress_bar:
                        for i in reversed(range(total_noise_levels)):
                            t = (torch.ones(n) * i).to(self.device)
                            if first_run:
                                starting_time = time.time()

                            if x0.shape[1] == 3:
                                _, sim_crox = statistical_align(x0_squeezed, x_squeezed,
                                                              patch_size=self.config.model.geo_patch_sz,
                                                              sig=self.config.model.MIsigma, normalize=self.config.model.MInormalization)  # x_: cross-correlation aligned x; sim: self-similarity matrix
                            else:
                                x_, sim_crox = statistical_align(x0, x,
                                                       patch_size=self.config.model.geo_patch_sz, sig=self.config.model.MIsigma, normalize=self.config.model.MInormalization)  # x_: cross-correlation aligned x; sim: self-similarity matrix

                            x_ = torch.cat((sim_crox, sim_self), dim=1)


                            # conditioner_weight = 1 - 1/(1+i)
                            # x_ = conditioner_weight * x_
                            if first_run:
                                ending_time = time.time()
                                print(f'Iteration {i} took {ending_time - starting_time} seconds')

                                #print('ATTENTION, IS IN REBUTTAL MODE, THIS IS WRONG FOR THE CORRECT EXPERIMENT')
                            x = image_editing_denoising_step_flexible_mask(x, x_, t=t, model=model,
                                                                            logvar=self.logvar,
                                                                            betas=self.betas)
                            first_run = False

                            # # added intermediate step vis
                            if ((i - 99) % 50 == 0 and i < 350) or i ==0:
                                x_inv = inverse_data_transform(self.config, x)
                                tvu.save_image(x_inv, os.path.join('./temp/plots_for_paper/',f'noise_t_{i}_{it}.png'))

                            progress_bar.update(1)

                    x_inv = inverse_data_transform(self.config, x)
                    # tvu.save_image(x_inv, os.path.join(self.args.image_folder,
                    #                                            f'samples_{it}.png'))
                    cv2.imwrite(os.path.join(self.args.image_folder,f'samples_{it}.png'), 255 * x_inv[0].permute(1,2,0).cpu().numpy())
                    y0_inv = inverse_data_transform(self.config, y0)
                    #tvu.save_image(y0, os.path.join(self.args.image_folder, f'groudtruth_{it}.png'))

                    #tvu.save_image(y0_inv, os.path.join(self.args.image_folder, f'groudtruth_{it}.png'))
                    cv2.imwrite(os.path.join(self.args.image_folder, f'groudtruth_{it}.png'),
                                y0_inv[0].permute(1, 2, 0).cpu().numpy() * 255)
                    avg_fid = []

                    PATH1 = './temp/' + self.args.image_folder + '/res/'
                    if not os.path.exists(PATH1):
                        os.makedirs(PATH1)
                    PATH2 = './temp/' + self.args.image_folder + '/tar/'
                    if not os.path.exists(PATH2):
                        os.makedirs(PATH2)
                    PATH3 = './temp/' + self.args.image_folder + '/src/'
                    if not os.path.exists(PATH3):
                        os.makedirs(PATH3)
                    for index in range(x.shape[0]):
                        fid = 0#calculate_fid(x_inv.cpu().detach().numpy()[index,0,:,:], y0_inv.cpu().detach().numpy()[index,0,:,:])
                        avg_fid.append(fid)
                        if y0_inv.shape[1] == 1:
                            plt.imsave(PATH1 + 'img_{}_{}.png'.format(it, index), x_inv.cpu().detach().numpy()[index,0,:,:], cmap = 'gray')
                            plt.imsave(PATH2 + 'img_{}_{}.png'.format(it, index),y0_inv.cpu().detach().numpy()[index,0,:,:], cmap = 'gray')
                            plt.imsave(PATH3 + 'img_{}_{}.png'.format(it, index), x0.cpu().detach().numpy()[index,0,:,:], cmap = 'gray')

                        else:
                            plt.imsave(PATH1 + 'img_{}_{}.png'.format(it, index), x_inv.cpu().detach().numpy()[index,:,:,:].swapaxes(0, 2).swapaxes(0, 1))
                            plt.imsave(PATH2 + 'img_{}_{}.png'.format(it, index),y0_inv.cpu().detach().numpy()[index,:,:,:].swapaxes(0, 2).swapaxes(0, 1))
                    avg_fid = np.mean(avg_fid)
                    logging.info('FID (different): %.3f' % avg_fid)

                fid_full_avg.append(avg_fid)
            fid_full_avg_mean = np.mean(fid_full_avg)
        except Exception as e:
            print(e)
            logging.info(e)
            fid_full_avg_mean = np.mean(fid_full_avg)
        print('************************************************************************************')
        logging.info('FULL FID (different): %.3f' % fid_full_avg_mean)
        print('************************************************************************************')
