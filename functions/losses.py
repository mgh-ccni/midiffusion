import torch
from models.conditioner import statistical_align_normal, statistical_align


def noise_estimation_loss_sdedit(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor, # Time step
                          e: torch.Tensor, # Noise, same shape as x0
                          b: torch.Tensor, keepdim=False, config=None):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1) # Noise scale,  Ho et al. (2020) consider a sequence of positive noise scales
    x = x0[:,0,:,:].unsqueeze(1) * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(x,t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def noise_estimation_loss2(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,  # Time step
                          e: torch.Tensor,  # Noise, same shape as x0
                          b: torch.Tensor, keepdim=False, config=None, ch_sz=1, sig=2.5):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1,
                                                       1)  # Noise scale,  Ho et al. (2020) consider a sequence of positive noise scales
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    x_, _ = statistical_align(x0, x,
                           patch_size=config.model.geo_patch_sz, ch_sz=ch_sz, sig=sig)  # x_: cross-correlation aligned x; sim: self-similarity matrix
    _, sim = statistical_align(x0, x0,
                            patch_size=config.model.geo_patch_sz, ch_sz=ch_sz, sig=sig)  # x_: cross-correlation aligned x; sim: self-similarity matrix

    x_ = torch.cat((x_, sim), dim=1)

    output = model(x, x_, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_loss_time(model,
                                     x0: torch.Tensor,
                                     t: torch.LongTensor,  # Time step
                                     e: torch.Tensor,  # Noise, same shape as x0
                                     b: torch.Tensor, keepdim=False, config=None):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1,
                                                       1)  # Noise scale,  Ho et al. (2020) consider a sequence of positive noise scales
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    time_delay = 1
    tt = t + time_delay
    tt[tt >= config.diffusion.num_diffusion_timesteps] = config.diffusion.num_diffusion_timesteps - 1

    a_t_future = (1 - b).cumprod(dim=0).index_select(0, tt).view(-1, 1, 1, 1)

    x_t_future = x0 * a_t_future.sqrt() + e * (1.0 - a_t_future).sqrt()
    if x0.shape[1] == 3:

        x0 = (x0[:, 0, :, :] * 0.299 + x0[:, 1, :, :] * 0.587 + x0[:, 2, :, :] * 0.114).unsqueeze(1)

        x_, sim_crox = statistical_align_normal(x0, x_t_future,
                               patch_size=config.model.geo_patch_sz, sig=config.model.MIsigma)  # x_: cross-correlation aligned x; sim: self-similarity matrix
        _, sim_self = statistical_align_normal(x0, x0,
                                patch_size=config.model.geo_patch_sz, sig=config.model.MIsigma)  # x_: cross-correlation aligned x; sim: self-similarity matrix

    else:
        x_, sim_crox = statistical_align(x0, x_t_future,
                               patch_size=config.model.geo_patch_sz, sig=config.model.MIsigma)  # x_: cross-correlation aligned x; sim: self-similarity matrix
        _, sim_self = statistical_align(x0, x0,
                                patch_size=config.model.geo_patch_sz, sig=config.model.MIsigma)  # x_: cross-correlation aligned x; sim: self-similarity matrix

    x_ = torch.cat((sim_crox, sim_self), dim=1)
    #print(x.shape)
    output = model(x, x_, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss_time,  # noise_estimation_loss,noise_estimation_loss_time_delay
    'sdedit': noise_estimation_loss_sdedit,  # noise_estimation_loss,noise_estimation_loss_time_delay
}
