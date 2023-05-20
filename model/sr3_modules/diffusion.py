import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from PIL import Image
from .losses import normal_kl, discretized_gaussian_log_likelihood

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        betas = betas_for_alpha_bar(
           n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    elif schedule == "cosine_orig":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        use_diff = False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.use_diff  = use_diff
        if self.use_diff:
            print("Use diff as training data type!!!!!!!!!!!!!")
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)# equation (4) variance
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(# equation (4) mean
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(# equation (4)
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):#equation (10)
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):# equation (4) q(y_t-1|y0,yt)
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t# equation (4)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        #print("&"*30,posterior_mean.shape, posterior_log_variance_clipped.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None,cam_save = False):#eq (4)
        batch_size = x.shape[0]
        #assert t.shape == ( batch_size,)
        noise_level = torch.FloatTensor(# Also Time in Unet (time encode)
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        att_cam = None
        if cam_save:
            Noise,att_cam = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level,cam_save=cam_save)
        else:
            Noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(#Equation (10)
                x, t=t, noise=Noise)#noise+low_res img to UNet input
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))# Unet output denoise img

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        #print("%"*30,x_recon.shape, x.shape,t)
        #print("%"*30,model_mean.shape, posterior_log_variance.shape)
        return model_mean, posterior_log_variance,x_recon,att_cam

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None,cam_save = False):# equation (9) p(y_t-1|yt,x)
        model_mean, model_log_variance,_,att_cam = self.p_mean_variance(#eq (4)
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x,cam_save=cam_save)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(),att_cam

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False,cam=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))#num_timesteps = 2000
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img,_ = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img,att_cam= self.p_sample(img, i, condition_x=x,cam_save = cam)# t-1 <- t
                if i % sample_inter == 0:
                    img_full = img+x if self.use_diff else img
                    ret_img = torch.cat([ret_img, img_full], dim=0)
                    #print("ATT MAP",att_cam.max(),att_cam.min(),att_cam.mean())
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):#Unconditional
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False,cam=False):# Conditional
        return self.p_sample_loop(x_in, continous,cam=cam)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):#Algo 1 line 5: get yt from y0
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (#Algo 1 line 5/ aka: yt
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None,use_vb = 0):
        x_start = x_in['HR']#original imag
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps ) # Remove +1!!!!
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(# (gamma)^0.5
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.use_diff:
            X_input =  x_start-x_in['SR']
        else:
            X_input = x_start
        x_noisy = self.q_sample(# get xt from x0
            x_start=X_input, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            #Diff = ((diff-diff.min())/(diff.max()-diff.min()))*255
            #Orig =  ((x_start[0]-x_start[0].min())/(x_start[0].max()-x_start[0].min()))*255# orig 
            #add = diff+x_in['SR'][0]
            #Add = ((add-add.min())/(add.max()-add.min()))*255#add back
            #out = torch.cat((Diff,Orig,Add),dim = 2)
            #out = torch.einsum('abc->bca',out).cpu().numpy()
            #im = Image.fromarray(out.astype(np.uint8))
            #im.save("combine.jpeg")
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        if use_vb:
            loss = self.loss_func(noise, x_recon) +  self._vb_terms_bpd( x_start=x_start,x_t=x_noisy,t=t,condition_x =x_in['SR'] ,clip_denoised=False,)["output"]
        else:
            loss = self.loss_func(noise, x_recon)
        return loss
    def mean_flat(self,tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def _vb_terms_bpd(
        self, x_start, x_t, t, condition_x = None,clip_denoised=True):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, true_log_variance_clipped = self.q_posterior(# y_t-1 from y0 and y_t
            x_start=x_start, x_t=x_t, t=t
        )
        est_mean, est_log_variance,x0_pred,_ = self.p_mean_variance(# y_t-1 from y_t only
             x=x_t, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        kl = normal_kl(# estimate true and est distribution by KL
            true_mean, true_log_variance_clipped, est_mean, est_log_variance
        )
        kl = self.mean_flat(kl) / np.log(2.0)
        #print("+"*30,x_start.shape,x_start.max(),x_start.min(),est_mean.shape,est_log_variance .shape)
        broadcast_shape = x_start.shape
        res = est_log_variance
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        Est_log_variance = res.expand(broadcast_shape)
        #Est_log_variance = torch.ones_like(x_start)*est_log_variance
        decoder_nll = -discretized_gaussian_log_likelihood(
            x=x_start, means=est_mean, log_scales=0.5 *Est_log_variance 
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = self.mean_flat(decoder_nll) / np.log(2.0)
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        #output = torch.where(t==0, decoder_nll, kl)
        if t==0:output = decoder_nll
        else: output = kl
        return {"output": output, "pred_xstart": x0_pred}
    
    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
