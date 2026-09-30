"""Flow-matching noise schedule with per-frame noise levels (diffusion forcing)."""

import torch


class FlowMatchScheduler:
    """Rectified-flow schedule ``x_t = (1 - sigma) x_0 + sigma * eps``.

    Every method that takes ``timesteps`` accepts a ``[batch, frames]`` tensor, so
    each latent frame can sit at its own noise level.
    """

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0,
                 sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        start = self.sigma_min + (self.sigma_max - self.sigma_min)
        if self.extra_one_step:
            sigmas = torch.linspace(start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(start, self.sigma_min, num_inference_steps)
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            y = torch.exp(-2 * ((self.timesteps - num_inference_steps / 2) / num_inference_steps) ** 2)
            y = y - y.min()
            self.linear_timesteps_weights = y * (num_inference_steps / y.sum())

    def _ids(self, timesteps):
        return (timesteps.cpu()[..., None] - self.timesteps).abs().argmin(-1)

    @staticmethod
    def _per_frame(sigma, x):
        batch, frames = sigma.shape
        return sigma.view(batch, 1, frames, *([1] * (x.dim() - 3)))

    def step_diff_noise_level(self, model_output, timesteps, sample, to_final=False):
        """One Euler step from each frame's timestep to the next one in the schedule
        (or straight to ``t = 0`` with ``to_final``)."""
        ids = self._ids(timesteps)
        sigma = self.sigmas[ids]
        if to_final:
            sigma_next = torch.zeros_like(sigma)
        else:
            nxt = ids + 1
            sigma_next = self.sigmas[nxt.clamp(max=len(self.timesteps) - 1)]
            sigma_next = torch.where(nxt >= len(self.timesteps), 0.0, sigma_next)
        diff = (sigma_next.to(model_output.device) - sigma.to(model_output.device))
        return sample + model_output * self._per_frame(diff, model_output)

    def batch_add_frame_noise(self, original_samples, noise, timesteps):
        sigma = self._per_frame(self.sigmas[self._ids(timesteps)].to(original_samples.device),
                                original_samples)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def batch_frame_training_weight(self, timesteps):
        timesteps = timesteps.cpu()
        ids = torch.argmin((self.timesteps.view(-1, 1, 1) - timesteps[None]).abs(), dim=0)
        return self.linear_timesteps_weights[ids]
