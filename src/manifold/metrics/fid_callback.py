"""Per-epoch unbiased 2.5D FID callback (issue #27).

Generates ``num_synth`` unconditional volumes through ``Module.sample`` (the
**slow** EMA shadow swapped in) + the held frozen VAE's decode — never the
inference Pipeline — extracts 2.5D RadImageNet features over the three
orthogonal planes, and logs the small-sample-bias-corrected Fréchet distance
``val/fid_avg`` (plus per-plane).

Fixed-sample mechanism:

- the **real** reference is a fixed validation subset, decoded once and cached;
- the **synthetic** arm re-seeds the generation noise every epoch (same small
  ``num_synth``), so only the model changes between epochs and drift is isolated
  from sampling stochasticity.

Single-GPU / rank-0 only: a multi-minute generation loop would deadlock the
other ranks at an NCCL collective, so under DDP the callback warns loudly and
skips. Configurable via an optional ``fid_eval`` block.
"""

from __future__ import annotations

import logging

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore

from .fid import frechet_distance_unbiased, get_features_2p5d

_log = logging.getLogger(__name__)


class FIDCallback(pl.Callback):
    """Per-epoch unbiased 2.5D FID on a fixed sample set (single-GPU / rank-0).

    Args:
        module: the :class:`~manifold.modules.LatentFlowModule` (its
            :meth:`~manifold.modules.LatentFlowModule.sample` generates).
        vae: the held frozen VAE; its :meth:`~manifold.AutoencoderKL.decode`
            decodes both arms (no Pipeline).
        ema_callback: the :class:`~manifold.training.DoubleEMACallback`; the slow
            shadow is swapped in around generation so reported quality reflects
            the published EMA model.
        real_latents: the FIXED real reference subset ``[N, C, D, H, W]`` (scaled
            latents — the seeded-shuffle prefix of ``val_subset_size``). Decoded
            once and cached.
        feature_net: callable ``[K, C, h, w] -> [K, D_feat]`` (the RadImageNet
            backbone, or a test fake). Injected so the callback is testable
            offline.
        latent_shape / spacing / modality / num_inference_steps / guidance_scale
            / cfg_interval: the generation recipe (defaults mirror inference).
        num_synth: synthetic volumes generated per run (re-seeded every epoch).
        every_n_epochs: run cadence (1 = every validation epoch).
        center_slices_ratio / cov_ridge: the ``fid_eval`` knobs forwarded to
            :func:`~manifold.metrics.fid.get_features_2p5d` /
            :func:`~manifold.metrics.fid.frechet_distance_unbiased`.
        seed: the re-seeded generation noise seed (fixed across epochs).
    """

    def __init__(
        self,
        *,
        module,
        vae,
        ema_callback,
        real_latents: torch.Tensor,
        feature_net,
        latent_shape,
        spacing,
        modality: int,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        cfg_interval=None,
        num_synth: int = 16,
        every_n_epochs: int = 1,
        center_slices_ratio: float = 0.5,
        cov_ridge: float = 1e-6,
        seed: int = 0,
    ):
        super().__init__()
        self.module = module
        self.vae = vae
        self.ema_callback = ema_callback
        self.real_latents = real_latents
        self.feature_net = feature_net
        self.latent_shape = tuple(latent_shape)
        self.spacing = spacing
        self.modality = int(modality)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.cfg_interval = cfg_interval
        self.num_synth = int(num_synth)
        self.every_n_epochs = int(every_n_epochs)
        self.center_slices_ratio = float(center_slices_ratio)
        self.cov_ridge = float(cov_ridge)
        self.seed = int(seed)
        self._real_planes: list[torch.Tensor] | None = None

    # -- internals -----------------------------------------------------------

    def _gated(self, trainer) -> bool:
        """Rank-0 + cadence gate; warn loudly under DDP and skip otherwise."""
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        if world > 1:
            _log.warning(
                "FIDCallback: running only on rank 0 (world_size=%d). The other "
                "ranks skip the generative FID and must NOT block on an NCCL "
                "collective here — single-GPU is the supported config.", world
            )
            if not trainer.is_global_zero:
                return False
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 1 or (epoch % self.every_n_epochs == 0):
            return True
        return False

    def _stage_eval_on_device(self) -> None:
        """Stage VAE + feature_net onto the UNet's (GPU) device for the FID phase.

        ``warm_latent_pipeline`` leaves the VAE on CPU to free VRAM for UNet
        *training*; the RadImageNet feature_net is CPU by default. During validation
        the UNet is idle, so decoding + feature extraction on GPU is feasible and far
        faster than CPU. Both are returned to CPU once the FID phase ends.
        """
        if not getattr(self, "_eval_staged", False):
            self._vae_cpu_state = {k: v.detach().clone() for k, v in self.vae.state_dict().items()}
            self.vae.to(self._device())
            if self.feature_net is not None:
                self.feature_net.to(self._device())
            self._eval_staged = True

    def _restore_eval_to_cpu(self) -> None:
        """Return VAE + feature_net to CPU after the FID phase (free VRAM for training)."""
        if getattr(self, "_eval_staged", False):
            self.vae.to("cpu")
            if hasattr(self, "_vae_cpu_state"):
                self.vae.load_state_dict(self._vae_cpu_state)
            if self.feature_net is not None:
                self.feature_net.to("cpu")
            self._eval_staged = False

    def _eval_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode for FID eval in float32 on the staged (GPU) device.

        ``norm_float16`` makes MaisiGroupNorm3D cast its output to float16
        unconditionally, so a downstream float32 conv raises a bias-type mismatch
        unless an outer autocast reconciles them -- which this validation hook cannot
        rely on. FID is an *evaluation* metric, so float32 decode is both robust and
        more correct than half precision. The latent is moved to the VAE's device.
        """
        if not getattr(self, "_norm16_disabled", False):
            for m in self.vae.modules():
                if hasattr(m, "norm_float16"):
                    m.norm_float16 = False
            self._norm16_disabled = True
        vae_device = next(self.vae.parameters()).device
        return self.vae.decode(latents.float().to(vae_device))

    @torch.no_grad()
    def _real_features(self) -> list[torch.Tensor]:
        """Decode the fixed real subset once and cache its per-plane features."""
        self.vae.eval()
        real_images = self._eval_decode(self.real_latents)
        return get_features_2p5d(real_images, self.feature_net, center_slices_ratio=self.center_slices_ratio)

    @torch.no_grad()
    def _synth_features(self) -> list[torch.Tensor]:
        """Generate num_synth volumes (slow-EMA-swapped) + held-VAE decode."""
        self.vae.eval()
        planes: list[list[torch.Tensor]] = [[], [], []]
        for i in range(self.num_synth):
            gen = torch.Generator(device=self._device()).manual_seed(self.seed + i)
            self.ema_callback.swap_in(self.module)
            try:
                latent = self.module.sample(
                    self.latent_shape,
                    self.spacing,
                    self.modality,
                    self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    cfg_interval=self.cfg_interval,
                    generator=gen,
                )
            finally:
                self.ema_callback.restore(self.module)
            image = self._eval_decode(latent)
            for axis, feats in enumerate(
                get_features_2p5d(image, self.feature_net, center_slices_ratio=self.center_slices_ratio)
            ):
                if feats.numel():
                    planes[axis].append(feats)
        return [torch.cat(p, dim=0) if p else torch.empty(0) for p in planes]

    def _device(self):
        return next(self.module.unet.parameters()).device

    # -- Lightning hook ------------------------------------------------------

    def on_validation_epoch_end(self, trainer, module) -> None:
        if not self._gated(trainer):
            return
        self._stage_eval_on_device()
        try:
            if self._real_planes is None:
                self._real_planes = self._real_features()
            synth_planes = self._synth_features()

            per_plane = {}
            total = 0.0
            n = 0
            for axis, (rp, sp) in enumerate(zip(self._real_planes, synth_planes)):
                name = ("xy", "yz", "zx")[axis]
                if rp.numel() == 0 or sp.numel() == 0 or rp.shape[0] < 2 or sp.shape[0] < 2:
                    continue
                fid = frechet_distance_unbiased(sp.float(), rp.float(), ridge=self.cov_ridge)
                per_plane[name] = float(fid)
                total += float(fid)
                n += 1
            if n:
                module.log("val/fid_avg", total / n)
                for name, val in per_plane.items():
                    module.log(f"val/fid_{name}", val)
        finally:
            self._restore_eval_to_cpu()
