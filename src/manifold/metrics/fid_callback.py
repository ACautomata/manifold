"""Per-epoch unbiased 2.5D FID callback (issue #27).

Generates ``num_synth`` unconditional volumes through ``Module.sample`` + the
held frozen VAE's decode — never the inference Pipeline — extracts 2.5D
RadImageNet features over the three orthogonal planes, and logs the
small-sample-bias-corrected Fréchet distance ``val/fid`` (plus per-plane).

Generation samples the **live (raw) optimizer weights** — ``Module.sample``
shares ``self.unet`` with training, so the reported FID reflects the weights
being optimized, with no EMA swap (EMA training was removed; ADR-0006). This is
the anti-reward-hacking selection metric (#58): a single arm, logged as
``val/fid`` (+ per-plane ``val/fid_{xy,yz,zx}``).

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

from .fid import (
    frechet_from_moments,
    features_to_sufficient_stats,
    moments_from_sufficient_stats,
    get_features_2p5d,
)


_log = logging.getLogger(__name__)


class FIDCallback(pl.Callback):
    """Per-epoch unbiased 2.5D FID on a fixed sample set (single-GPU / rank-0).

    Args:
        module: the :class:`~manifold.modules.LatentFlowModule` (its
            :meth:`~manifold.modules.LatentFlowModule.sample` generates).
        vae: the held frozen VAE; its :meth:`~manifold.AutoencoderKL.decode`
            decodes the generated latents (no Pipeline).
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
        real_latents=None,
        real_latents_source=None,
        feature_net=None,
        feature_net_factory=None,
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
        # F5 (ADR-0017): ``real_latents`` may be ``None`` at construction (the warm
        # has moved into DataModule.setup(), so the val reference does not exist
        # until the first validation epoch). ``real_latents_source`` is a callable
        # ``() -> Tensor`` (or an object exposing ``.val_latents``) pulled lazily at
        # the first ``_real_moments`` call. Both are populated before any FID math.
        self._real_latents_source = real_latents_source
        self.real_latents = real_latents
        self.feature_net = feature_net
        # L3 (ADR-0016): lazy RadImageNet build. ``_stage_eval_on_device`` is
        # rank-0-gated, so a factory is only invoked on rank 0 - non-root ranks do
        # no ``torch.hub``/disk load (saves ~100 MB × (N-1)). Construction is the
        # only place the backbone was eagerly built on every rank pre-PG.
        self.feature_net_factory = feature_net_factory
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
        self._real_moments_cache: list | None = None
        self._feat_dim: int | None = None

    # -- internals -----------------------------------------------------------

    def _gated(self, trainer) -> bool:
        """Cadence gate only. All ranks generate + decode + extract features for
        their own shard and per-plane sufficient statistics are all-reduced in
        :meth:`_planes_to_global_moments` for the exact global FID (ADR-0025). The
        prior rank-0 / single-GPU-only gate is removed.
        """
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
            # Mark staged BEFORE any early return (codex #85 P2): the ``finally`` in
            # on_validation_epoch_end calls _restore_eval_to_cpu(), which only
            # restores the VAE to CPU when _eval_staged is True. A skip-path return
            # before this flag would leave the full VAE resident on the training GPU
            # for the rest of the run (the VRAM pressure the skip is meant to avoid).
            self._eval_staged = True
            # L3 + codex #85 P2: lazy feature_net build on the rank-0-gated stage
            # path (the only place it is used). A factory defers the ~100 MB
            # ``torch.hub`` load so non-root ranks never touch it; a direct
            # ``feature_net`` is honored as-is. The call is FAIL-SAFE here (not just
            # in main's factory): a raising factory (a bad/corrupt cache, a version
            # mismatch, or a direct caller's non-fail-safe factory) is caught ->
            # feature_net stays None -> FID is skipped gracefully
            # (on_validation_epoch_end logs a sentinel so the checkpoint monitor does
            # not crash on a never-logged metric) instead of aborting training mid-fit.
            if self.feature_net is None and self.feature_net_factory is not None:
                try:
                    self.feature_net = self.feature_net_factory()
                except Exception:  # pragma: no cover - backbone load failure
                    _log.warning("RadImageNet backbone build failed; FID will be skipped.", exc_info=True)
                    self.feature_net = None
            if self.feature_net is None:
                self._fid_disabled = True
                return  # _eval_staged=True -> the finally restores the VAE to CPU.
            self._fid_disabled = False
            if self.feature_net is not None:
                self.feature_net.to(self._device())
                # eval so BatchNorm uses fixed running stats (RadImageNet ResNet50
                # is BN-based). In train mode every forward updates them, so the
                # raw arm would inherit stats drifted by the real/slow arms — and
                # since the raw arm is the checkpoint monitor, that contamination
                # would distort selection. Also matches hope (net.eval().to()).
                self.feature_net.eval()
            # ADR-0025: the feature dim is needed to size zero sufficient-stats
            # buffers for empty per-plane shards (symmetric all_reduce). Probed
            # once on the staged device; deterministic across ranks.
            if self._feat_dim is None:
                with torch.no_grad():
                    self._feat_dim = int(self.feature_net(
                        torch.zeros(1, 1, 64, 64, device=self._device())
                    ).shape[1])
            # _eval_staged was set right after staging the VAE (above), so the
            # finally's _restore_eval_to_cpu() always restores it - even on the
            # skip-path early return.

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
    def _rank_world(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
        return 0, 1

    def _planes_to_global_moments(
        self, planes: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor, int] | None]:
        """Per-plane local features -> all-reduce sufficient stats -> global (mu, sigma, n).

        Every rank enters one all_reduce per plane (symmetric over the always-3
        planes), contributing zero stats for an empty local shard so the collective
        cannot deadlock; ``_feat_dim`` (probed once at stage time) sizes the zero
        buffers. A plane with a global count < 2 yields None (FID undefined).
        """
        _, world = self._rank_world()
        d = self._feat_dim
        dev = self._device()
        out: list[tuple[torch.Tensor, torch.Tensor, int] | None] = []
        for feats in planes:
            if feats.numel() == 0 or feats.shape[0] < 2:
                sum_x = torch.zeros(d, device=dev, dtype=torch.float32)
                sum_xxT = torch.zeros(d, d, device=dev, dtype=torch.float32)
                n = 0
            else:
                sum_x, sum_xxT, n = features_to_sufficient_stats(feats.float())
            if world > 1:
                torch.distributed.all_reduce(sum_x)
                torch.distributed.all_reduce(sum_xxT)
                n_t = torch.tensor([float(n)], device=dev, dtype=torch.float32)
                torch.distributed.all_reduce(n_t)
                n = int(n_t.item())
            if n >= 2:
                mu, sigma, _ = moments_from_sufficient_stats(sum_x, sum_xxT, n)
                out.append((mu, sigma, n))
            else:
                out.append(None)
        return out

    @torch.no_grad()
    def _real_moments(self) -> list[tuple[torch.Tensor, torch.Tensor, int] | None]:
        """Decode this rank's strided shard of the fixed real subset once; all-reduce
        per-plane sufficient stats; cache the global (mu, sigma, n). F5: real_latents
        may be None until the DataModule warm populates it (pulled lazily here)."""
        if self.real_latents is None and self._real_latents_source is not None:
            src = self._real_latents_source
            self.real_latents = src() if callable(src) else getattr(src, "val_latents")
        if self.real_latents is None:
            raise RuntimeError(
                "FIDCallback.real_latents is None at the first _real_moments call - "
                "the DataModule.setup() warm has not populated it (F5 wiring bug)."
            )
        rank, world = self._rank_world()
        shard = self.real_latents[rank::world]
        self.vae.eval()
        real_images = self._eval_decode(shard)
        planes = get_features_2p5d(real_images, self.feature_net, center_slices_ratio=self.center_slices_ratio)
        return self._planes_to_global_moments(planes)

    @torch.no_grad()
    def _synth_moments(self) -> list[tuple[torch.Tensor, torch.Tensor, int] | None]:
        """Generate this rank's rank-strided slice of num_synth (seeds ``seed + i`` for
        ``i % world == rank`` so the global synth set is the union, not world x
        rank-0), decode, extract features, all-reduce -> global (mu, sigma, n).
        Generation samples the live optimizer weights (no EMA swap; EMA removed)."""
        self.vae.eval()
        rank, world = self._rank_world()
        per_plane: list[list[torch.Tensor]] = [[], [], []]
        for i in range(rank, self.num_synth, world):
            gen = torch.Generator(device=self._device()).manual_seed(self.seed + i)
            latent = self.module.sample(
                self.latent_shape,
                self.spacing,
                self.modality,
                self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                cfg_interval=self.cfg_interval,
                generator=gen,
            )
            image = self._eval_decode(latent)
            for axis, feats in enumerate(
                get_features_2p5d(image, self.feature_net, center_slices_ratio=self.center_slices_ratio)
            ):
                if feats.numel():
                    per_plane[axis].append(feats)
        planes = [torch.cat(p, dim=0) if p else torch.empty(0) for p in per_plane]
        return self._planes_to_global_moments(planes)

    def _device(self):
        return next(self.module.unet.parameters()).device

    # -- Lightning hook ------------------------------------------------------

    def on_validation_epoch_end(self, trainer, module) -> None:
        if not self._gated(trainer):
            return
        self._stage_eval_on_device()
        try:
            # codex #85 P2: a failed/absent feature_net logs +inf so the checkpoint
            # monitor (mode='min') falls through to save_last instead of crashing.
            if getattr(self, "_fid_disabled", False):
                module.log("val/fid", float("inf"))
                return
            if self._real_moments_cache is None:
                self._real_moments_cache = self._real_moments()
            self._compute_and_log(module, total_key="val/fid", plane_key="val/fid")
        finally:
            self._restore_eval_to_cpu()

    def _compute_and_log(self, module, *, total_key: str, plane_key: str) -> None:
        """Generate one arm, all-reduce its per-plane sufficient stats vs the cached
        real moments, log the global per-plane unbiased FID. No-op on planes with
        <2 global samples in either set (the single raw arm; ADR-0006)."""
        synth = self._synth_moments()
        per_plane: dict[str, float] = {}
        total = 0.0
        counted = 0
        for axis, (real_m, synth_m) in enumerate(zip(self._real_moments_cache, synth)):
            if real_m is None or synth_m is None:
                continue
            mu_r, sigma_r, n_r = real_m
            mu_g, sigma_g, n_g = synth_m
            fid = float(frechet_from_moments(mu_g, mu_r, sigma_g, sigma_r, n_g, n_r, ridge=self.cov_ridge))
            name = ("xy", "yz", "zx")[axis]
            per_plane[name] = fid
            total += fid
            counted += 1
        if not counted:
            return
        module.log(total_key, total / counted)
        for name, val in per_plane.items():
            module.log(f"{plane_key}_{name}", val)
