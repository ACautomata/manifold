"""Per-epoch unbiased 2.5D FID callback (issue #27).

Generates ``num_synth`` unconditional volumes through ``Module.sample`` + the
held frozen VAE's decode — never the inference Pipeline — extracts 2.5D
RadImageNet features over the three orthogonal planes, and logs the small-sample-
bias-corrected Fréchet distance ``val/fid_avg`` (plus per-plane).

Two arms share the same fixed real reference and per-sample seeds:

- **slow** (``val/fid_avg``): the 0.9999 EMA shadow swapped in — the published
  model (matches hope's policy).
- **raw** (``val/fid_raw``): the raw optimizer weights, sampled without the EMA
  swap — blind to the slow EMA's convergence lag, so it tracks whether the model
  is actually learning. The best checkpoint monitors this arm.

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
            the published EMA model. ``None`` selects the no-EMA regime (e.g. GRPO
            policy post-training, #59): a single arm with no swap, logged as
            ``val/fid`` (the anti-reward-hacking selection metric, #58).
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
        log_raw_fid: also log ``val/fid_raw`` (+ per-plane) sampled from the RAW
            optimizer weights — blind to the slow EMA's convergence lag, so it
            tracks whether the model is actually learning (this is the metric
            the best checkpoint monitors). Costs one extra generation pass per
            validation epoch. The slow-EMA arm stays logged as ``val/fid_avg``.
    """

    def __init__(
        self,
        *,
        module,
        vae,
        ema_callback,
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
        log_raw_fid: bool = True,
    ):
        super().__init__()
        self.module = module
        self.vae = vae
        self.ema_callback = ema_callback
        # F5 (ADR-0017): ``real_latents`` may be ``None`` at construction (the warm
        # has moved into DataModule.setup(), so the val reference does not exist
        # until the first validation epoch). ``real_latents_source`` is a callable
        # ``() -> Tensor`` (or an object exposing ``.val_latents``) pulled lazily at
        # the first ``_real_features`` call. Both are populated before any FID math.
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
        self.log_raw_fid = bool(log_raw_fid)
        self._real_planes: list[torch.Tensor] | None = None

    # -- internals -----------------------------------------------------------

    def _gated(self, trainer) -> bool:
        """Rank-0 + cadence gate; warn loudly under DDP and skip otherwise.

        The warning is hoisted BELOW the ``is_global_zero`` guard so it fires
        only on rank 0 (L1, ADR-0016) - emitting it on every rank every val
        epoch was noise.
        """
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        if world > 1:
            if not trainer.is_global_zero:
                return False
            _log.warning(
                "FIDCallback: running only on rank 0 (world_size=%d). The other "
                "ranks skip the generative FID and must NOT block on an NCCL "
                "collective here — single-GPU is the supported config.", world
            )
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
                return
            self._fid_disabled = False
            if self.feature_net is not None:
                self.feature_net.to(self._device())
                # eval so BatchNorm uses fixed running stats (RadImageNet ResNet50
                # is BN-based). In train mode every forward updates them, so the
                # raw arm would inherit stats drifted by the real/slow arms — and
                # since the raw arm is the checkpoint monitor, that contamination
                # would distort selection. Also matches hope (net.eval().to()).
                self.feature_net.eval()
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
        """Decode the fixed real subset once and cache its per-plane features.

        F5: if ``real_latents`` was ``None`` at construction (the warm moved into
        ``DataModule.setup()``, so the val reference did not exist yet), pull it
        from the ``real_latents_source`` here - the first ``_real_features`` call
        runs inside the first validation epoch, after ``setup()`` populated it.
        """
        if self.real_latents is None and self._real_latents_source is not None:
            src = self._real_latents_source
            self.real_latents = src() if callable(src) else getattr(src, "val_latents")
        if self.real_latents is None:
            raise RuntimeError(
                "FIDCallback.real_latents is None at the first _real_features call - "
                "the DataModule.setup() warm has not populated it (F5 wiring bug)."
            )
        self.vae.eval()
        real_images = self._eval_decode(self.real_latents)
        return get_features_2p5d(real_images, self.feature_net, center_slices_ratio=self.center_slices_ratio)

    @torch.no_grad()
    def _synth_features(self, *, raw: bool = False) -> list[torch.Tensor]:
        """Generate num_synth volumes + held-VAE decode (slow-EMA-swapped).

        raw=True samples the raw optimizer weights — it skips ema.swap_in/restore
        so the result is blind to the slow EMA's convergence lag (the
        ``val/fid_raw`` arm). Both arms share the same per-sample seeds, so the
        slow-vs-raw gap reflects weights, not sampling noise.
        """
        self.vae.eval()
        planes: list[list[torch.Tensor]] = [[], [], []]
        for i in range(self.num_synth):
            gen = torch.Generator(device=self._device()).manual_seed(self.seed + i)
            if not raw:
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
                if not raw:
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
            # codex #85 P2: the lazy feature_net factory is fail-safe; if it returned
            # None (bad/corrupt cache, no network), FID is unavailable. Log the
            # monitored metrics as +inf ONCE so ModelCheckpoint(monitor='val/fid_*',
            # mode='min') does not crash on a never-logged metric (inf is never the
            # best, so selection falls through to save_last) instead of aborting the
            # run mid-fit. The online torch.hub fallback stays reachable on hosts
            # with network (no launch-time pre-disable on a missing cache).
            if getattr(self, "_fid_disabled", False):
                keys = (
                    ["val/fid"]
                    if self.ema_callback is None
                    else (["val/fid_avg"] + (["val/fid_raw"] if self.log_raw_fid else []))
                )
                for key in keys:
                    module.log(key, float("inf"))
                return
            if self._real_planes is None:
                self._real_planes = self._real_features()
            if self.ema_callback is None:
                # No-EMA regime (e.g. GRPO policy post-training, #59: the supervised-
                # decay shadows are useless under RL). One arm, no swap, logged as
                # ``val/fid`` — the anti-reward-hacking selection metric (#58).
                self._compute_and_log(module, raw=True, total_key="val/fid", plane_key="val/fid")
            else:
                # Slow-EMA(0.9999) arm — the published model (matches hope's policy).
                self._compute_and_log(
                    module, raw=False, total_key="val/fid_avg", plane_key="val/fid"
                )
                # Raw-optimizer arm — decoupled from the slow EMA's convergence lag,
                # so it tracks whether the model is actually learning. Monitored for
                # the best checkpoint (a lagging EMA otherwise hides raw progress).
                if self.log_raw_fid:
                    self._compute_and_log(
                        module, raw=True, total_key="val/fid_raw", plane_key="val/fid_raw"
                    )
        finally:
            self._restore_eval_to_cpu()

    def _compute_and_log(self, module, *, raw: bool, total_key: str, plane_key: str) -> None:
        """Generate one arm + log its per-plane unbiased FID vs the cached real.

        ``total_key`` is the metric logged for the arm's mean FID; ``plane_key`` is
        the prefix for the per-plane diagnostics (``f"{plane_key}_{xy,yz,zx}"``).
        The JiT two-arm regime uses ``val/fid_avg`` (+ ``val/fid_{axis}``) for the
        slow-EMA arm and ``val/fid_raw`` (+ ``val/fid_raw_{axis}``) for the raw arm;
        the no-EMA regime (#58) uses a single ``val/fid`` (+ ``val/fid_{axis}``).
        ``raw=True`` skips the EMA swap (absent under no-EMA). No-op when no plane
        has ≥2 features.
        """
        synth_planes = self._synth_features(raw=raw)
        per_plane: dict[str, float] = {}
        total = 0.0
        n = 0
        for axis, (rp, sp) in enumerate(zip(self._real_planes, synth_planes)):
            name = ("xy", "yz", "zx")[axis]
            if rp.numel() == 0 or sp.numel() == 0 or rp.shape[0] < 2 or sp.shape[0] < 2:
                continue
            fid = float(frechet_distance_unbiased(sp.float(), rp.float(), ridge=self.cov_ridge))
            per_plane[name] = fid
            total += fid
            n += 1
        if not n:
            return
        module.log(total_key, total / n)
        for name, val in per_plane.items():
            module.log(f"{plane_key}_{name}", val)
