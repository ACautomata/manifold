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

Pipeline stages are delegated to composable helper objects (ADR-0030):
:class:`VramStage`, :class:`FixedSampleRollout`, :class:`LatentDecoder`,
:class:`FeatureExtractor`, :class:`SufficientStatsReducer`. ``FIDCallback``
stays one callback owning one metric and sequences them in one call stack.

Collective-count invariant (ADR-0030 hardening): before any reduction-bearing
phase, an error flag is all-reduced (MAX) so every rank takes the same abort
branch together — a rank-local exception cannot cause one rank to skip a
collective while others block in it.
"""

from __future__ import annotations

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore

from manifold.metrics.fid.decoder import LatentDecoder
from manifold.metrics.fid.extractor import FeatureExtractor
from manifold.metrics.fid.math import features_to_sufficient_stats, frechet_from_moments
from manifold.metrics.fid.reducer import SufficientStatsReducer
from manifold.metrics.fid.rollout import FixedSampleRollout
from manifold.metrics.fid.vram import VramStage


class FIDCallback(pl.Callback):
    """Per-epoch unbiased 2.5D FID on a fixed sample set.

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
        center_slices_ratio / cov_ridge: the ``fid`` knobs forwarded to
            :func:`~manifold.metrics.fid.math.get_features_2p5d` /
            :func:`~manifold.metrics.fid.math.frechet_from_moments`.
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
        # F5 (ADR-0017): ``real_latents`` may be ``None`` at construction.
        self._real_latents_source = real_latents_source
        self.real_latents = real_latents
        self.feature_net = feature_net
        # L3 (ADR-0016): lazy RadImageNet build. The factory is invoked inside
        # VramStage on every rank (all-rank FID under ADR-0025).
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
        :class:`SufficientStatsReducer` for the exact global FID (ADR-0025). The
        prior rank-0 / single-GPU-only gate is removed.
        """
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 1 or (epoch % self.every_n_epochs == 0):
            return True
        return False

    def _device(self):
        return next(self.module.unet.parameters()).device

    # -- collective-count helpers (ADR-0030 hardening) -----------------------

    @staticmethod
    def _all_reduce_flag(flag_t: torch.Tensor) -> bool:
        """All-reduce a flag tensor (MAX); return True if any rank has flag > 0.

        A no-op under single-process (returns the raw tensor value). Under DDP,
        every rank enters one all_reduce — the count and order of collectives is
        identical on every rank in every code path (collective-count invariant).
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag_t, op=torch.distributed.ReduceOp.MAX)
        return bool(flag_t.item())

    def _make_error_flag(self, has_error: bool) -> torch.Tensor:
        """Return a 1-element float32 tensor on the UNet's device for ``_all_reduce_flag``."""
        return torch.tensor(
            [1.0 if has_error else 0.0],
            device=self._device(), dtype=torch.float32,
        )

    # -- real / synth moments orchestration ----------------------------------

    def _ensure_real_latents(self) -> None:
        """F5: lazy pull of ``real_latents`` from the datamodule (post-setup)."""
        if self.real_latents is None and self._real_latents_source is not None:
            src = self._real_latents_source
            self.real_latents = src() if callable(src) else getattr(src, "val_latents")
        if self.real_latents is None:
            raise RuntimeError(
                "FIDCallback.real_latents is None at the first _real_moments call - "
                "the DataModule.setup() warm has not populated it (F5 wiring bug)."
            )

    @torch.no_grad()
    def _real_planes(
        self, decoder: LatentDecoder, extractor: FeatureExtractor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Decode + extract real features; return three ``[M_axis, D_feat]``
        tensors (one per plane), or three ``empty(0)`` for an empty shard.

        Does NOT reduce — the caller adds an error rendezvous before calling
        the reducer, so a rank-local decode/extraction failure cannot cause
        peers to block in a missing all_reduce.
        """
        self._ensure_real_latents()
        rank, world = self._rank_world()
        shard = self.real_latents[rank::world]
        if shard.shape[0] == 0:
            return [torch.empty(0) for _ in range(3)]
        self.vae.eval()
        real_images = decoder(shard)
        return extractor(real_images)

    @torch.no_grad()
    def _synth_planes(
        self, rollout: FixedSampleRollout, decoder: LatentDecoder,
        extractor: FeatureExtractor, device: torch.device,
    ) -> list[torch.Tensor]:
        """Generate this rank's strided synth latents, decode, extract features.

        Returns three ``[M_axis, D_feat]`` tensors — one per plane.
        """
        per_plane: list[list[torch.Tensor]] = [[], [], []]
        latents = rollout(device)
        for latent in latents:
            image = decoder(latent)
            for axis, feats in enumerate(extractor(image)):
                if feats.numel():
                    per_plane[axis].append(feats)
        return [torch.cat(p, dim=0) if p else torch.empty(0) for p in per_plane]

    @torch.no_grad()
    def _rank_world(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
        return 0, 1

    # -- Lightning hook ------------------------------------------------------

    def on_validation_epoch_end(self, trainer, module) -> None:
        if not self._gated(trainer):
            return

        device = self._device()

        # -- Stage VAE + feature_net (may raise rank-locally). ---------------
        # Python does NOT call ``__exit__`` when ``__enter__`` raises, so
        # lifecycle is managed manually with a stage-error rendezvous before
        # any collective (collective-count invariant).
        local_stage_error = False
        stage = VramStage(
            self.vae,
            feature_net=self.feature_net,
            feature_net_factory=self.feature_net_factory,
            device_fn=self._device,
            feat_dim=self._feat_dim,
        )
        try:
            stage.__enter__()
        except Exception:
            local_stage_error = True

        # Rendezvous #1: stage error flag — first collective, symmetric for
        # ALL ranks. If ANY rank failed during staging, EVERY rank returns
        # after this single collective (no divergence).
        if self._all_reduce_flag(self._make_error_flag(local_stage_error)):
            if stage._staged:
                stage.__exit__(None, None, None)
            module.log("val/fid", float("inf"))
            return

        # All ranks staged successfully — manual lifecycle.
        try:
            # Rendezvous #2 (ex-#1): disabled flag so all ranks agree on skip.
            if self._all_reduce_flag(
                torch.tensor(
                    [1.0 if stage.fid_disabled else 0.0],
                    device=device, dtype=torch.float32,
                )
            ):
                module.log("val/fid", float("inf"))
                return

            # Update the callback from the stage (factory may have built
            # feature_net, feat_dim may have been probed).
            self.feature_net = stage.feature_net
            if self._feat_dim is None:
                self._feat_dim = stage.feat_dim

            decoder = LatentDecoder(self.vae)
            extractor = FeatureExtractor(
                self.feature_net,
                center_slices_ratio=self.center_slices_ratio,
            )
            rollout = FixedSampleRollout(
                module=self.module,
                latent_shape=self.latent_shape,
                spacing=self.spacing,
                modality=self.modality,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                cfg_interval=self.cfg_interval,
                num_synth=self.num_synth,
                seed=self.seed,
            )
            reducer = SufficientStatsReducer(self._feat_dim)

            # -- Real moments (once, cached). --------------------------------
            # Decode/extract + stats computation is fallible — error
            # rendezvous before reduction so a rank-local failure cannot
            # orphan peers in the reducer's all_reduce (codex #171 P1).
            if self._real_moments_cache is None:
                local_real_error = False
                try:
                    real_planes = self._real_planes(
                        decoder, extractor, device,
                    )
                    d = self._feat_dim
                    real_stats = []
                    for p in real_planes:
                        if p.numel() == 0:
                            # Empty shard → correctly-sized zero stats
                            # (codex #171 P1 round 2).
                            real_stats.append((
                                torch.zeros(d, device=device,
                                            dtype=torch.float32),
                                torch.zeros(d, d, device=device,
                                            dtype=torch.float32),
                                0,
                            ))
                        else:
                            real_stats.append(
                                features_to_sufficient_stats(p.float()),
                            )
                    del real_planes  # release before the synthetic phase
                except Exception:
                    local_real_error = True
                    real_stats = None

                # Rendezvous #3: real preproc error before reducer.
                if self._all_reduce_flag(self._make_error_flag(local_real_error)):
                    module.log("val/fid", float("inf"))
                    return

                if real_stats is not None:
                    self._real_moments_cache = reducer(real_stats, device)
                else:
                    d = self._feat_dim
                    zero_s = (
                        torch.zeros(d, device=device, dtype=torch.float32),
                        torch.zeros(d, d, device=device, dtype=torch.float32),
                        0,
                    )
                    self._real_moments_cache = reducer(
                        [zero_s for _ in range(3)], device,
                    )
                # Release preproc buffers after caching (codex #171 P2).
                del real_stats

            # -- Synth moments + log. ----------------------------------------
            # Generate, decode, extract, AND compute sufficient stats under
            # error capture — ``features_to_sufficient_stats`` (float
            # conversion + ``features.T @ features``) is fallible and must
            # complete before the error rendezvous (codex #171 P1). Empty
            # feature tensors (num_synth < world_size) are converted to
            # correctly-sized zero stats before the rendezvous (P1 round 2).
            local_error = False
            try:
                synth_planes = self._synth_planes(
                    rollout, decoder, extractor, device,
                )
                d = self._feat_dim
                synth_stats = []
                for p in synth_planes:
                    if p.numel() == 0:
                        synth_stats.append((
                            torch.zeros(d, device=device,
                                        dtype=torch.float32),
                            torch.zeros(d, d, device=device,
                                        dtype=torch.float32),
                            0,
                        ))
                    else:
                        synth_stats.append(
                            features_to_sufficient_stats(p.float()),
                        )
            except Exception:
                local_error = True
                synth_stats = None

            # Rendezvous #4 (ex-#2): error flag before reducer's all_reduce.
            if self._all_reduce_flag(self._make_error_flag(local_error)):
                module.log("val/fid", float("inf"))
                return

            # Reduce (no fallible work — stats already computed).
            if synth_stats is not None:
                synth = reducer(synth_stats, device)
            else:
                d = self._feat_dim
                zero_s = (
                    torch.zeros(d, device=device, dtype=torch.float32),
                    torch.zeros(d, d, device=device, dtype=torch.float32),
                    0,
                )
                synth = reducer([zero_s for _ in range(3)], device)

            self._compute_and_log(module, synth,
                                  total_key="val/fid", plane_key="val/fid")
        finally:
            stage.__exit__(None, None, None)

    # -- compute + log -------------------------------------------------------

    def _compute_and_log(
        self, module, synth: list[tuple[torch.Tensor, torch.Tensor, int] | None],
        *, total_key: str, plane_key: str,
    ) -> None:
        """For each plane, compute unbiased FID vs cached real moments; log the
        global per-plane unbiased FID. No-op on planes with <2 global samples."""
        per_plane: dict[str, float] = {}
        total = 0.0
        counted = 0
        for axis, (real_m, synth_m) in enumerate(zip(self._real_moments_cache, synth)):
            if real_m is None or synth_m is None:
                continue
            mu_r, sigma_r, n_r = real_m
            mu_g, sigma_g, n_g = synth_m
            fid = float(frechet_from_moments(
                mu_g, mu_r, sigma_g, sigma_r, n_g, n_r, ridge=self.cov_ridge,
            ))
            name = ("xy", "yz", "zx")[axis]
            per_plane[name] = fid
            total += fid
            counted += 1
        if not counted:
            return
        module.log(total_key, total / counted)
        for name, val in per_plane.items():
            module.log(f"{plane_key}_{name}", val)
