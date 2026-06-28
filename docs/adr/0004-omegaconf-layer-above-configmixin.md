# OmegaConf experiment-config layer above the JSON component config

manifold has two config systems by design. **Experiment config** is OmegaConf: the
run-driver composing env (paths) → train/inference recipe → network-construction
kwargs, top-level replace with `_base_` inheritance, required paths as `???`
(fail-fast) and CLI/dotlist overrides. It builds components at launch. **Component
config** is the JSON `config.json` each component writes via
`register_to_config`/`ConfigMixin` and `from_pretrained`/`save_pretrained`
round-trips — the diffusers-style persistence contract for a trained component,
independent of how it was launched.

OmegaConf never touches load/save: the converter and the (deferred) train/infer
entrypoints read OmegaConf to *build* components; `save_pretrained` writes JSON.

## Considered options

- **Replace ConfigMixin with OmegaConf** (one system end-to-end). Rejected: couples
  the persistence contract to launch-time YAML and rewrites the tested load/save
  path and persistence tests; `from_pretrained` must round-trip exactly a saved
  component without depending on how it was launched.
- **OmegaConf for paths/recipe only, hand-written JSON network config** (like the
  converter's `--unet-config` today). Rejected: network-construction knobs get no
  `???`/CLI override and "完全重构" is half-done — and the network config is exactly
  what must be overridable per run (VAE/UNet widths, attention levels).

## Network config is plain-kwarg YAML

Network construction is plain-kwarg OmegaConf YAML (manifold wrapper kwargs,
`${...}` interpolation for shared `spatial_dims`/`image_channels`/`latent_channels`)
— **not** MONAI `_target_`/`@name` bundles. A MONAI `ConfigParser` would instantiate
MAISI directly and bypass the wrappers' timestep-scaling / spacing×1e2 / scale_factor
logic (ADR-0001). The plain-kwarg form forces the wrappers to pass through the knobs
hope's MONAI configs carry that MAISI defaults differ on (VAE `norm_float16`,
`dim_split`; UNet `resblock_updown`, `include_fc`, `use_flash_attention`, per-level
`num_head_channels`) — required to load hope checkpoints and to match decode
numerics.
