# CLAUDE.md

Manifold: pretraining & medical-imaging experiments built on stable-pretraining and MONAI.
The project follows the diffusers architecture: pipeline, scheduler, and models.

## Agent skills

### Issue tracker

GitHub Issues (no external-PR triage). See `docs/agents/issue-tracker.md`.

### Triage labels

All five canonical roles use their default strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo (`CONTEXT.md` + `docs/adr/` at root). See `docs/agents/domain.md`.

<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

- Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
