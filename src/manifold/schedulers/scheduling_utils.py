"""Scheduler base class, mimicking ``diffusers.SchedulerMixin``.

A manifold Scheduler owns the rectified-flow transport and the reverse-time step
(see ADR-0001). The base is a thin marker carrying ``prediction_type`` plus the
:class:`~manifold.configuration.ConfigMixin` persistence; it does not subclass
``diffusers.SchedulerMixin``.
"""

from __future__ import annotations

from ..configuration import ConfigMixin


class SchedulerMixin(ConfigMixin):
    """Base for manifold schedulers.

    ``prediction_type`` declares what the coupled model predicts. For the JiT
    x0-denoiser it is ``"sample"`` (the clean latent x0); the Scheduler derives
    the rectified-flow velocity from that prediction internally.
    """

    #: What the coupled model's output represents. Overridden by subclasses.
    prediction_type: str = "sample"
