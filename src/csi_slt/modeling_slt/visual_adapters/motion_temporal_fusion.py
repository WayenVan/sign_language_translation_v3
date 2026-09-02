"""Window-level motion fusion, shared by the adapters that use it.

Extracted for the same reason as ``next_frame_fusion``: identical copies of a
module in separate files drift apart silently, because nothing links them and
no test compares them. Two byte-identical copies of this class existed when it
was pulled out here.

The fusion replaces a window's fixed temporal mean with that mean plus a gated
residual computed from the window's adjacent differences, so a gate of zero
recovers the mean exactly.

``spatiotemporal_next_frame_motion_adapter`` keeps its own copy on purpose --
its docstring says so and a test asserts it -- so it is deliberately not a
consumer of this module.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class MotionTemporalFusion(nn.Module):
    """Fuse ``s`` consecutive frame features into one, mean plus motion.

    ::

        m = mean(x_t .. x_{t+s-1})                     # static content
        d = [x_{t+1} - x_t, ..., x_{t+s-1} - x_{t+s-2}]  # motion
        y = m + sigmoid(g) * LayerNorm(SwiGLU(LayerNorm(d)))

    Two deliberate departures from the older ``TemporalShuffleAdapter``:

    - **The motion path sees only the differences.**  That module fed it
      ``[x_t, ..., x_{t+s-1}, d]``, which is rank-deficient: ``d`` is a linear
      function of the frames beside it, so a linear layer on the concatenation
      spans exactly the space that ``[x_t, ..., x_{t+s-1}]`` already spans.  The
      redundant block cost a third of the largest weight matrix and bought no
      expressive power.  Splitting the window into ``m`` (handled by the
      residual) and ``d`` (handled here) is a change of basis over the same
      space, so nothing is lost and the input width drops from ``D*(2s-1)`` to
      ``D*(s-1)``.
    - **The branch output is normalized before the gate.**  ``sigmoid(g) * f(x)``
      alone is not identifiable -- ``f`` can grow its own weights to offset a
      small gate, which is exactly what happened in the earlier runs: their
      gates sat at their 0.12 initialization for 80k steps while the branch's
      input projection grew fivefold.  Normalizing first pins the branch scale,
      so ``sigmoid(g)`` becomes a readable measure of how much motion the model
      actually uses, comparable across runs.

    ``gate_init`` is deliberately not zero.  A gate that starts at exactly zero
    multiplies the whole branch by zero, so every parameter behind it receives
    zero gradient and has to bootstrap through a single scalar; sigmoid(-2) is
    about 0.12, small enough to start near the mean-pooling baseline but large
    enough that the branch trains from the first step.

    The residual ``m`` carries whatever scale the backbone's features have,
    while the normalized branch is unit-scale, so ``sigmoid(g)`` is a ratio
    against that scale rather than an absolute fraction.  The adapter's own
    LayerNorm runs after this module and absorbs the combined scale.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale_factor: int = 2,
        motion_hidden_dim: int | None = None,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        _validate_dimension("hidden_dim", hidden_dim)
        _validate_dimension("scale_factor", scale_factor)
        if scale_factor < 2:
            raise ValueError(
                "scale_factor must be at least 2 for a temporal difference"
            )
        if motion_hidden_dim is not None:
            _validate_dimension("motion_hidden_dim", motion_hidden_dim)
        if isinstance(gate_init, bool) or not isinstance(gate_init, (int, float)):
            raise TypeError("gate_init must be a real number")
        if not math.isfinite(float(gate_init)):
            raise ValueError("gate_init must be finite")

        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.motion_hidden_dim = (
            hidden_dim if motion_hidden_dim is None else motion_hidden_dim
        )

        difference_dim = hidden_dim * (scale_factor - 1)
        self.motion_norm = nn.LayerNorm(difference_dim)
        # SwiGLU: one projection produces both the gate and the value half.
        self.motion_in = nn.Linear(difference_dim, self.motion_hidden_dim * 2)
        self.motion_out = nn.Linear(self.motion_hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for projection in (self.motion_in, self.motion_out):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            nn.init.zeros_(projection.bias)

    @property
    def motion_weight(self) -> float:
        """Current gate value: how much motion rides on the mean."""
        return float(torch.sigmoid(self.gate).item())

    def forward(self, frame_features: Tensor) -> Tensor:
        """Fuse ``[sum(T), D]`` frame features into ``[sum(T) / s, D]``.

        Every video length is a multiple of ``scale_factor``, so video
        boundaries in the packed batch land on window boundaries and a window
        can never span two videos.  Windowing the packed tensor directly is
        therefore equivalent to splitting per video first, and it avoids the
        device synchronization that reading the lengths would cost.
        """
        if frame_features.ndim != 2:
            raise ValueError(
                "frame_features must have shape [sum(T), D], got "
                f"{tuple(frame_features.shape)}"
            )
        if frame_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"frame feature dimension must be {self.hidden_dim}, got "
                f"{frame_features.shape[-1]}"
            )

        windows = frame_features.unflatten(0, (-1, self.scale_factor))
        static = windows.mean(dim=1)
        differences = (windows[:, 1:] - windows[:, :-1]).flatten(start_dim=1)

        hidden = self.motion_in(self.motion_norm(differences))
        gate_half, value_half = hidden.chunk(2, dim=-1)
        motion = self.motion_out(F.silu(gate_half) * value_half)

        motion_weight = torch.sigmoid(self.gate).to(dtype=static.dtype)
        return static + motion_weight * self.gate_norm(motion)
