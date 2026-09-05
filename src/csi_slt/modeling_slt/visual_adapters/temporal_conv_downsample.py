"""Boundary-safe temporal downsampling by a learnable per-channel filter.

The boundary-safe temporal mean used across the spatiotemporal adapters --
``features.unflatten(0, (-1, s)).mean(dim=1)`` -- is the special case of a
depthwise, stride-``s`` ``Conv1d`` whose kernel is exactly ``s`` wide and whose
weights are frozen at ``1/s``. This module keeps that geometry but lets the
weights train, and optionally widens the kernel so a window can borrow context
from its neighbours::

    kernel_size = scale_factor + 2 * radius
    padding     = radius                        # both sides, always exact

``radius`` -- not ``kernel_size`` -- is the parameter exposed, because it is
the only one of the two that can never produce an invalid configuration. A
downsampling conv must hit an exact output length of ``T / scale_factor`` to
keep every consumer downstream (token counts, ``video_token_scale``, the CTC
head) unchanged, and for ``stride > 1`` that forces ``padding = (kernel_size -
scale_factor) / 2``: an arbitrary ``kernel_size`` can make this non-integer
(no symmetric solution exists at all, not merely an inconvenient one) whenever
``kernel_size - scale_factor`` is odd. Parameterizing by ``radius`` instead
makes every non-negative integer valid by construction and self-documenting:
it is directly "how many extra frames of context each side contributes",
which is also exactly the padding amount. ``radius=0`` recovers the current
mean's non-overlapping window exactly; ``radius=1`` on ``scale_factor=1``
recovers an ordinary ``kernel_size=3, stride=1, padding=1`` convolution.

Padding is per-video edge replication, not zero-padding. The two virtual
frames a zero would insert are a fixed, out-of-distribution point (the
embedding space's origin) rather than plausible content, which would dilute
the boundary windows' initial average away from the interior windows' and
leave a padding-shaped artifact for training to undo. Replicating the video's
own edge frame keeps a boundary window's initial output a convex combination
of real content -- consistent with ``NextFramePatchFusion``, which treats "no
further frames" as "assume nothing changes" rather than "assume the signal
vanishes" -- and is always computed per video, so it can never read into a
neighbouring video packed in the same batch.

The kernel is initialized to a uniform ``1/kernel_size`` and the bias to zero,
so at ``radius=0`` the very first forward pass reproduces the mean it
replaces exactly. Training is then free to learn an arbitrary per-channel
weighting of the window instead of a fixed average.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class TemporalConvDownsample(nn.Module):
    """Depthwise strided ``Conv1d`` generalizing the boundary-safe temporal mean.

    Takes packed frame features ``[sum(T), C]`` and their per-video lengths,
    exactly as the temporal mean it replaces did, and returns packed pooled
    features ``[sum(T) / scale_factor, C]``. Every video length must be
    divisible by ``scale_factor``, so a window can never span two videos.
    """

    def __init__(
        self,
        hidden_dim: int,
        scale_factor: int,
        radius: int = 0,
    ) -> None:
        super().__init__()
        _validate_dimension("hidden_dim", hidden_dim)
        _validate_dimension("scale_factor", scale_factor)
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ValueError(f"radius must be a non-negative integer, got {radius!r}")

        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        self.radius = radius
        self.kernel_size = scale_factor + 2 * radius

        # Depthwise (groups=hidden_dim): a shared window mean never mixed
        # channels either, and a dense Conv1d would cost kernel_size * C^2
        # parameters against this module's kernel_size * C -- prohibitive at
        # the 2*input_dim widths this replaces the mean under.
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=self.kernel_size,
            stride=scale_factor,
            groups=hidden_dim,
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.kernel_size)
            self.conv.bias.zero_()

    def forward(self, frame_features: Tensor, visual_length: Tensor) -> Tensor:
        """Downsample ``[sum(T), C]`` packed frame features by ``scale_factor``."""
        self._validate_inputs(frame_features, visual_length)
        video_features = torch.split(frame_features, visual_length.tolist(), dim=0)
        pooled = []
        for features in video_features:
            # [T, C] -> [1, C, T]: Conv1d convolves along the last dimension.
            window = features.t().unsqueeze(0).contiguous()
            if self.radius > 0:
                window = F.pad(window, (self.radius, self.radius), mode="replicate")
            window = self.conv(window)
            pooled.append(window.squeeze(0).t())
        return torch.cat(pooled, dim=0)

    def _validate_inputs(self, frame_features: Tensor, visual_length: Tensor) -> None:
        if frame_features.ndim != 2:
            raise ValueError(
                "frame_features must have shape [sum(T), C], got "
                f"{tuple(frame_features.shape)}"
            )
        if frame_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"frame feature dimension must be {self.hidden_dim}, got "
                f"{frame_features.shape[-1]}"
            )
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if int(visual_length.sum().item()) != frame_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )
        if bool(visual_length.remainder(self.scale_factor).ne(0).any()):
            raise ValueError(
                f"every visual length must be divisible by scale_factor "
                f"{self.scale_factor}"
            )
