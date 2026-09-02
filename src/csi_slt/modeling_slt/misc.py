import torch
from torch import nn


def packed_temporal_windows(
    packed_features: torch.Tensor,
    lengths: torch.Tensor | list[int] | tuple[int, ...],
    window_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group packed sequences into boundary-aware temporal windows.

    Window anchors are local positions ``0, stride, 2 * stride, ...``. Odd
    windows are centred on the anchor; even windows are centred halfway between
    their two middle frames. For example, sizes 2, 3, and 4 use offsets
    ``[0, 1]``, ``[-1, 0, 1]``, and ``[-1, 0, 1, 2]``. Positions outside a
    sequence are replicate-padded without crossing packed boundaries.

    Args:
        packed_features: Features shaped ``[sum(lengths), ...]``.
        lengths: Length of each packed sequence, shaped ``[B]``.
        window_size: Positive number of elements in each window.
        stride: Distance between consecutive window centres.

    Returns:
        A pair ``(windows, output_lengths)`` shaped
        ``[sum(output_lengths), window_size, ...]`` and ``[B]``.
    """
    if not isinstance(packed_features, torch.Tensor):
        raise TypeError("packed_features must be a torch.Tensor")
    if packed_features.ndim < 1:
        raise ValueError("packed_features must have shape [sum(lengths), ...]")
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError("stride must be an integer")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty 1D tensor or sequence")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        lengths_tensor = lengths.to(device=packed_features.device, dtype=torch.long)
    else:
        lengths_tensor = torch.as_tensor(lengths, device=packed_features.device)
        if lengths_tensor.ndim != 1 or lengths_tensor.numel() == 0:
            raise ValueError("lengths must be a non-empty 1D tensor or sequence")
        if lengths_tensor.is_floating_point() or lengths_tensor.is_complex():
            raise TypeError(
                f"lengths must use an integer dtype, got {lengths_tensor.dtype}"
            )
        lengths_tensor = lengths_tensor.to(dtype=torch.long)

    if bool((lengths_tensor <= 0).any()):
        raise ValueError("all temporal lengths must be positive")
    if int(lengths_tensor.sum().item()) != packed_features.shape[0]:
        raise ValueError("lengths.sum() must equal the packed feature count")
    if bool(lengths_tensor.remainder(stride).ne(0).any()):
        raise ValueError(f"all temporal lengths must be divisible by stride ({stride})")

    output_lengths = lengths_tensor // stride
    total_frames = packed_features.shape[0]
    left_radius = (window_size - 1) // 2
    offsets = torch.arange(window_size, device=packed_features.device) - left_radius
    frame_indices = torch.arange(total_frames, device=packed_features.device)

    video_ends = torch.cumsum(lengths_tensor, dim=0)
    video_ids = torch.searchsorted(video_ends, frame_indices, right=True)
    video_starts = video_ends - lengths_tensor
    local_indices = frame_indices - video_starts[video_ids]

    centre_mask = local_indices.remainder(stride).eq(0)
    centre_video_ids = video_ids[centre_mask]
    centre_local_indices = local_indices[centre_mask]
    centre_lengths = lengths_tensor[centre_video_ids]
    window_local_indices = centre_local_indices[:, None] + offsets[None, :]
    window_local_indices = torch.minimum(
        window_local_indices.clamp_min(0), centre_lengths[:, None] - 1
    )
    packed_window_indices = video_starts[centre_video_ids, None] + window_local_indices
    return packed_features[packed_window_indices], output_lengths


def packed_to_padded(
    packed_features: torch.Tensor,
    lengths: torch.Tensor | list[int] | tuple[int, ...],
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed features to a padded batch and its valid-token mask.

    Args:
        packed_features: Features shaped ``[sum(lengths), ...]``.
        lengths: Length of each sequence, shaped ``[B]``.
        padding_value: Value used to pad the feature tensor.

    Returns:
        A pair ``(padded_features, mask)`` shaped ``[B, T, ...]`` and
        ``[B, T]`` respectively. The mask uses 1 for valid positions and 0
        for padding positions.
    """
    if not isinstance(packed_features, torch.Tensor):
        raise TypeError("packed_features must be a torch.Tensor")
    if packed_features.ndim < 1:
        raise ValueError("packed_features must have shape [sum(lengths), ...]")

    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1:
            raise ValueError("lengths must be a 1D tensor or sequence")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError(f"lengths must use an integer dtype, got {lengths.dtype}")
        lengths_tensor = lengths.to(device=packed_features.device, dtype=torch.long)
    else:
        lengths_tensor = torch.as_tensor(lengths, device=packed_features.device)
        if lengths_tensor.ndim != 1:
            raise ValueError("lengths must be a 1D tensor or sequence")
        if lengths_tensor.numel() and (
            lengths_tensor.is_floating_point() or lengths_tensor.is_complex()
        ):
            raise TypeError(
                f"lengths must use an integer dtype, got {lengths_tensor.dtype}"
            )
        lengths_tensor = lengths_tensor.to(dtype=torch.long)

    if bool((lengths_tensor < 0).any()):
        raise ValueError("lengths must be non-negative")
    if int(lengths_tensor.sum().item()) != packed_features.shape[0]:
        raise ValueError("lengths.sum() must equal the packed feature count")

    batch_size = lengths_tensor.numel()
    max_length = int(lengths_tensor.max().item()) if batch_size else 0
    positions = torch.arange(max_length, device=packed_features.device)
    mask = positions.unsqueeze(0) < lengths_tensor.unsqueeze(1)
    padded = packed_features.new_full(
        (batch_size, max_length, *packed_features.shape[1:]), padding_value
    )
    padded[mask] = packed_features
    return padded, mask.long()


def packed_position_ids_to_padded(
    packed_position_ids: torch.Tensor,
    lengths: torch.Tensor | list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad packed visual position IDs with the ``-100`` sentinel.

    Visual adapters emit one-dimensional IDs aligned with packed visual tokens
    ``[sum(lengths)]``. Consumers such as batched alignment require
    ``[B, max(lengths)]`` plus a mask; this helper is the explicit boundary
    between those two representations.
    """
    if not isinstance(packed_position_ids, torch.Tensor):
        raise TypeError("packed_position_ids must be a torch.Tensor")
    if packed_position_ids.ndim != 1:
        raise ValueError("packed_position_ids must have shape [sum(lengths)]")
    if packed_position_ids.is_floating_point() or packed_position_ids.is_complex():
        raise TypeError("packed_position_ids must use an integer dtype")
    return packed_to_padded(
        packed_position_ids,
        lengths,
        padding_value=-100,
    )


def padded_to_packed(
    padded_features: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove padding from a feature batch using a valid-token mask.

    Args:
        padded_features: Features shaped ``[B, T, ...]``.
        mask: Tensor shaped ``[B, T]``, with 1 for valid positions and 0 for
            padding positions.

    Returns:
        A pair ``(packed_features, lengths)`` shaped ``[sum(lengths), ...]``
        and ``[B]`` respectively. Valid features retain their batch-major,
        temporal order.
    """
    if not isinstance(padded_features, torch.Tensor):
        raise TypeError("padded_features must be a torch.Tensor")
    if padded_features.ndim < 2:
        raise ValueError("padded_features must have shape [B, T, ...]")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(padded_features.shape[:2]):
        raise ValueError("mask shape must equal padded_features.shape[:2]")
    if mask.device != padded_features.device:
        raise ValueError("mask and padded_features must be on the same device")
    valid_values = (mask == 0) | (mask == 1)
    if not bool(valid_values.all()):
        raise ValueError("mask must contain only 0 (padding) and 1 (valid)")

    valid_mask = mask.bool()
    lengths = valid_mask.sum(dim=1, dtype=torch.long)
    packed = padded_features[valid_mask]
    return packed, lengths


def mark_module_tree_as_initialized(module: nn.Module) -> None:
    """Mark a module tree as initialized for Hugging Face model composition.

    Use this after a module has completed either custom initialization or
    checkpoint loading. ``PreTrainedModel.post_init`` respects the
    ``_is_hf_initialized`` marker and leaves the complete module tree
    unchanged when it is later attached to an outer model.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")

    for submodule in module.modules():
        submodule._is_hf_initialized = True


def random_derangement(video_lengths, device=None):
    """Derange frames independently within every packed video.

    The returned indices never map a frame to its original position.  A
    one-frame video cannot be deranged, so it is rejected explicitly instead
    of silently returning an unchanged frame.
    """
    lengths = (
        video_lengths.tolist() if torch.is_tensor(video_lengths) else video_lengths
    )
    permutations = []
    offset = 0
    for length in lengths:
        if length < 0:
            raise ValueError(f"video lengths must be non-negative, got {length}")
        if length == 1:
            raise ValueError("a video with one frame cannot be deranged")
        if length == 0:
            continue

        identity = torch.arange(length, device=device)
        permutation = torch.randperm(length, device=device)
        while torch.any(permutation == identity):
            permutation = torch.randperm(length, device=device)
        permutations.append(permutation + offset)
        offset += length

    if not permutations:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(permutations)


class SpatialDropoutMean(nn.Module):
    """Mean over a frame's patches, with patches randomly dropped in training.

    ``[F, P, D] -> [F, D]``.  At ``p = 0`` and in eval this is exactly
    ``patch_features.mean(dim=1)``, so it is a drop-in replacement that changes
    nothing until it is switched on.

    The point is not generic regularization.  A plain spatial mean gives every
    patch a fixed weight of 1/P, which lets a handful of always-present patches
    -- the backdrop, a station logo, the signer's own face -- act as a
    fingerprint the model can key on instead of reading the signing.  Dropping
    patches makes any individual one unreliable, so a representation that
    depends on a specific few stops paying off.

    Two details that are wrong in the obvious implementation:

    * **Renormalize by the survivors, not by P.**  Scaling by 1/P after dropping
      would shrink every training feature while eval kept full scale, and that
      train/eval mismatch produces no error at all -- only a worse dev number
      with no obvious cause.
    * **Draw the mask per frame**, not once per batch, or every frame in a step
      loses the same regions and the noise stops being independent.

    A caveat about when this fires: the module follows its parent's ``training``
    flag, and ``SltModel.train()`` puts the visual adapter in train mode only
    when the trainability plan sets ``visual_adapter.runtime_mode = "train"``.
    That field defaults to ``"eval"``, so a plan that leaves it alone silently
    disables this everywhere.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError(f"p must be a float, got {type(p).__name__}")
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.ndim != 3:
            raise ValueError(
                "patch_features must have shape [F, P, D], got "
                f"{tuple(patch_features.shape)}"
            )
        if self.p == 0.0 or not self.training:
            return patch_features.mean(dim=1)

        draw = torch.rand(patch_features.shape[:2], device=patch_features.device)
        keep = draw >= self.p
        # Whatever the draw, keep the frame's largest sample: at P=196 and a
        # sane p an all-dropped frame is astronomically unlikely, but it would
        # divide by zero rather than fail loudly, and an ablation at p=0.9 makes
        # it merely unlikely.
        keep.scatter_(1, draw.argmax(dim=1, keepdim=True), True)

        weights = keep.unsqueeze(-1).to(dtype=patch_features.dtype)
        return (patch_features * weights).sum(dim=1) / weights.sum(dim=1)

    def extra_repr(self) -> str:
        return f"p={self.p}"


if __name__ == "__main__":
    video_lengths = [5, 3, 4]
    permutations = random_derangement(video_lengths)
    print(permutations)
