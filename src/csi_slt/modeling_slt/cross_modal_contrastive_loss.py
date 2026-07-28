"""Visual-textual CLIP-style contrastive loss."""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as all_gather_with_grad


class CrossModalContrastiveLoss(nn.Module):
    """用于一一配对视觉、文本样本的 CLIP 风格双向对比损失。

    一个 batch 中下标相同的视觉样本与文本样本构成正样本对，例如
    ``visual_features[i]`` 与 ``text_features[i]``；同 batch 的其余配对
    自动作为负样本。输入特征会先按 mask 或 lengths 做平均池化（如有必要），
    再进行 L2 归一化、计算余弦相似度和双向交叉熵。

    对变长视频，推荐传入打包（packed）特征 ``[sum(Lv_i), D]`` 与
    ``visual_lengths=[Lv_1, ..., Lv_B]``。该路径不会构造 padding，也不会
    生成 ``[B, max(Lv), D]`` 的中间张量。

    分布式训练初始化后（DDP），默认仍只使用本 rank 的 batch 计算损失，
    避免在模型 forward 中隐式执行 collective。若需要跨 rank 负样本，可通过
    ``gather_distributed=True`` 显式开启。

    Example:
        >>> criterion = CrossModalContrastiveLoss()
        >>> loss = criterion(visual_features, text_features,
        ...                  visual_lengths=visual_lengths)

    Args:
        temperature: 初始温度 ``tau``，为正标量。相似度会乘以 ``1 / tau``；
            较小的温度会使正负样本区分更尖锐。默认 ``0.07``。
        learnable_temperature: 是否将温度对应的缩放系数设为可训练参数。
        max_logit_scale: 缩放系数 ``exp(logit_scale)`` 的上界，防止训练时
            温度过小造成数值不稳定。默认 ``100``。
        local_loss: DDP 下是否只计算本 rank 的 query 对全局候选的损失，
            默认为 ``True``，可避免在每个 rank 重复构造完整的全局 logits，
            并将 logits 显存占用从 ``O(B_global^2)`` 降至
            ``O(B_local * B_global)``。

            WARN: 设为 ``False`` 时，每个 rank 都会重复计算相同的全局损失。
            除了显著增加显存和通信开销，配合 ``gather_with_grad=True`` 使用
            时，特征编码器经过 autograd all-gather 得到的梯度与普通 DDP
            参数（例如 ``logit_scale``）还可能具有不同的 world-size 缩放。
            除非明确需要完整全局 logits 并已验证梯度缩放，否则不建议关闭。
        gather_with_grad: 当 ``gather_distributed=True`` 时，是否使用支持
            autograd 的跨卡 all-gather，默认为 ``False``。关闭时仍会收集
            其他 rank 的特征作为负样本，但会 detach 远端特征，同时保留
            本 rank 特征的梯度路径。

            WARN: 设为 ``True`` 会在 backward 中引入额外的分布式
            collective。所有 rank 必须以完全相同的顺序和次数执行该损失的
            forward 与 backward；如果某个 rank 跳过 batch、提前异常、
            DataLoader 步数不同或条件分支不一致，其余 rank 可能永久等待，
            表现为训练卡死。启用前请确认各 rank 的训练控制流严格一致；排查
            问题时建议开启 ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` 与
            ``NCCL_ASYNC_ERROR_HANDLING=1``。
        gather_distributed: DDP 下是否跨 rank 收集特征作为全局负样本，
            默认为 ``False``。关闭时每个 rank 独立计算本地 batch 的对比
            损失，梯度仍会由 DDP 正常同步，不会在此损失中执行额外的
            collective。

            WARN: 设为 ``True`` 后，所有 rank 必须在每一步以相同顺序调用
            此损失，否则即使 ``gather_with_grad=False``，前向
            ``dist.all_gather`` 仍可能永久等待。只有在确认各 rank 的
            DataLoader 步数、条件分支和 batch 执行次数完全一致时才应开启。
        process_group: 可选的 PyTorch 分布式进程组；默认使用全局进程组。
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_logit_scale: float = 100.0,
        local_loss: bool = True,
        gather_with_grad: bool = False,
        gather_distributed: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if max_logit_scale <= 0:
            raise ValueError("max_logit_scale must be greater than zero")

        # 保存 log(1 / temperature)，训练/前向时经 exp() 转回缩放系数。
        initial_logit_scale = torch.tensor(1.0 / temperature).log()
        if learnable_temperature:
            self.logit_scale = nn.Parameter(initial_logit_scale)
        else:
            self.register_buffer("logit_scale", initial_logit_scale)
        self.max_logit_scale = max_logit_scale
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.gather_distributed = gather_distributed
        self.process_group = process_group

    def _gather_batch_sizes(
        self, local_batch_size: int, device: torch.device
    ) -> Tuple[int, ...]:
        """收集每个 rank 的 batch 大小，仅执行一次普通 all-gather。"""
        local_size = torch.tensor([local_batch_size], device=device)
        world_size = dist.get_world_size(group=self.process_group)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size, group=self.process_group)
        return tuple(size.item() for size in gathered_sizes)

    def _gather_features(
        self,
        features: torch.Tensor,
        batch_sizes: Tuple[int, ...],
        rank: int,
    ) -> torch.Tensor:
        """跨 rank 收集 ``[local_batch, D]`` 特征，并保留梯度路径。

        ``batch_sizes`` 已由调用方收集。这里仅将特征临时补齐到最大本地
        batch，all-gather 后移除补齐部分，因此支持不同 rank 的 batch 大小。
        """
        max_batch_size = max(batch_sizes)
        if max_batch_size == 0:
            raise ValueError("distributed batch must contain at least one sample")

        padded_features = F.pad(features, (0, 0, 0, max_batch_size - features.shape[0]))
        if self.gather_with_grad:
            # 与 OpenCLIP 的 gather_with_grad 路径一致：backward 会将来自所有
            # rank 的梯度归约回本地特征。
            gathered_features = all_gather_with_grad(
                padded_features, group=self.process_group
            )
        else:
            # 省显存模式：远端特征不保留图，但本 rank 特征仍保持可导。
            gathered_features = [torch.zeros_like(padded_features) for _ in batch_sizes]
            dist.all_gather(
                gathered_features, padded_features, group=self.process_group
            )
            gathered_features[rank] = padded_features

        return torch.cat(
            [part[:size] for part, size in zip(gathered_features, batch_sizes)], dim=0
        )

    @staticmethod
    def _pool_packed(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """对 ``[sum(lengths), D]`` 的打包序列做高效分段平均池化。

        ``repeat_interleave`` 只生成一个长度为真实 token 总数的一维索引，
        ``index_add_`` 直接累加到 ``[B, D]``，因此不会引入 padding 开销。

        Args:
            features: 连续拼接的序列特征，形状 ``[sum(L_i), D]``。
            lengths: 每个样本在 ``features`` 中的连续片段长度，形状 ``[B]``。
                所有值必须为正整数，且 ``sum(lengths) == features.shape[0]``。

        Returns:
            按每段平均后的特征，形状 ``[B, D]``。
        """
        if features.ndim != 2:
            raise ValueError("packed features must have shape [sum(lengths), dim]")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape [batch]")
        if lengths.numel() == 0:
            raise ValueError("lengths must contain at least one sample")
        if torch.is_floating_point(lengths) or torch.is_complex(lengths):
            raise ValueError("lengths must contain integer values")

        lengths = lengths.to(device=features.device, dtype=torch.long)
        if torch.any(lengths <= 0):
            raise ValueError("every packed sequence must contain at least one token")
        if lengths.sum().item() != features.shape[0]:
            raise ValueError("sum(lengths) must equal the packed feature length")

        batch_indices = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=features.device), lengths
        )
        pooled_features = features.new_zeros((lengths.numel(), features.shape[-1]))
        pooled_features.index_add_(0, batch_indices, features)
        return pooled_features / lengths.to(features.dtype).unsqueeze(-1)

    @classmethod
    def _pool(
        cls,
        features: torch.Tensor,
        mask: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """将序列特征池化为单个样本向量。

        Args:
            features: 特征张量。可以是已池化的 ``[B, D]``、补齐后的序列特征
                ``[B, L, D]``，或拼接后的打包序列 ``[sum(L_i), D]``；后者
                必须同时提供 ``lengths``。``B`` 是 batch 大小，``L`` 是序列
                长度，``D`` 是视觉和文本共用的特征维度。
            mask: 可选的布尔/0-1 张量，形状为 ``[B, L]``。``True``/1 表示
                对应位置有效，``False``/0 表示 padding。若为 ``None``，序列
                所有位置都会参与平均池化；当 ``features`` 为 ``[B, D]`` 时
                此参数会被忽略。
            lengths: 打包序列中每个样本的长度，形状 ``[B]``。传入时
                ``features`` 必须是 ``[sum(L_i), D]``，且不能同时传 ``mask``。

        Returns:
            池化后的特征，形状为 ``[B, D]``。
        """
        if lengths is not None:
            if mask is not None:
                raise ValueError("mask and lengths cannot be used together")
            return cls._pool_packed(features, lengths)
        if features.ndim == 2:
            return features
        if features.ndim != 3:
            raise ValueError(
                "features must have shape [batch, dim] or [batch, length, dim]"
            )
        if mask is None:
            return features.mean(dim=1)
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [batch, length]")

        weights = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
        valid_token_counts = weights.sum(dim=1)
        if torch.any(valid_token_counts == 0):
            raise ValueError("every sample must contain at least one valid token")
        return (features * weights).sum(dim=1) / valid_token_counts

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算视觉—文本双向对比损失。

        Args:
            visual_features: 每个样本的视觉特征，可为 ``[B, D]``、``[B, Lv, D]``
                或打包形式 ``[sum(Lv_i), D]``。最后一种形式必须配合
                ``visual_lengths`` 使用，适合未 padding 的变长视频；第 ``i``
                个视频必须和第 ``i`` 个文本表示同一个语义样本。
            text_features: 每个样本的文本特征，可为 ``[B, D]``、``[B, Lt, D]``
                或打包形式 ``[sum(Lt_i), D]``。其池化后的 batch 大小和特征维
                ``D`` 必须与视觉侧一致。
            visual_mask: 视觉序列的有效位置掩码，形状 ``[B, Lv]``；仅当
                ``visual_features`` 为三维时生效。``True`` 表示有效帧/token。
            text_mask: 文本序列的有效位置掩码，形状 ``[B, Lt]``；仅当
                ``text_features`` 为三维时生效。通常可直接传 tokenizer 的
                ``attention_mask``。
            visual_lengths: 打包视频的帧/token 长度，形状 ``[B]``，例如
                ``tensor([length1, length2, length3])``。仅适用于
                ``visual_features=[length1+length2+length3, D]``，不能与
                ``visual_mask`` 同时使用。
            text_lengths: 打包文本的 token 长度，形状 ``[B]``。使用规则与
                ``visual_lengths`` 相同；常规 tokenizer 输出通常使用
                ``text_mask`` 即可。

        Returns:
            标量张量 ``[]``：text-to-visual 与 visual-to-text 交叉熵的平均值。
        """
        visual_embeddings = F.normalize(
            self._pool(visual_features, visual_mask, visual_lengths), dim=-1
        )
        text_embeddings = F.normalize(
            self._pool(text_features, text_mask, text_lengths), dim=-1
        )
        if visual_embeddings.shape[0] == 0:
            raise ValueError("batch must contain at least one pair")
        if visual_embeddings.shape != text_embeddings.shape:
            raise ValueError("pooled visual and text features must have the same shape")

        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        if (
            not self.gather_distributed
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            # [B, D] @ [D, B] -> [B, B]，第 i 行、第 i 列即第 i 个正样本对。
            similarity = text_embeddings @ visual_embeddings.t() * scale
            targets = torch.arange(similarity.shape[0], device=similarity.device)
            return (
                F.cross_entropy(similarity, targets)
                + F.cross_entropy(similarity.t(), targets)
            ) / 2.0

        rank = dist.get_rank(group=self.process_group)
        batch_sizes = self._gather_batch_sizes(
            visual_embeddings.shape[0], visual_embeddings.device
        )
        all_visual_embeddings = self._gather_features(
            visual_embeddings, batch_sizes, rank
        )
        all_text_embeddings = self._gather_features(text_embeddings, batch_sizes, rank)

        if self.local_loss:
            # 仅计算本 rank 的 B_local 行，显存从 O(B_global^2) 降到
            # O(B_local * B_global)。正样本列从此前 rank 的样本数开始。
            positive_offset = sum(batch_sizes[:rank])
            targets = (
                torch.arange(
                    visual_embeddings.shape[0], device=visual_embeddings.device
                )
                + positive_offset
            )
            text_to_visual_logits = text_embeddings @ all_visual_embeddings.t() * scale
            visual_to_text_logits = visual_embeddings @ all_text_embeddings.t() * scale
        else:
            # 每张 rank 都计算相同的完整全局 logits，数值与标准全局 CLIP 一致。
            targets = torch.arange(
                all_visual_embeddings.shape[0], device=visual_embeddings.device
            )
            text_to_visual_logits = (
                all_text_embeddings @ all_visual_embeddings.t() * scale
            )
            visual_to_text_logits = text_to_visual_logits.t()

        return (
            F.cross_entropy(text_to_visual_logits, targets)
            + F.cross_entropy(visual_to_text_logits, targets)
        ) / 2.0


def _run_local_debug_test() -> None:
    """验证 packed 输入与等价的 padded 输入一致，且均可反传。"""
    torch.manual_seed(42)
    feature_dim = 8
    video_lengths = torch.tensor([2, 3, 1])
    batch_size = video_lengths.numel()

    # 真实业务中的视频输入：所有样本沿第 0 维连续拼接。
    packed_video_features = torch.randn(
        video_lengths.sum(), feature_dim, requires_grad=True
    )
    text_features = torch.randn(batch_size, 4, feature_dim, requires_grad=True)
    text_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool
    )

    criterion = CrossModalContrastiveLoss(learnable_temperature=False)
    packed_loss = criterion(
        visual_features=packed_video_features,
        text_features=text_features,
        visual_lengths=video_lengths,
        text_mask=text_mask,
    )
    packed_loss.backward()
    assert packed_video_features.grad is not None
    assert text_features.grad is not None

    # 将同一批视频手动补齐，用于检查两种输入格式的数值一致性。
    max_video_length = video_lengths.max().item()
    padded_video_features = torch.zeros(batch_size, max_video_length, feature_dim)
    offset = 0
    for sample_index, length in enumerate(video_lengths.tolist()):
        padded_video_features[sample_index, :length] = packed_video_features.detach()[
            offset : offset + length
        ]
        offset += length
    video_mask = torch.arange(max_video_length).unsqueeze(0) < video_lengths.unsqueeze(
        1
    )
    padded_loss = criterion(
        visual_features=padded_video_features,
        text_features=text_features.detach(),
        visual_mask=video_mask,
        text_mask=text_mask,
    )
    assert torch.allclose(packed_loss.detach(), padded_loss, atol=1e-6)

    print("Packed sequence test passed")
    print(f"packed video shape: {tuple(packed_video_features.shape)}")
    print(f"video lengths: {video_lengths.tolist()}")
    print(f"contrastive loss: {packed_loss.item():.6f}")


def _run_distributed_debug_test() -> None:
    """验证 DDP all-gather（含不同 rank batch 大小）和反向传播。"""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    try:
        rank = dist.get_rank()
        # 故意让不同 rank 的 batch 大小不同，以验证 gather 的 padding/crop 路径。
        local_batch_size = rank + 2
        visual_features = torch.randn(
            local_batch_size, 8, device=device, requires_grad=True
        )
        text_features = torch.randn(
            local_batch_size, 8, device=device, requires_grad=True
        )
        criterion = CrossModalContrastiveLoss(
            learnable_temperature=False,
            local_loss=True,
            gather_with_grad=True,
            gather_distributed=True,
        ).to(device)
        loss = criterion(visual_features, text_features)
        loss.backward()
        assert visual_features.grad is not None
        assert text_features.grad is not None
        dist.barrier()
        if rank == 0:
            print("DDP autograd all-gather test passed")
            print(
                f"world size: {dist.get_world_size()}, loss on rank 0: {loss.item():.6f}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 单进程：python spamo/losses/cross_modal_contrastive.py
    # DDP：torchrun --standalone --nproc_per_node=2 \
    #          spamo/losses/cross_modal_contrastive.py
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_distributed_debug_test()
    else:
        _run_local_debug_test()
