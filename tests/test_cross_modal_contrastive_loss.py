import torch
import torch.nn.functional as F

from csi_slt.modeling_slt.cross_modal_contrastive_loss import (
    CrossModalContrastiveLoss,
)


def test_text_queue_is_used_after_current_batch():
    criterion = CrossModalContrastiveLoss(
        temperature=1.0,
        learnable_temperature=False,
        gather_distributed=False,
        text_queue_size=4,
    )
    first_visual = torch.eye(2)
    first_text = torch.eye(2)

    first_loss = criterion(first_visual, first_text)
    expected_first = F.cross_entropy(torch.eye(2), torch.arange(2))
    torch.testing.assert_close(first_loss, expected_first)
    assert criterion.text_queue_count.item() == 2

    second_visual = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    second_text = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    normalized_visual = F.normalize(second_visual, dim=-1)
    normalized_text = F.normalize(second_text, dim=-1)
    queued_text = first_text
    targets = torch.arange(2)
    expected_second = 0.5 * (
        F.cross_entropy(
            normalized_visual @ torch.cat((normalized_text, queued_text)).t(),
            targets,
        )
        + F.cross_entropy(normalized_text @ normalized_visual.t(), targets)
    )

    second_loss = criterion(second_visual, second_text)
    torch.testing.assert_close(second_loss, expected_second)
    assert criterion.text_queue_count.item() == 4


def test_local_loss_targets_remain_global_with_text_queue():
    criterion = CrossModalContrastiveLoss(
        temperature=1.0,
        learnable_temperature=False,
        text_queue_size=2,
    )
    criterion._enqueue_text_features(torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))

    all_visual = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    all_text = all_visual.clone()
    local_visual = all_visual[2:]
    local_text = all_text[2:]
    targets = torch.tensor([2, 3])

    loss = criterion._symmetric_loss(
        visual_queries=local_visual,
        text_queries=local_text,
        visual_candidates=all_visual,
        text_candidates=all_text,
        targets=targets,
    )
    expected = 0.5 * (
        F.cross_entropy(
            local_visual @ torch.cat((all_text, criterion.text_queue)).t(), targets
        )
        + F.cross_entropy(local_text @ all_visual.t(), targets)
    )
    torch.testing.assert_close(loss, expected)


def test_text_queue_wraps_and_keeps_configured_size():
    criterion = CrossModalContrastiveLoss(text_queue_size=3)
    criterion._enqueue_text_features(torch.tensor([[1.0], [2.0]]))
    criterion._enqueue_text_features(torch.tensor([[3.0], [4.0]]))

    assert criterion.text_queue_count.item() == 3
    assert criterion.text_queue_ptr.item() == 1
    assert set(criterion.text_queue.squeeze(-1).tolist()) == {2.0, 3.0, 4.0}


def test_same_semantic_id_is_a_multi_positive_not_a_negative():
    criterion = CrossModalContrastiveLoss(
        temperature=1.0,
        learnable_temperature=False,
        gather_distributed=False,
    )
    features = torch.eye(3)
    semantic_ids = torch.tensor([7, 7, 9])

    loss = criterion(features, features, semantic_ids=semantic_ids)
    logits = features @ features.t()
    positive_mask = semantic_ids[:, None].eq(semantic_ids[None, :])
    expected = criterion._multi_positive_nce(logits, positive_mask)

    torch.testing.assert_close(loss, expected)
    assert loss < F.cross_entropy(logits, torch.arange(3))


def test_queue_semantic_ids_follow_feature_ring_buffer():
    criterion = CrossModalContrastiveLoss(text_queue_size=3)
    criterion._enqueue_text_features(
        torch.tensor([[1.0], [2.0]]), torch.tensor([10, 20])
    )
    criterion._enqueue_text_features(
        torch.tensor([[3.0], [4.0]]), torch.tensor([30, 40])
    )

    paired = set(
        zip(
            criterion.text_queue.squeeze(-1).tolist(),
            criterion.text_queue_ids.tolist(),
        )
    )
    assert paired == {(2.0, 20), (3.0, 30), (4.0, 40)}


def test_matching_queue_semantic_id_is_an_additional_positive():
    criterion = CrossModalContrastiveLoss(
        temperature=1.0,
        learnable_temperature=False,
        gather_distributed=False,
        text_queue_size=1,
    )
    criterion._enqueue_text_features(torch.tensor([[1.0, 0.0]]), torch.tensor([7]))
    features = torch.eye(2)
    semantic_ids = torch.tensor([7, 9])

    loss = criterion._symmetric_loss(
        visual_queries=features,
        text_queries=features,
        visual_candidates=features,
        text_candidates=features,
        targets=torch.arange(2),
        query_ids=semantic_ids,
        candidate_ids=semantic_ids,
    )
    video_logits = features @ torch.cat((features, criterion.text_queue)).t()
    video_candidate_ids = torch.tensor([7, 9, 7])
    expected = 0.5 * (
        criterion._multi_positive_nce(
            video_logits, semantic_ids[:, None].eq(video_candidate_ids[None, :])
        )
        + criterion._multi_positive_nce(
            features @ features.t(),
            semantic_ids[:, None].eq(semantic_ids[None, :]),
        )
    )

    torch.testing.assert_close(loss, expected)


def test_text_queue_is_inactive_during_evaluation():
    criterion = CrossModalContrastiveLoss(text_queue_size=2)
    criterion._enqueue_text_features(torch.eye(2))
    criterion.eval()

    queued = criterion._queued_text_features(feature_dim=2, device=torch.device("cpu"))
    assert queued.shape == (0, 2)
    criterion._enqueue_text_features(torch.ones(2, 2))
    assert criterion.text_queue_count.item() == 2


def test_empty_local_batch_returns_differentiable_zero():
    criterion = CrossModalContrastiveLoss(gather_distributed=False)
    visual = torch.empty(0, 2, requires_grad=True)
    text = torch.empty(0, 2, requires_grad=True)

    loss = criterion(visual, text)
    loss.backward()

    assert loss.item() == 0.0
    assert visual.grad is not None
    assert text.grad is not None
