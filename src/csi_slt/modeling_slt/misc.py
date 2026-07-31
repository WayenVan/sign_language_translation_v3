import torch


def random_derangement(video_lengths, device=None):
    """
    Generate a random derangement for every video in a batch.

    A derangement is a permutation with no fixed points, i.e.
    permutation[i] != i for all valid i, so that every frame/token is
    moved to a different position.

    Args:
        video_lengths: Iterable of video lengths, one per video in the batch.
            Each value is the number of frames (or tokens) of that video.
        device: Optional torch device on which the tensors are created.

    Returns:
        A 1-D tensor of concatenated derangements, one per video, in the
        same order as ``video_lengths``. Its total size is
        ``sum(video_lengths)``.
    """
    permutations = []
    for length in video_lengths:
        identity = torch.arange(length, device=device)
        permutation = identity.clone()
        while permutation.equal(identity):
            permutation = torch.randperm(length, device=device)
            permutations.append(permutation)
    permutations = torch.cat(permutations)
    return permutations


if __name__ == "__main__":
    video_lengths = [5, 3, 4]
    permutations = random_derangement(video_lengths)
    print(permutations)
