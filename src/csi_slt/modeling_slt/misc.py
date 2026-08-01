import torch


def random_derangement(video_lengths, device=None):
    """Derange frames independently within every packed video.

    The returned indices never map a frame to its original position.  A
    one-frame video cannot be deranged, so it is rejected explicitly instead
    of silently returning an unchanged frame.
    """
    lengths = video_lengths.tolist() if torch.is_tensor(video_lengths) else video_lengths
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


if __name__ == "__main__":
    video_lengths = [5, 3, 4]
    permutations = random_derangement(video_lengths)
    print(permutations)
